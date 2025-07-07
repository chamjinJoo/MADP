from __future__ import annotations
from pathlib import Path
from typing import Tuple, Optional, Sequence, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
# ---------------------------------------------------------------------------
# GPU 최적화를 위한 추가 import
# ---------------------------------------------------------------------------
from contextlib import nullcontext

# ---------------------------------------------------------------------------
# GPU 최적화 헬퍼 함수들
# ---------------------------------------------------------------------------

def to_gpu_optimized(tensor, device=None):
    """GPU 최적화된 텐서 변환"""
    if device is not None and device.type == 'cuda':
        if tensor.is_contiguous():
            return tensor.pin_memory().to(device, non_blocking=True)
        else:
            return tensor.to(device)
    else:
        return tensor.to(device) if device is not None else tensor

def batch_forward(model, inputs, device=None):
    """배치 전방 전파 최적화"""
    if device is not None and device.type == 'cuda':
        # GPU에서 배치 처리 최적화
        with torch.cuda.amp.autocast(enabled=True):
            return model(inputs)
    else:
        return model(inputs)

# ---------------------------------------------------------------------------
# 1.  VRNN building blocks
# ---------------------------------------------------------------------------
class VRNNCell(nn.Module):
    """GPU 최적화된 Single‑step Variational RNN cell (Chung et al., 2015).

    Prior, encoder, decoder are simple MLPs. Replace with CNNs/TCNs for images.
    """
    LOGVAR_CLAMP = 10.0

    def __init__(
        self, 
        o_dim: int,
        a_dim: int, 
        h_dim: int, 
        z_dim: int, 
    ) -> None:
        super().__init__()
        self.o_dim = o_dim
        self.z_dim = z_dim

        # RNN core
        self.rnn = nn.GRUCell(o_dim + a_dim + z_dim, h_dim)
        
        # Prior p(z_t | h_{t-1})
        self.prior_net = nn.Sequential(
            nn.Linear(h_dim, h_dim), nn.ReLU(), 
            nn.Linear(h_dim, z_dim * 2)
        )
        # Encoder q(z_t | x_t, h_{t-1})
        self.enc_net = nn.Sequential(
            nn.Linear(o_dim + h_dim, h_dim), nn.ReLU(), 
            nn.Linear(h_dim, z_dim * 2)
        )
        # Decoder p(x_t | z_t, h_{t-1}) — predicts μ and log σ² (diagonal)
        out_dim = o_dim * 2
        self.dec_net = nn.Sequential(
            nn.Linear(z_dim + h_dim, h_dim), nn.ReLU(),
            nn.Linear(h_dim, out_dim)
        )

    def _split_mu_logvar(self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mu, logvar = t.chunk(2, dim=-1)
        return mu, logvar

    def forward(
        self,
        x_t: torch.Tensor,
        a_prev: torch.Tensor,
        h_prev: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """One VRNN step.
        
        Returns
        -------
        h_t : next hidden state
        recon_x : decoder output
        kl : KL divergence at this step (no beta factor)
        z_t : latent sample
        mu_q, logvar_q : encoder stats (useful for diagnostics)
        """
        # Ensure tensors are float32 for consistency (only if needed)
        if x_t.dtype != torch.float32:
            x_t = x_t.to(torch.float32)
        if a_prev.dtype != torch.float32:
            a_prev = a_prev.to(torch.float32)
        if h_prev.dtype != torch.float32:
            h_prev = h_prev.to(torch.float32)

        # Prior h_{t-1} -> z_t 
        prior_stats = self.prior_net(h_prev)
        mu_p, logvar_p = self._split_mu_logvar(prior_stats)

        # Encoder (posterior) x_t, h_{t-1} -> z_t
        enc_input = torch.cat([x_t, h_prev], dim=-1)
        enc_stats = self.enc_net(enc_input)
        mu_q, logvar_q = self._split_mu_logvar(enc_stats)
        logvar_q = torch.clamp(logvar_q, min=-self.LOGVAR_CLAMP, max=self.LOGVAR_CLAMP)

        # Reparameterisation
        std_q = (0.5 * logvar_q).exp()
        eps = torch.randn_like(std_q)
        z_t = mu_q + eps * std_q # latent variable  

        # Decoder (reconstruction) z_t, h_{t-1} -> x_t 
        dec_out = self.dec_net(torch.cat([z_t, h_prev], dim=-1))
        mu_x, logvar_x = self._split_mu_logvar(dec_out)
        logvar_x = torch.clamp(logvar_x, min=-self.LOGVAR_CLAMP, max=self.LOGVAR_CLAMP)
        # NLL (Gaussian, independent dims), -log⁡[𝑝_(𝜃_𝑖)(𝑜│𝑝)]
        inv_var = (-logvar_x).exp()
        nll = 0.5 * ((x_t - mu_x).pow(2) * inv_var + logvar_x).sum(dim=-1)
        
        # KL divergence analytically (Normal) D_KL(q||p)
        var_p = logvar_p.exp()
        var_q = logvar_q.exp()
        kl = 0.5 * (
            ((mu_q - mu_p).pow(2) / var_p)
            + (var_q / var_p)
            + (logvar_p - logvar_q)
            - 1.0
        ).sum(dim=-1)
        # GRU update
        rnn_in = torch.cat([x_t, z_t, a_prev], dim=-1)
        h_t = self.rnn(rnn_in, h_prev)
        
        return h_t, nll, kl, z_t, mu_q, logvar_q

# ---------------------------------------------------------------------------
# 2.  Decentralized Attention Communication Module
# ---------------------------------------------------------------------------

class AttentionComm(nn.Module):
    """
    Fully decentralized attention without parameter sharing across agents.
    Each agent has its own Q, K, V projections and computes inference auxiliary loss.
    """
    def __init__(self, num_agents: int, z_dim: int, d_model: int, infer_hidden=128):
        super().__init__()
        self.num_agents = num_agents
        # separate projections per agent
        self.Wq_list = nn.ModuleList([nn.Linear(z_dim, d_model, bias=False) for _ in range(num_agents)])
        self.Wk_list = nn.ModuleList([nn.Linear(z_dim, d_model, bias=False) for _ in range(num_agents)])
        self.Wv_list = nn.ModuleList([nn.Linear(z_dim, d_model, bias=False) for _ in range(num_agents)])

        # auxiliary inference head: reconstruct z_i from message
        self.infer_head = nn.Linear(d_model, z_dim)
        self.scale = d_model ** 0.5

    def forward(self, z_list):
        assert len(z_list) == self.num_agents, "Mismatch in number of agents"
        N = self.num_agents
        # compute per-agent Q, K, V
        Q = [self.Wq_list[i](z_list[i]) for i in range(N)]
        K = [self.Wk_list[i](z_list[i]) for i in range(N)]
        V = [self.Wv_list[i](z_list[i]) for i in range(N)]

        messages = []
        infer_losses = []
        for j in range(N):
            qj = Q[j]  # (d_model,)
            # gather keys/values from all other agents
            other_idx = [i for i in range(N) if i != j]
            if len(other_idx) == 0:  # 단일 에이전트인 경우
                messages.append(torch.zeros_like(qj))
                continue
                
            keys = torch.stack([K[i] for i in other_idx], dim=0)  # (num_others, d_model)
            vals = torch.stack([V[i] for i in other_idx], dim=0)  # (num_others, d_model)

            # Attention scores: (num_others, d_model) @ (d_model,) -> (num_others,)
            scores = torch.matmul(keys, qj) / self.scale
            alpha = F.softmax(scores, dim=0)  # (num_others,)

            # weighted sum for message: (num_others,) * (num_others, d_model) -> (d_model,)
            m_j = torch.sum(alpha.unsqueeze(-1) * vals, dim=0)
            messages.append(m_j)

            # auxiliary inference loss: reconstruct each neighbor's z
            for idx, i in enumerate(other_idx):
                hat_z = self.infer_head(vals[idx])  # vals[idx] is (d_model,)
                infer_losses.append(F.mse_loss(hat_z, z_list[i]))

        total_infer_loss = torch.stack(infer_losses).mean() if infer_losses else torch.full((), 0.0, device=z_list[0].device, dtype=torch.float32)
        return messages, total_infer_loss

class ActorCriticHead(nn.Module):
    """GPU 최적화된 Actor-Critic 헤드"""
    def __init__(self, in_dim: int, act_dim: int) -> None:
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.ReLU(),
            nn.Linear(in_dim // 2, act_dim)
        )
        self.critic = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.ReLU(),
            nn.Linear(in_dim // 2, 1)
        )

    def forward(self, input: torch.Tensor):
        return self.actor(input), self.critic(input)

class VRNNDecentralizedA2C(nn.Module):
    """GPU 최적화된 VRNN + Decentralized Attention + Actor-Critic 모델"""
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dim: int,
        z_dim: int,
        n_agents: int,
        d_model: int,
        use_attention: bool = True,  # Attention 사용 옵션
        use_rnn: bool = True,  # RNN 사용 여부 옵션
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_dim = hidden_dim
        self.z_dim = z_dim
        self.nagents = n_agents
        self.d_model = d_model
        self.use_attention = use_attention
        self.use_rnn = use_rnn

        # VRNN cells for each agent
        self.vrnn_cells = nn.ModuleList([
            VRNNCell(obs_dim, act_dim, hidden_dim, z_dim)
            for _ in range(n_agents)
        ])

        # Attention Communication module
        if use_attention:
            self.attention_module = AttentionComm(n_agents, z_dim, d_model)
        else:
            self.attention_module = None

        # Actor-Critic heads
        if use_rnn:
            input_dim = z_dim + (d_model if use_attention else 0)
        else:
            input_dim = z_dim
        
        self.actor_critic = ActorCriticHead(input_dim, act_dim)

    def forward_step(
        self,
        obs: torch.Tensor,      # (N, obs_dim)
        a_prev: torch.Tensor,   # (N, act_dim)
        h_prev: torch.Tensor,   # (N, hidden_dim)
        rolling_mean_error: Optional[torch.Tensor] = None,  # (N,) or None
    ) -> tuple:
        N = obs.size(0)
        device = obs.device
        
        # VRNN forward pass for each agent
        h_new_list = []
        nll_list = []
        kl_list = []
        z_list = []
        mu_list = []
        logvar_list = []
        
        for i in range(N):
            h_new, nll, kl, z, mu, logvar = self.vrnn_cells[i](
                obs[i], a_prev[i], h_prev[i]
            )
            h_new_list.append(h_new)
            nll_list.append(nll)
            kl_list.append(kl)
            z_list.append(z)
            mu_list.append(mu)
            logvar_list.append(logvar)
        
        h_new = torch.stack(h_new_list, dim=0)  # (N, hidden_dim)
        nll = torch.stack(nll_list, dim=0)      # (N,)
        kl = torch.stack(kl_list, dim=0)        # (N,)
        z = torch.stack(z_list, dim=0)          # (N, z_dim)
        mu = torch.stack(mu_list, dim=0)        # (N, z_dim)
        logvar = torch.stack(logvar_list, dim=0) # (N, z_dim)

        # Decentralized Attention processing
        attention_output = None
        infer_loss = torch.tensor(0.0, device=device)
        if self.use_attention and self.attention_module is not None:
            # Convert to list format for attention module
            z_list_for_attention = [z[i] for i in range(N)]
            messages, infer_loss = self.attention_module(z_list_for_attention)
            attention_output = torch.stack(messages, dim=0)  # (N, d_model)

        # Actor-Critic forward pass
        if self.use_rnn:
            if self.use_attention and attention_output is not None:
                ac_input = torch.cat([z, attention_output], dim=-1)
            else:
                ac_input = z
        else:
            if self.use_attention and attention_output is not None:
                ac_input = torch.cat([obs, attention_output], dim=-1)
            else:
                ac_input = obs

        logits, values = self.actor_critic(ac_input)

        return h_new, nll, kl, logits, None, values, mu, logvar, z, attention_output, infer_loss

# ---------------------------------------------------------------------------
# Loss functions (GPU 최적화)
# ---------------------------------------------------------------------------



def kl_gauss(mu1, logvar1, mu2, logvar2):
    """GPU 최적화된 Gaussian KL divergence"""
    return 0.5 * (
        (mu1 - mu2).pow(2) / torch.exp(logvar2) +
        torch.exp(logvar1) / torch.exp(logvar2) +
        logvar2 - logvar1 - 1.0
    ).sum(dim=-1)

def compute_vae_loss(
    nll: torch.Tensor,      # (B,)
    kl: torch.Tensor,       # (B,)
    nll_coef: float,
    kl_coef: float,
) -> tuple[torch.Tensor, dict]:
    """GPU 최적화된 VAE loss 계산"""
    # Basic VAE loss
    loss_nll = nll.mean()
    loss_kl = kl.mean()
    
    # Total loss
    total_loss = nll_coef * loss_nll + kl_coef * loss_kl
    
    metrics = {
        'vae_nll': loss_nll.item(),
        'vae_kl': loss_kl.item(),
        'loss_vae': total_loss.item()
    }
    
    return total_loss, metrics

def compute_rl_loss(logits: torch.Tensor,
                    actions: torch.Tensor,
                    adv: torch.Tensor,
                    values_pred: torch.Tensor,
                    returns: torch.Tensor,
                    ent_coef: float,
                    value_coef: float
)->tuple[torch.Tensor, dict]:
    """GPU 최적화된 RL loss 계산"""
    # 1) Policy loss
    log_probs = F.log_softmax(logits, dim=-1)
    action_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
    policy_loss = -(action_log_probs * adv).mean()
    
    # 2) Value loss
    value_loss = F.mse_loss(values_pred, returns)
    
    # 3) Entropy bonus
    entropy = -(log_probs * torch.exp(log_probs)).sum(dim=-1).mean()
    
    # 4) Total loss
    total_loss = policy_loss + value_coef * value_loss - ent_coef * entropy
    
    metrics = {
        'policy_loss': policy_loss.item(),
        'value_loss': value_loss.item(),
        'entropy': entropy.item(),
        'loss_rl': total_loss.item()
    }
    
    return total_loss, metrics

# ---------------------------------------------------------------------------
# VAE-only cell (RNN 없이)
# ---------------------------------------------------------------------------
class VAECell(nn.Module):
    """GPU 최적화된 RNN 없이 VAE만 수행하는 셀."""
    LOGVAR_CLAMP = 10.0

    def __init__(self, o_dim: int, a_dim: int, h_dim: int, z_dim: int) -> None:
        super().__init__()
        self.o_dim = o_dim
        self.z_dim = z_dim

        # Encoder q(z_t | x_t)
        self.enc_net = nn.Sequential(
            nn.Linear(o_dim + a_dim, h_dim), nn.ReLU(), 
            nn.Linear(h_dim, z_dim * 2)
        )
        
        # Decoder p(x_t | z_t)
        out_dim = o_dim * 2
        self.dec_net = nn.Sequential(
            nn.Linear(z_dim, h_dim), nn.ReLU(),
            nn.Linear(h_dim, out_dim)
        )

    def _split_mu_logvar(self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mu, logvar = t.chunk(2, dim=-1)
        return mu, logvar

    def forward(
        self,
        x_t: torch.Tensor,
        a_prev: torch.Tensor,
        h_prev: torch.Tensor,  # Unused for VAE-only
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """VAE-only forward pass"""
        # Encoder (posterior) x_t, a_prev -> z_t
        enc_input = torch.cat([x_t, a_prev], dim=-1)
        enc_stats = self.enc_net(enc_input)
        mu_q, logvar_q = self._split_mu_logvar(enc_stats)
        logvar_q = torch.clamp(logvar_q, min=-self.LOGVAR_CLAMP, max=self.LOGVAR_CLAMP)

        # Reparameterisation
        std_q = (0.5 * logvar_q).exp()
        eps = torch.randn_like(std_q)
        z_t = mu_q + eps * std_q

        # Decoder (reconstruction) z_t -> x_t 
        dec_out = self.dec_net(z_t)
        mu_x, logvar_x = self._split_mu_logvar(dec_out)
        logvar_x = torch.clamp(logvar_x, min=-self.LOGVAR_CLAMP, max=self.LOGVAR_CLAMP)
        
        # NLL (Gaussian, independent dims)
        inv_var = (-logvar_x).exp()
        nll = 0.5 * ((x_t - mu_x).pow(2) * inv_var + logvar_x).sum(dim=-1)
        
        # KL divergence (prior is standard normal)
        kl = 0.5 * (mu_q.pow(2) + logvar_q.exp() - logvar_q - 1.0).sum(dim=-1)
        
        # Return dummy hidden state for compatibility
        h_t = h_prev  # No RNN update
        
        return h_t, nll, kl, z_t, mu_q, logvar_q
