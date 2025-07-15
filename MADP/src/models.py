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

# ---------------------------------------------------------------------------
# 1.  VRNN building blocks
# ---------------------------------------------------------------------------
class VRNNCell(nn.Module):
    """GPU 최적화된 Variational RNN cell - 개별 agent 정보만 사용"""

    LOGVAR_CLAMP = 10.0

    def __init__(
        self, 
        o_dim: int,
        h_dim: int, 
        z_dim: int,
        n_agents: int = 2,
    ) -> None:
        super().__init__()
        self.o_dim = o_dim
        self.z_dim = z_dim
        self.n_agents = n_agents
        # 개별 agent hidden state dimension (multi-agent 사용 안함)
        self.multi_h_dim = h_dim        
        # RNN core
        self.rnn = nn.GRUCell(o_dim + z_dim, h_dim)
        # Prior p(z_t | h_{t-1}) - 개별 agent 정보만 사용
        self.prior_net = nn.Sequential(
            nn.Linear(self.multi_h_dim, h_dim), nn.ReLU(), 
            nn.Linear(h_dim, z_dim * 2)
        )
        # Encoder q(z_t | x_t, h_{t-1}) - 개별 agent 정보만 사용
        self.enc_net = nn.Sequential(
            nn.Linear(o_dim + self.multi_h_dim, h_dim), nn.ReLU(), 
            nn.Linear(h_dim, z_dim * 2)
        )
        # Decoder p(x_t | z_t, h_{t-1}) — predicts μ and log σ² (diagonal)
        out_dim = o_dim * 2
        self.dec_net = nn.Sequential(
            nn.Linear(z_dim + self.multi_h_dim, h_dim), nn.ReLU(),
            nn.Linear(h_dim, out_dim)
        )
    def _split_mu_logvar(self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mu, logvar = t.chunk(2, dim=-1)
        return mu, logvar
    def forward(
        self,
        x_t: torch.Tensor,
        h_prev: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """개별 agent VRNN step.
        Args:
            x_t: 현재 agent의 observation (obs_dim,)
            h_prev: 현재 agent의 이전 hidden state (hidden_dim,)
        Returns
        -------
        h_t : next hidden state
        nll : negative log likelihood
        kl : KL divergence at this step
        z_t : latent sample
        mu_q, logvar_q : encoder stats
        """
        x_t = x_t.float()
        h_prev = h_prev.float()

        # Prior h_{t-1} -> z_t (개별 agent 정보만 사용)
        prior_stats = self.prior_net(h_prev)
        mu_p, logvar_p = self._split_mu_logvar(prior_stats)
        logvar_p = torch.clamp(logvar_p, min=-self.LOGVAR_CLAMP, max=self.LOGVAR_CLAMP) 
        # Encoder (posterior) x_t, h_{t-1} -> z_t (개별 agent 정보만 사용)
        enc_input = torch.cat([x_t, h_prev], dim=-1)
        enc_stats = self.enc_net(enc_input)
        mu_q, logvar_q = self._split_mu_logvar(enc_stats)
        logvar_q = torch.clamp(logvar_q, min=-self.LOGVAR_CLAMP, max=self.LOGVAR_CLAMP)
        # Reparameterisation
        std_q = (0.5 * logvar_q).exp()
        eps = torch.randn_like(std_q)
        z_t = mu_q + eps * std_q # latent variable  
        # Decoder (reconstruction) z_t, h_{t-1} -> x_t (개별 agent 정보만 사용)
        dec_out = self.dec_net(torch.cat([z_t, h_prev], dim=-1))
        mu_x, logvar_x = self._split_mu_logvar(dec_out)
        logvar_x = torch.clamp(logvar_x, min=-self.LOGVAR_CLAMP, max=self.LOGVAR_CLAMP)
        # NLL (Gaussian, independent dims)
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
        # GRU update (기존과 동일)
        rnn_in = torch.cat([x_t, z_t], dim=-1)
        h_t = self.rnn(rnn_in, h_prev)
        return h_t, nll, kl, z_t, mu_q, logvar_q

# ---------------------------------------------------------------------------
# 2.  Graph Attention layer (toy, self‑contained)
# ---------------------------------------------------------------------------

class GATLayer(nn.Module):
    """GPU 최적화된 Cooperation-aware Graph Attention - cooperation loss를 연결 강도에 반영"""
    def __init__(self, in_dim: int, hid_dim: int, out_dim: int, dropout_p: float = 0.6, 
                 use_coop_attention: bool = True) -> None:
        super().__init__()
        self.beta = 0.01
        self.use_coop_attention = use_coop_attention

        # 1st attention stage
        self.W1 = nn.Linear(in_dim, hid_dim, bias=False)
        self.a_src1 = nn.Linear(hid_dim, 1, bias=False)
        self.a_dst1 = nn.Linear(hid_dim, 1, bias=False)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(dropout_p)
        
        # 2nd attention stage
        self.W2 = nn.Linear(hid_dim, out_dim, bias=False)
        self.a_src2 = nn.Linear(out_dim, 1, bias=False)
        self.a_dst2 = nn.Linear(out_dim, 1, bias=False)
        
        # Cooperation attention weights
        if use_coop_attention:
            self.coop_weight = nn.Parameter(torch.ones(1) * 0.1)
            self.coop_bias = nn.Parameter(torch.zeros(1))

    def forward(self, V: torch.Tensor, adj: torch.Tensor, 
            coop_loss: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        V: (N, in_dim)
        adj: (N, N) adjacency mask (1 for edge, 0 for no edge)
        coop_loss: (N,) cooperation loss per agent (높을수록 더 강한 연결)
        """
        N = V.size(0)
        
        # --- Cooperation-based attention bias ---
        coop_bias = None
        if self.use_coop_attention and coop_loss is not None:
            # Cooperation loss를 정규화하여 attention bias로 사용
            coop_norm = torch.sigmoid(self.coop_weight * coop_loss + self.coop_bias)
            # Pairwise cooperation bias matrix 생성
            coop_bias = coop_norm.unsqueeze(1) + coop_norm.unsqueeze(0)  # (N,N)
        
        # --- 1st Attention Layer ---
        Wh1 = self.W1(V)                               # (N, hid_dim)
        src1 = self.a_src1(Wh1)                        # (N,1)
        dst1 = self.a_dst1(Wh1)                        # (N,1)
        scores1 = src1 + dst1.transpose(0, 1)          # (N,N)
        
        # Apply cooperation bias
        if coop_bias is not None:
            scores1 = scores1 + coop_bias
        e1 = self.leaky_relu(scores1) 
        if adj is not None:
            e1 = e1.masked_fill(adj == 0, float('-inf'))
        alpha1 = self.softmax(e1)  # (N,N)
        H1 = alpha1 @ Wh1          # (N, hid_dim)
        H1 = F.elu(H1)             # (N, hid_dim)

        # --- 2nd Attention Layer ---
        Wh2 = self.W2(H1)                              # (N, out_dim)
        src2 = self.a_src2(Wh2)                        # (N,1)
        dst2 = self.a_dst2(Wh2)                        # (N,1)
        scores2 = src2 + dst2.transpose(0, 1)          # (N,N)

        # Reapply cooperation bias
        if coop_bias is not None:
            scores2 = scores2 + coop_bias

        e2 = self.leaky_relu(scores2)
        if adj is not None:
            e2 = e2.masked_fill(adj == 0, float('-inf'))
        alpha2 = self.softmax(e2)
        H2 = alpha2 @ Wh2                              # (N, out_dim)
        H2 = F.elu(H2)
        return H2

class ActorCriticHead(nn.Module):
    """GPU 최적화된 Actor-Critic 헤드 (Parameter Sharing)"""
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

class VRNNGATA2C(nn.Module):
    """개선된 VRNN + GAT + Actor-Critic 모델 - 개별 agent 정보 활용 및 Cooperation-aware GAT"""
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dim: int,
        z_dim: int,
        n_agents: int,
        gat_dim: int,
        use_gat: bool = True,
        use_rnn: bool = True,
        use_coop_attention: bool = True,
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_dim = hidden_dim
        self.z_dim = z_dim
        self.nagents = n_agents
        self.gat_dim = gat_dim
        self.use_gat = use_gat
        self.use_rnn = use_rnn
        self.use_coop_attention = use_coop_attention

        # 개별 agent VRNN cell - 개별 agent 정보만 사용
        self.vrnn_cell = VRNNCell(
            obs_dim, hidden_dim, z_dim, 
            n_agents=n_agents
        )

        # Cooperation-aware GAT layer
        if use_gat:
            self.gat_layer = GATLayer(
                hidden_dim, gat_dim, gat_dim, 
                use_coop_attention=use_coop_attention
            )
        else:
            self.gat_layer = None

        # Actor-Critic heads (각 agent별로 별도 네트워크)
        if use_rnn:
            input_dim = obs_dim + (gat_dim if use_gat else z_dim)
        else:
            input_dim = obs_dim
        
        self.actor_critic_heads = nn.ModuleList([
            ActorCriticHead(input_dim, act_dim)
            for _ in range(n_agents)
        ])

    def forward_step(
        self,
        obs: torch.Tensor,      # (N, obs_dim)
        h_prev: torch.Tensor,   # (N, hidden_dim)
    ) -> tuple:
        N = obs.size(0)
        device = obs.device
        
        # 개별 agent VRNN forward pass
        h_new_list = []
        nll_list = []
        kl_list = []
        z_list = []
        mu_list = []
        logvar_list = []
        
        for i in range(N):
            h_new, nll, kl, z, mu, logvar = self.vrnn_cell(obs[i], h_prev[i])
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

        # Cooperation loss 계산 (GAT 연결 강도에 사용)
        coop_loss = None
        if self.use_coop_attention and N > 1:
            # 각 agent-pair 간의 KL divergence 계산
            coop_loss = compute_pairwise_kl_for_gat(mu, logvar, N)

        # Cooperation-aware GAT processing
        V_gat = None
        if self.use_gat and self.gat_layer is not None:
            adj = torch.ones(N, N, device=device)
            V_gat = self.gat_layer(h_new, adj, coop_loss=coop_loss)

        # Actor-Critic forward pass (각 agent별로 별도 네트워크)
        if self.use_rnn:
            if self.use_gat and V_gat is not None:
                ac_input = torch.cat([obs, V_gat], dim=-1)
            else:
                ac_input = torch.cat([obs, z], dim=-1)
        else:
            if self.use_gat and V_gat is not None:
                ac_input = torch.cat([obs, V_gat], dim=-1)
            else:
                ac_input = obs
        
        # 각 agent별로 별도의 Actor-Critic 네트워크 적용
        logits_list = []
        values_list = []
        for i in range(N):
            agent_input = ac_input[i].unsqueeze(0)  # (1, input_dim)
            agent_logits, agent_values = self.actor_critic_heads[i](agent_input.float())
            logits_list.append(agent_logits.squeeze(0))  # (act_dim,)
            values_list.append(agent_values.squeeze(0))   # (1,)
        
        logits = torch.stack(logits_list, dim=0)  # (N, act_dim)
        values = torch.stack(values_list, dim=0)   # (N, 1)
        
        return h_new, nll, kl, logits, values, mu, logvar, z, V_gat

# ---------------------------------------------------------------------------
# Loss functions (GPU 최적화) - 개선된 버전
# ---------------------------------------------------------------------------

def kl_annealing_schedule(step: int, total_steps: int, min_beta: float = 0.0, max_beta: float = 1.0) -> float:
    """KL annealing schedule - 점진적으로 KL weight를 증가시킴"""
    if step < total_steps * 0.1:  # 처음 10%는 0
        return min_beta
    elif step < total_steps * 0.5:  # 10-50%는 선형 증가
        progress = (step - total_steps * 0.1) / (total_steps * 0.4)
        return min_beta + (max_beta - min_beta) * progress
    else:  # 50% 이후는 최대값
        return max_beta

def compute_beta_vae_loss(
    nll: torch.Tensor,      # (B,)
    kl: torch.Tensor,       # (B,)
    mu: torch.Tensor,       # (B, d)
    logvar: torch.Tensor,   # (B, d)
    nll_coef: float,
    kl_coef: float,
    coop_coef: float,
    n_agents: int,
    beta: float = 1.0,
    capacity: float = 0.0,
    target_capacity: float = 0.0
) -> tuple[torch.Tensor, dict]:
    """Beta-VAE loss with capacity constraint"""
    # Basic VAE loss
    loss_nll = nll.mean()
    loss_kl = kl.mean()
    
    # Capacity constraint (KL divergence should be close to target_capacity)
    capacity_loss = torch.abs(loss_kl - target_capacity)
    
    # Cooperation loss with improved stability
    loss_coop = compute_stable_cooperation_loss(mu, logvar, n_agents)
    
    # Total loss with beta and capacity
    total_loss = nll_coef * loss_nll + beta * kl_coef * loss_kl + capacity * capacity_loss + coop_coef * loss_coop
    
    metrics = {
        'vae_nll': loss_nll.item(),
        'vae_kl': loss_kl.item(),
        'coop_kl': loss_coop.item(),
        'capacity_loss': capacity_loss.item(),
        'beta': beta,
        'loss_vae': total_loss.item()
    }
    
    return total_loss, metrics

def compute_stable_cooperation_loss(mu: torch.Tensor, logvar: torch.Tensor, n_agents: int) -> torch.Tensor:
    """더 안정적인 cooperation loss"""
    B, d = mu.shape
    assert B % n_agents == 0, f"Batch size {B} must be divisible by n_agents {n_agents}"
    
    # Reshape to (n_agents, batch_per_agent, d)
    mu_reshaped = mu.view(n_agents, -1, d)      # (n_agents, batch_per_agent, d)
    logvar_reshaped = logvar.view(n_agents, -1, d)  # (n_agents, batch_per_agent, d)
    
    # 1. Centered KL divergence (모든 agent의 평균과의 KL)
    mu_mean = mu_reshaped.mean(dim=0, keepdim=True)  # (1, batch_per_agent, d)
    logvar_mean = logvar_reshaped.mean(dim=0, keepdim=True)
    
    center_kl = 0.0
    for i in range(n_agents):
        kl_to_center = kl_gauss(mu_reshaped[i], logvar_reshaped[i], mu_mean.squeeze(0), logvar_mean.squeeze(0))
        center_kl += kl_to_center.mean()
    
    # 2. Pairwise JSD (더 안정적)
    pairwise_jsd = compute_pairwise_jsd_stable(mu_reshaped, logvar_reshaped)
    
    # 3. Variance regularization (agent 간 분산을 줄임)
    var_reg = torch.var(mu_reshaped, dim=0).mean() + torch.var(logvar_reshaped, dim=0).mean()
    
    # 조합된 cooperation loss
    total_coop_loss = 0.5 * center_kl + 0.3 * pairwise_jsd + 0.2 * var_reg
    
    return total_coop_loss

def compute_pairwise_kl_for_gat(mu: torch.Tensor, logvar: torch.Tensor, n_agents: int) -> torch.Tensor:
    """각 agent-pair 간의 KL divergence 계산 (GAT 연결 강도에 사용)"""
    # 각 agent별로 다른 agent들과의 평균 KL divergence 계산
    agent_coop_losses = []
    for i in range(n_agents):
        # 현재 agent의 분포
        mu_i = mu[i:i+1]  # (1, z_dim)
        logvar_i = logvar[i:i+1]  # (1, z_dim)
        # 다른 agent들과의 KL divergence 계산
        kl_to_others = []
        for j in range(n_agents):
            if i != j:
                mu_j = mu[j:j+1]  # (1, z_dim)
                logvar_j = logvar[j:j+1]  # (1, z_dim)
                # KL(i||j) 계산
                kl_ij = kl_gauss(mu_i, logvar_i, mu_j, logvar_j)
                kl_to_others.append(kl_ij.mean())
        if kl_to_others:
            # 평균 KL을 cooperation loss로 사용 (높을수록 더 강한 연결 필요)
            agent_coop_loss = torch.stack(kl_to_others).mean()
        else:
            agent_coop_loss = torch.tensor(0.0, device=mu.device, dtype=mu.dtype)
        agent_coop_losses.append(agent_coop_loss)
    return torch.stack(agent_coop_losses)  # (n_agents,)

def compute_pairwise_jsd_stable(mu_reshaped: torch.Tensor, logvar_reshaped: torch.Tensor) -> torch.Tensor:
    """안정적인 pairwise JSD 계산"""
    n_agents = mu_reshaped.shape[0]
    total_jsd = torch.tensor(0.0, device=mu_reshaped.device, dtype=mu_reshaped.dtype)
    count = 0
    
    for i in range(n_agents):
        for j in range(i + 1, n_agents):
            mu_i = mu_reshaped[i]
            logvar_i = logvar_reshaped[i]
            mu_j = mu_reshaped[j]
            logvar_j = logvar_reshaped[j]
            
            # 안정적인 JSD 계산
            mu_m = 0.5 * (mu_i + mu_j)
            # logvar의 안정적인 평균 계산
            var_i = torch.exp(torch.clamp(logvar_i, min=-10, max=10))
            var_j = torch.exp(torch.clamp(logvar_j, min=-10, max=10))
            var_m = 0.5 * (var_i + var_j)
            logvar_m = torch.log(torch.clamp(var_m, min=1e-8))
            
            jsd = 0.5 * (kl_gauss(mu_i, logvar_i, mu_m, logvar_m) + 
                        kl_gauss(mu_j, logvar_j, mu_m, logvar_m))
            
            total_jsd += jsd.mean()
            count += 1
    
    if count > 0:
        return (total_jsd / count).detach().clone()
    else:
        return torch.tensor(0.0, device=mu_reshaped.device, dtype=mu_reshaped.dtype)

# 기존 함수들 (하위 호환성을 위해 유지)
def pairwise_coop_kl(mu, logvar, n_agents):
    B, d = mu.shape
    b = B // n_agents
    mu = mu.view(n_agents, b, d)           # (N, Bp, d)
    logvar = logvar.view(n_agents, b, d)   # (N, Bp, d)

    # 모든 (i,j) 조합을 브로드캐스트
    mu_i = mu.unsqueeze(1)                # (N, 1, Bp, d)
    mu_j = mu.unsqueeze(0)                # (1, N, Bp, d)
    lv_i = logvar.unsqueeze(1)
    lv_j = logvar.unsqueeze(0)

    # KL(i||j)
    var_i = lv_i.exp()
    var_j = lv_j.exp()
    kl_ij = 0.5 * (
        ((mu_i - mu_j)**2 / var_j)
        + (var_i / var_j)
        + (lv_j - lv_i)
        - 1.0
    ).sum(-1)  # (N, N, Bp)

    # 대각선 제외, 대칭 평균
    mask = ~torch.eye(n_agents, dtype=torch.bool, device=mu.device)
    kl_pair = (kl_ij + kl_ij.transpose(0,1)) / 2  # (N, N, Bp)
    sym_kl = kl_pair[mask].mean()               # 스칼라

    return sym_kl

def pairwise_jsd_gaussian(mu: torch.Tensor, logvar: torch.Tensor, n_agents: int) -> torch.Tensor:
    """GPU 최적화된 pairwise Jensen-Shannon divergence"""
    B, d = mu.shape
    assert B % n_agents == 0
    
    mu_reshaped = mu.view(n_agents, -1, d)
    logvar_reshaped = logvar.view(n_agents, -1, d)
    
    total_jsd = 0.0
    count = 0
    
    for i in range(n_agents):
        for j in range(i + 1, n_agents):
            mu_i = mu_reshaped[i]
            logvar_i = logvar_reshaped[i]
            mu_j = mu_reshaped[j]
            logvar_j = logvar_reshaped[j]
            
            # JSD = 0.5 * (KL(p||m) + KL(q||m)) where m = 0.5 * (p + q)
            mu_m = 0.5 * (mu_i + mu_j)
            logvar_m = torch.log(0.5 * (torch.exp(logvar_i) + torch.exp(logvar_j)))
            
            jsd = 0.5 * (kl_gauss(mu_i, logvar_i, mu_m, logvar_m) + 
                        kl_gauss(mu_j, logvar_j, mu_m, logvar_m))
            
            total_jsd += jsd.mean()
            count += 1
    
    if count > 0:
        return torch.tensor(total_jsd / count, device=mu.device, dtype=mu.dtype)
    else:
        return torch.tensor(0.0, device=mu.device, dtype=mu.dtype)

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
    mu: torch.Tensor,       # (B, d)
    logvar: torch.Tensor,   # (B, d)
    nll_coef: float,
    kl_coef: float,
    coop_coef: float,
    n_agents: int
) -> tuple[torch.Tensor, dict]:
    """GPU 최적화된 VAE loss 계산"""
    # Basic VAE loss
    loss_nll = nll.mean()
    loss_kl = kl.mean()
    # Cooperation loss
    loss_coop = pairwise_coop_kl(mu, logvar, n_agents)
    # Total loss
    total_loss = nll_coef * loss_nll + kl_coef * loss_kl + coop_coef * loss_coop
    
    metrics = {
        'vae_nll': loss_nll.item(),
        'vae_kl': loss_kl.item(),
        'coop_kl': loss_coop.item(),
        'loss_vae': total_loss.item()
    }
    
    return total_loss, metrics

def compute_rl_loss(logits: torch.Tensor,
                    actions: torch.Tensor,
                    adv: torch.Tensor,
                    values_pred: torch.Tensor,
                    returns: torch.Tensor,
                    ent_coef: float,
                    value_coef: float):
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

