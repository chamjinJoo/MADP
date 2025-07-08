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

        x_t = x_t.float()
        a_prev = a_prev.float()
        h_prev = h_prev.float()

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
# 2.  Graph Attention layer (toy, self‑contained)
# ---------------------------------------------------------------------------

class GATLayer(nn.Module):
    """GPU 최적화된 Two-depth single-head Graph Attention with dropout, layer norm, and optional adjacency mask."""
    def __init__(self, in_dim: int, hid_dim: int, out_dim: int, dropout_p: float = 0.6) -> None:
        super().__init__()
        self.beta = 0.01

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

    def forward(self, V: torch.Tensor, adj: torch.Tensor, delta: torch.Tensor | None = None) -> torch.Tensor:
        """
        V: (N, in_dim)
        adj: (N, N) adjacency mask (1 for edge, 0 for no edge). If provided, blocks attention.
        """
        N = V.size(0)
        # --- Normalize delta surprises ---
        if delta is not None:
            d_vec = delta.squeeze(-1)                  # (N,)
            # Create pairwise bias matrix
            d_mat = d_vec.unsqueeze(1) + d_vec.unsqueeze(0)  # (N,N)
        # --- 1st Attention Layer ---
        Wh1 = self.W1(V)                               # (N, hid_dim)
        src1 = self.a_src1(Wh1)                        # (N,1)
        dst1 = self.a_dst1(Wh1)                        # (N,1)
        scores1 = src1 + dst1.transpose(0, 1)          # (N,N)
        
        # Apply normalized delta bias
        if delta is not None:
            scores1 = scores1 + self.beta * d_mat       
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

        # Reapply normalized delta bias
        if delta is not None:
            scores2 = scores2 + self.beta * d_mat

        e2 = self.leaky_relu(scores2)
        if adj is not None:
            e2 = e2.masked_fill(adj == 0, float('-inf'))
        alpha2 = self.softmax(e2)
        H2 = alpha2 @ Wh2                              # (N, out_dim)
        H2 = F.elu(H2)
        return H2

class CausalGATLayer(nn.Module):
    """GPU 최적화된 Multi-head Causal Graph Attention with distinct reasoning capabilities."""
    def __init__(self, in_dim: int, hid_dim: int, out_dim: int, n_agents: int, 
                 n_heads: int = 4, dropout_p: float = 0.6) -> None:
        super().__init__()
        self.n_agents = n_agents
        self.n_heads = n_heads
        self.head_dim = out_dim // n_heads
        assert out_dim % n_heads == 0, "out_dim must be divisible by n_heads"
        
        # Multi-head attention components
        self.W1 = nn.Linear(in_dim, hid_dim, bias=False)
        self.W2 = None
        
        # Head 1: Standard attention
        self.standard_attn = nn.ModuleList([
            nn.Linear(hid_dim, 1, bias=False) for _ in range(2)  # src, dst
        ])
        
        # Head 2: Causal attention (pairwise causal relationships)
        self.causal_encoder = nn.Sequential(
            nn.Linear(in_dim * 2, hid_dim), nn.ReLU(),
            nn.Linear(hid_dim, self.head_dim)
        )
        self.causal_attn = nn.ModuleList([
            nn.Linear(self.head_dim, 1, bias=False) for _ in range(2)
        ])
        
        # Head 3: Temporal attention (previous hidden states)
        self.temporal_encoder = nn.Sequential(
            nn.Linear(in_dim + hid_dim, hid_dim), nn.ReLU(),
            nn.Linear(hid_dim, self.head_dim)
        )
        self.temporal_attn = nn.ModuleList([
            nn.Linear(self.head_dim, 1, bias=False) for _ in range(2)
        ])
        
        # Head 4: Cooperative attention (cooperation patterns)
        self.coop_encoder = nn.Sequential(
            nn.Linear(in_dim * 3, hid_dim), nn.ReLU(),
            nn.Linear(hid_dim, self.head_dim)
        )
        self.coop_attn = nn.ModuleList([
            nn.Linear(self.head_dim, 1, bias=False) for _ in range(2)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(out_dim, out_dim)
        self.layer_norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout_p)
        
        # Causal mask for temporal attention
        self.register_buffer('causal_mask', torch.triu(torch.ones(n_agents, n_agents), diagonal=1).bool())

    def forward(self, V: torch.Tensor, adj: torch.Tensor, delta: torch.Tensor | None = None, 
                prev_hidden: torch.Tensor | None = None) -> torch.Tensor:
        """
        V: (N, in_dim)
        adj: (N, N) adjacency mask
        delta: (N,) or None - surprise signals
        prev_hidden: (N, hidden_dim) or None - previous hidden states
        """
        N = V.size(0)
        device = V.device
        
        # Head 1: Standard attention
        Wh1 = self.W1(V)
        src1 = self.standard_attn[0](Wh1)
        dst1 = self.standard_attn[1](Wh1)
        scores1 = src1 + dst1.transpose(0, 1)
        if adj is not None:
            scores1 = scores1.masked_fill(adj == 0, float('-inf'))
        alpha1 = F.softmax(scores1, dim=1)
        H1 = alpha1 @ Wh1
        
        # Head 2: Causal attention
        causal_inputs = []
        for i in range(N):
            for j in range(N):
                if i != j:
                    causal_inputs.append(torch.cat([V[i], V[j]], dim=0))
        if causal_inputs:
            causal_inputs = torch.stack(causal_inputs, dim=0)  # (N*(N-1), in_dim*2)
            causal_features = self.causal_encoder(causal_inputs)
            causal_scores = self.causal_attn[0](causal_features) + self.causal_attn[1](causal_features)
            causal_alpha = F.softmax(causal_scores, dim=0)
            H2 = (causal_alpha * causal_features).sum(dim=0, keepdim=True).expand(N, -1)
        else:
            H2 = torch.zeros(N, self.head_dim, device=device)
        
        # Head 3: Temporal attention
        if prev_hidden is not None:
            temp_inputs = torch.cat([V, prev_hidden], dim=-1)
            temp_features = self.temporal_encoder(temp_inputs)
            temp_scores = self.temporal_attn[0](temp_features) + self.temporal_attn[1](temp_features)
            temp_scores = temp_scores.masked_fill(self.causal_mask, float('-inf'))
            temp_alpha = F.softmax(temp_scores, dim=1)
            H3 = temp_alpha @ temp_features
        else:
            H3 = torch.zeros(N, self.head_dim, device=device)
        
        # Head 4: Cooperative attention
        coop_inputs = []
        for i in range(N):
            neighbors = []
            for j in range(N):
                if adj is None or adj[i, j] == 1:
                    neighbors.append(V[j])
            if len(neighbors) >= 2:
                # Take first two neighbors for cooperation pattern
                coop_input = torch.cat([V[i], neighbors[0], neighbors[1]], dim=0)
                coop_inputs.append(coop_input)
            else:
                # Pad with zeros if not enough neighbors
                coop_input = torch.cat([V[i], torch.zeros_like(V[i]), torch.zeros_like(V[i])], dim=0)
                coop_inputs.append(coop_input)
        
        coop_inputs = torch.stack(coop_inputs, dim=0)  # (N, in_dim*3)
        coop_features = self.coop_encoder(coop_inputs)
        coop_scores = self.coop_attn[0](coop_features) + self.coop_attn[1](coop_features)
        coop_alpha = F.softmax(coop_scores, dim=1)
        H4 = coop_alpha.transpose(-2, -1) @ coop_features
        H4 = H4.squeeze(-2)  # (N, D)로 맞춤
        
        # 모든 텐서의 차원 확인 및 맞춤
        if H4.dim() == 1:
            H4 = H4.unsqueeze(-1)  # (N,) -> (N, 1)
        elif H4.dim() == 3:
            H4 = H4.squeeze(-2)    # (N, 1, D) -> (N, D)
            
        # 첫 번째 차원이 다른 텐서들과 맞는지 확인
        expected_size = H1.size(0)  # N
        if H4.size(0) != expected_size:
            # H4의 첫 번째 차원을 맞춤
            if H4.size(0) > expected_size:
                H4 = H4[:expected_size]  # 앞에서부터 잘라냄
            else:
                # 부족한 경우 패딩
                padding = torch.zeros(expected_size - H4.size(0), H4.size(1), device=H4.device)
                H4 = torch.cat([H4, padding], dim=0)
            
        # Combine all heads
        H_combined = torch.cat([H1, H2, H3, H4], dim=-1)  # (N, ?)
        # W2를 동적으로 생성 (최초 1회만)
        if self.W2 is None or H_combined.size(-1) != self.W2.in_features:
            self.W2 = nn.Linear(H_combined.size(-1), self.n_heads * self.head_dim, bias=False).to(H_combined.device)
        H_combined = self.W2(H_combined)
        
        # Output projection and normalization
        output = self.output_proj(H_combined)
        output = self.layer_norm(output)
        output = self.dropout(output)
        
        return F.elu(output)

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

class VRNNGATA2C(nn.Module):
    """GPU 최적화된 VRNN + GAT + Actor-Critic 모델"""
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dim: int,
        z_dim: int,
        n_agents: int,
        gat_dim: int,
        use_gat: bool = True,  # GAT ablation 옵션 추가
        use_causal_gat: bool = True,  # Causal GAT 사용 옵션
        use_rnn: bool = True,  # RNN 사용 여부 옵션 추가
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_dim = hidden_dim
        self.z_dim = z_dim
        self.nagents = n_agents
        self.gat_dim = gat_dim
        self.use_gat = use_gat
        self.use_causal_gat = use_causal_gat
        self.use_rnn = use_rnn

        # VRNN cells for each agent
        self.vrnn_cells = nn.ModuleList([
            VRNNCell(obs_dim, act_dim, hidden_dim, z_dim)
            for _ in range(n_agents)
        ])

        # GAT layers
        if use_gat:
            if use_causal_gat:
                self.gat_layer = CausalGATLayer(
                    hidden_dim, gat_dim, gat_dim, n_agents
                )
            else:
                self.gat_layer = GATLayer(hidden_dim, gat_dim, gat_dim)
        else:
            self.gat_layer = None

        # Actor-Critic heads
        if use_rnn:
            input_dim = hidden_dim + (gat_dim if use_gat else z_dim)
        else:
            input_dim = obs_dim #+ (gat_dim if use_gat else z_dim)
        
        self.actor_critic = ActorCriticHead(input_dim, act_dim)

    def get_decoders(self) -> list[Any]:
        """각 VRNNCell의 decoder(dec_net)를 list로 반환"""
        return [cell.dec_net for cell in self.vrnn_cells]

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

        # Belief consistency loss 계산 (h_prev도 전달)
        decoders = self.get_decoders()
        belief_consis_loss = belief_consistency_loss(z, obs, decoders, h_prev)

        # GAT processing
        V_gat = None
        if self.use_gat and self.gat_layer is not None:
            # Adjacency matrix (fully connected for now)
            adj = torch.ones(N, N, device=device)
            
            # GAT forward pass
            if self.use_causal_gat:
                # Use previous GAT output for causal attention
                prev_gat_output = getattr(self, 'prev_gat_output', None)
                V_gat = self.gat_layer(h_new, adj, rolling_mean_error, prev_gat_output)
                self.prev_gat_output = V_gat.detach()
            else:
                V_gat = self.gat_layer(h_new, adj, rolling_mean_error)

        # Actor-Critic forward pass
        if self.use_rnn:
            if self.use_gat and V_gat is not None:
                ac_input = torch.cat([h_new, V_gat], dim=-1)
            else:
                ac_input = torch.cat([h_new, z], dim=-1)
        else:
            if self.use_gat and V_gat is not None:
                ac_input = torch.cat([obs, V_gat], dim=-1)
            else:
                ac_input = obs

        logits, values = self.actor_critic(ac_input.float())

        return h_new, nll, kl, logits, None, values, mu, logvar, z, V_gat, belief_consis_loss

# ---------------------------------------------------------------------------
# Loss functions (GPU 최적화)
# ---------------------------------------------------------------------------

def pairwise_coop_kl(
    mu: torch.Tensor,        # (B, d)
    logvar: torch.Tensor,    # (B, d)
    n_agents: int
) -> torch.Tensor:
    """GPU 최적화된 pairwise cooperation KL divergence"""
    B, d = mu.shape
    assert B % n_agents == 0, f"Batch size {B} must be divisible by n_agents {n_agents}"
    
    # Reshape to (n_agents, batch_per_agent, d)
    mu_reshaped = mu.view(n_agents, -1, d)      # (n_agents, batch_per_agent, d)
    logvar_reshaped = logvar.view(n_agents, -1, d)  # (n_agents, batch_per_agent, d)
    
    total_kl = 0.0
    count = 0
    
    # Compute pairwise KL between agents
    for i in range(n_agents):
        for j in range(i + 1, n_agents):
            mu_i = mu_reshaped[i]  # (batch_per_agent, d)
            logvar_i = logvar_reshaped[i]
            mu_j = mu_reshaped[j]
            logvar_j = logvar_reshaped[j]
            
            # KL divergence between agent i and j
            kl_ij = kl_gauss(mu_i, logvar_i, mu_j, logvar_j)
            kl_ji = kl_gauss(mu_j, logvar_j, mu_i, logvar_i)
            
            total_kl += kl_ij.mean() + kl_ji.mean()
            count += 2
    
    if count > 0:
        return torch.tensor(total_kl / count, device=mu.device, dtype=mu.dtype)
    else:
        return torch.tensor(0.0, device=mu.device, dtype=mu.dtype)

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
# ---------------------------------------------------------------------------
# Belief Consistency Loss (VRNN 디코더 기반)
# ---------------------------------------------------------------------------
def belief_consistency_loss(zs: torch.Tensor, obs: torch.Tensor, decoders: Sequence[nn.Module], h_prevs: torch.Tensor) -> torch.Tensor:
    """
    zs: (N, z_dim)
    obs: (N, obs_dim)
    decoders: list of decoder modules (length N)
    h_prevs: (N, h_dim) 각 에이전트의 prev hidden state
    각 에이전트의 디코더로 (z, h_prev)를 concat하여 obs를 복원, MSE loss 계산
    """
    N = zs.size(0)
    total_loss = 0.0
    count = 0
    for i in range(N):
        for j in range(N):
            dec_in = torch.cat([zs[j], h_prevs[i]], dim=-1)
            dec_out = decoders[i](dec_in)  # (obs_dim*2)
            mu_x, _ = dec_out.chunk(2, dim=-1)  # (obs_dim)
            total_loss += F.mse_loss(mu_x, obs[j])
            count += 1
    if count > 0:
        return torch.tensor(total_loss / count, device=zs.device, dtype=zs.dtype)
    else:
        return torch.tensor(0.0, device=zs.device, dtype=zs.dtype)
