from __future__ import annotations
from pathlib import Path
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
# ---------------------------------------------------------------------------
# 1.  VRNN building blocks
# ---------------------------------------------------------------------------
class VRNNCell(nn.Module):
    """Single‑step Variational RNN cell (Chung et al., 2015).

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
        
        gae_Returns
        -------
        h_t : next hidden state
        recon_x : decoder output
        kl : KL divergence at this step (no beta factor)
        z_t : latent sample
        mu_q, logvar_q : encoder stats (useful for diagnostics)
        """

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
        z_t = mu_q + eps * std_q

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
    """Two-depth single-head Graph Attention with dropout, layer norm, and optional adjacency mask."""
    def __init__(self, in_dim: int, hid_dim: int, out_dim: int, dropout_p: float = 0.6) -> None:
        super().__init__()
        self.beta = 0.1

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
    """Multi-head Causal Graph Attention with distinct reasoning capabilities."""
    def __init__(self, in_dim: int, hid_dim: int, out_dim: int, n_agents: int, 
                 n_heads: int = 4, dropout_p: float = 0.6) -> None:
        super().__init__()
        self.n_agents = n_agents
        self.n_heads = n_heads
        self.head_dim = out_dim // n_heads
        assert out_dim % n_heads == 0, "out_dim must be divisible by n_heads"
        
        # Multi-head attention components
        self.W1 = nn.Linear(in_dim, hid_dim, bias=False)
        self.W2 = nn.Linear(hid_dim, out_dim, bias=False)
        
        # Head 1: Standard attention
        self.standard_attn = nn.ModuleList([
            nn.Linear(hid_dim, 1, bias=False) for _ in range(2)  # src, dst
        ])
        
        # Head 2: Causal attention (pairwise causal relationships)
        self.causal_encoder = nn.Sequential(
            nn.Linear(in_dim * 2, hid_dim), nn.ReLU(),
            nn.Linear(hid_dim, self.head_dim)
        )
        self.causal_attn = nn.Linear(self.head_dim, 1, bias=False)
        
        # Head 3: Temporal attention (temporal dependencies)
        self.temporal_encoder = nn.Sequential(
            nn.Linear(in_dim + self.head_dim, hid_dim), nn.ReLU(),
            nn.Linear(hid_dim, self.head_dim)
        )
        self.temporal_attn = nn.Linear(self.head_dim, 1, bias=False)
        
        # Head 4: Situation-aware attention (context-dependent)
        self.situation_encoder = nn.Sequential(
            nn.Linear(in_dim, hid_dim), nn.ReLU(),
            nn.Linear(hid_dim, self.head_dim)
        )
        self.situation_attn = nn.Linear(self.head_dim, 1, bias=False)
        
        # Output projection for each head
        self.head_projections = nn.ModuleList([
            nn.Linear(self.head_dim, self.head_dim) for _ in range(n_heads)
        ])
        
        # Head fusion
        self.head_fusion = nn.Sequential(
            nn.Linear(out_dim, out_dim), nn.ReLU(),
            nn.Linear(out_dim, out_dim)
        )
        
        # Utilities
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(dropout_p)
        self.layer_norm = nn.LayerNorm(out_dim)

    def forward(self, V: torch.Tensor, adj: torch.Tensor, delta: torch.Tensor | None = None, 
                prev_hidden: torch.Tensor | None = None) -> torch.Tensor:
        """
        V: (N, in_dim)
        adj: (N, N) adjacency mask
        delta: (N,) rolling mean errors
        prev_hidden: (N, head_dim) previous hidden states for temporal reasoning
        """
        N = V.size(0)
        
        # Shared feature transformation
        Wh1 = self.W1(V)  # (N, hid_dim)
        Wh2 = self.W2(Wh1)  # (N, out_dim)
        
        # Initialize head outputs
        head_outputs = []
        
        # Head 1: Standard attention
        src1, dst1 = self.standard_attn[0](Wh1), self.standard_attn[1](Wh1)
        scores1 = src1 + dst1.transpose(0, 1)  # (N, N)
        
        if delta is not None:
            d_vec = delta.squeeze(-1)  # (N,)
            d_mat = d_vec.unsqueeze(1) + d_vec.unsqueeze(0)  # (N, N)
            scores1 = scores1 + 0.1 * d_mat
        
        e1 = self.leaky_relu(scores1)
        if adj is not None:
            e1 = e1.masked_fill(adj == 0, float('-inf'))
        alpha1 = self.softmax(e1)  # (N, N)
        H1 = alpha1 @ Wh1  # (N, hid_dim)
        H1 = F.elu(H1)
        
        # Project to head dimension
        H1_proj = self.head_projections[0](H1[:, :self.head_dim])
        head_outputs.append(H1_proj)
        
        # Head 2: Causal attention
        causal_features = []
        causal_scores = torch.zeros(N, N, device=V.device)
        
        for i in range(N):
            for j in range(N):
                if i != j:
                    # Pairwise causal relationship
                    pair_input = torch.cat([V[i], V[j]], dim=-1)  # (in_dim * 2)
                    causal_feat = self.causal_encoder(pair_input)  # (head_dim)
                    causal_features.append(causal_feat)
                    
                    # Causal attention score
                    causal_score = self.causal_attn(causal_feat)  # (1,)
                    causal_scores[i, j] = causal_score.squeeze()
        
        if len(causal_features) > 0:
            # Apply causal attention
            e2 = self.leaky_relu(causal_scores)
            if adj is not None:
                e2 = e2.masked_fill(adj == 0, float('-inf'))
            alpha2 = self.softmax(e2)  # (N, N)
            
            # Aggregate causal features
            causal_features = torch.stack(causal_features)  # (N*(N-1), head_dim)
            causal_aggregate = causal_features.mean(dim=0)  # (head_dim)
            H2 = causal_aggregate.unsqueeze(0).expand(N, -1)  # (N, head_dim)
        else:
            H2 = torch.zeros(N, self.head_dim, device=V.device)
        
        H2_proj = self.head_projections[1](H2)
        head_outputs.append(H2_proj)
        
        # Head 3: Temporal attention
        if prev_hidden is not None:
            temporal_input = torch.cat([V, prev_hidden], dim=-1)  # (N, in_dim + head_dim)
            temporal_features = self.temporal_encoder(temporal_input)  # (N, head_dim)
        else:
            temporal_features = torch.zeros(N, self.head_dim, device=V.device)
        
        # Temporal attention scores
        temporal_scores = self.temporal_attn(temporal_features)  # (N, 1)
        temporal_scores = temporal_scores.expand(-1, N)  # (N, N)
        
        e3 = self.leaky_relu(temporal_scores)
        if adj is not None:
            e3 = e3.masked_fill(adj == 0, float('-inf'))
        alpha3 = self.softmax(e3)  # (N, N)
        H3 = alpha3 @ temporal_features  # (N, head_dim)
        
        H3_proj = self.head_projections[2](H3)
        head_outputs.append(H3_proj)
        
        # Head 4: Situation-aware attention
        situation_features = self.situation_encoder(V)  # (N, head_dim)
        situation_scores = self.situation_attn(situation_features)  # (N, 1)
        situation_scores = situation_scores.expand(-1, N)  # (N, N)
        
        e4 = self.leaky_relu(situation_scores)
        if adj is not None:
            e4 = e4.masked_fill(adj == 0, float('-inf'))
        alpha4 = self.softmax(e4)  # (N, N)
        H4 = alpha4 @ situation_features  # (N, head_dim)
        
        H4_proj = self.head_projections[3](H4)
        head_outputs.append(H4_proj)
        
        # Concatenate all head outputs
        H_concat = torch.cat(head_outputs, dim=-1)  # (N, out_dim)
        
        # Head fusion
        H_fused = self.head_fusion(H_concat)  # (N, out_dim)
        H_fused = F.elu(H_fused)
        
        # Layer normalization and dropout
        H_output = self.layer_norm(H_fused)
        H_output = self.dropout(H_output)
        
        return H_output
    
# ---------------------------------------------------------------------------
# 3. Actor‑Critic heads (option 1)
# ---------------------------------------------------------------------------
class ActorCriticHead(nn.Module):
    def __init__(self, in_dim: int, act_dim: int) -> None:
        super().__init__()
        # 현재 정책 네트워크
        self.actor = nn.Sequential(
            nn.Linear(in_dim, in_dim), nn.ReLU(),
            nn.Linear(in_dim, act_dim)
        )
        # Critic
        self.critic = nn.Sequential(
            nn.Linear(in_dim, in_dim), nn.ReLU(),
            nn.Linear(in_dim, 1)
        )

    def forward(self, input: torch.Tensor):
        policy_logits = self.actor(input)
        value = self.critic(input).squeeze(-1)
        return policy_logits, value
# ---------------------------------------------------------------------------
# 4.  Full model wrapper
# ---------------------------------------------------------------------------

class VRNNGATA2C(nn.Module):
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
    ) -> None:
        super().__init__()
        self.nagents = n_agents
        self.act_dim = act_dim
        self.hidden_dim = hidden_dim
        self.z_dim = z_dim
        self.gat_dim = gat_dim
        self.use_gat = use_gat
        self.use_causal_gat = use_causal_gat

        self.vrnn_cells = nn.ModuleList(
            [VRNNCell(obs_dim, act_dim, hidden_dim, z_dim) for _ in range(n_agents)]
        )
        if use_gat:
            if use_causal_gat:
                self.gat = CausalGATLayer(hidden_dim + z_dim, 
                                         gat_dim, 
                                         gat_dim,
                                         n_agents,
                                         n_heads=4)  # Multi-head Causal GAT
            else:
                self.gat = GATLayer(hidden_dim + z_dim, 
                                   gat_dim, 
                                   gat_dim)  # Standard GAT
        else:
            self.gat = None
      
        # 정책 입력: obs_dim + hidden_dim + gat_dim (원본 + 개인 + 이웃 정보)
        self.a2c_heads = nn.ModuleList(
            [ActorCriticHead(obs_dim + hidden_dim + gat_dim, act_dim) for _ in range(n_agents)]
        )
        
        # Causal GAT를 위한 이전 GAT 출력 저장
        self.prev_gat_output = None

    def forward_step(
        self,
        obs: torch.Tensor,      # (N, obs_dim)
        a_prev: torch.Tensor,   # (N, act_dim)
        h_prev: torch.Tensor,   # (N, hidden_dim)
        rolling_mean_error: Optional[torch.Tensor] = None,  # (N,) or None
    ) -> tuple:
        hs, nlls, kls, zs, deltas, mus, logvars = [], [], [], [], [], [], []
        for i in range(self.nagents):
            h_i, nll_i, kl_i, z_i, mu_i, logvar_i = self.vrnn_cells[i](obs[i], a_prev[i], h_prev[i])
            hs.append(h_i)
            nlls.append(nll_i)
            kls.append(kl_i)
            zs.append(z_i)
            deltas.append(nll_i.unsqueeze(-1))  # scalar surprise
            mus.append(mu_i)
            logvars.append(logvar_i)

        h_next = torch.stack(hs)
        nlls = torch.stack(nlls)
        kls = torch.stack(kls)
        zs = torch.stack(zs)
        deltas = torch.stack(deltas)  # (N,1)
        mus = torch.stack(mus)
        logvars = torch.stack(logvars)

        # === JSD 기반 neighbor adjacency mask 생성 ===
        mu_now = mus
        logvar_now = logvars
        jsd_mat = pairwise_jsd_gaussian(mu_now, logvar_now, self.nagents)  # (N,N)
        
        # 방법 1: Top-K neighbor selection (더 안정적)
        k_neighbors = min(2, self.nagents - 1)  # 최대 2개 neighbor
        adj = torch.zeros(self.nagents, self.nagents, device=jsd_mat.device)
        
        for i in range(self.nagents):
            jsd_row = jsd_mat[i].clone()
            jsd_row[i] = -float('inf')  # 자기 자신 제외
            # Top-K neighbor 선택
            top_k_indices = torch.topk(jsd_row, k=k_neighbors, largest=True).indices
            for j in top_k_indices:
                adj[i, j] = 1.0
        
        # 방법 2: Mutual selection (대칭성 보장)
        # adj = (adj + adj.t()) > 0  # 상호 선택된 것만 유지

        # Build node features and apply GAT
        V_nodes = torch.cat([h_next, zs], dim=-1)
        if self.use_gat and self.gat is not None:
            if rolling_mean_error is not None:
                delta_for_gat = rolling_mean_error.view(-1, 1)  # (N,1)
            else:
                delta_for_gat = deltas
            
            # Causal GAT vs Standard GAT
            if self.use_causal_gat:
                # Causal GAT: temporal reasoning을 위해 이전 GAT 출력 전달
                # prev_hidden을 head_dim 차원으로 변환
                if hasattr(self, 'prev_gat_output') and self.prev_gat_output is not None:
                    # 이전 GAT 출력을 head_dim으로 변환
                    head_dim = self.gat_dim // 4  # 4 heads
                    prev_gat_output = self.prev_gat_output[:, :head_dim]  # 첫 번째 head 차원만 사용
                else:
                    head_dim = self.gat_dim // 4
                    prev_gat_output = torch.zeros(h_next.shape[0], head_dim, device=h_next.device)
                V_gat = self.gat(V_nodes, adj=adj, delta=delta_for_gat, prev_hidden=prev_gat_output)
                # 현재 GAT 출력을 다음 스텝을 위해 저장
                self.prev_gat_output = V_gat.detach()
            else:
                # Standard GAT
                V_gat = self.gat(V_nodes, adj=adj, delta=delta_for_gat)
        else:
            # GAT 비활성화: 이웃 정보 없음
            V_gat = torch.zeros(h_next.shape[0], self.gat_dim, device=h_next.device)
        
        # Communication reconstruction을 dummy로 설정 (loss 계산용)
        comm_recons = torch.zeros(self.nagents, self.nagents, self.z_dim, device=V_gat.device)

        # 정책 및 가치 계산 (원본 + 개인 + 이웃 정보 분리)
        logits_list, values_list = [], []
        for i in range(self.nagents):
            # obs + h_next + V_gat (원본 관찰 + 개인 상태 + 이웃 집계)
            a2c_in = torch.cat([obs[i], h_next[i], V_gat[i]], dim=-1)
            policy_logits, value = self.a2c_heads[i](a2c_in)
            logits_list.append(policy_logits)
            values_list.append(value)
        logits = torch.stack(logits_list)
        values = torch.stack(values_list)
        ref_logits = torch.zeros_like(logits)

        return h_next, nlls, kls, logits, ref_logits, values, mus, logvars, zs, V_gat, comm_recons
# ---------------------------------------------------------------------------
# 5.  Compute loss
# ---------------------------------------------------------------------------
def pairwise_coop_kl(
    mu: torch.Tensor,        # (B, d)
    logvar: torch.Tensor,    # (B, d)
    n_agents: int
) -> torch.Tensor:
    """
    Compute pairwise KL divergences for diagonal Gaussians, vectorized.
    Returns tensor of shape (T, N, N), where T = B // n_agents.
    """
    B, d = mu.shape
    T = B // n_agents

    # (T, N, d)
    mu_t     = mu.view(T, n_agents, d)
    logvar_t = logvar.view(T, n_agents, d)
    var_t    = logvar_t.exp()

    # (T, N, 1, d) vs (T, 1, N, d)
    mu_i   = mu_t.unsqueeze(2)       # (T, N, 1, d)
    mu_j   = mu_t.unsqueeze(1)       # (T, 1, N, d)
    v_i    = var_t.unsqueeze(2)      # (T, N, 1, d)
    v_j    = var_t.unsqueeze(1)      # (T, 1, N, d)
    lv_i   = logvar_t.unsqueeze(2)   # (T, N, 1, d)
    lv_j   = logvar_t.unsqueeze(1)   # (T, 1, N, d)

    # term1: trace(var_i / var_j)
    term1 = (v_i / v_j).sum(dim=-1)  
    # term2: (mu_j - mu_i)^2 / var_j
    term2 = ((mu_j - mu_i).pow(2) / v_j).sum(dim=-1)
    # term3: logvar_j - logvar_i
    term3 = (lv_j - lv_i).sum(dim=-1)

    # KL matrix: (T, N, N)
    kl_mat = 0.5 * (term1 + term2 + term3 - d)
    return kl_mat

# === 정규분포 간 pairwise JSD 계산 함수 추가 ===
def pairwise_jsd_gaussian(mu: torch.Tensor, logvar: torch.Tensor, n_agents: int) -> torch.Tensor:
    """
    mu, logvar: (N, d)
    return: (N, N) pairwise JSD matrix
    """
    N, d = mu.shape
    # KL(P||Q) 계산 함수
    def kl_gauss(mu1, logvar1, mu2, logvar2):
        var1 = logvar1.exp()
        var2 = logvar2.exp()
        return 0.5 * ((logvar2 - logvar1) + (var1 + (mu1 - mu2).pow(2)) / var2 - 1).sum(-1)
    # pairwise KL(P||Q)
    mu1 = mu.unsqueeze(1)  # (N,1,d)
    mu2 = mu.unsqueeze(0)  # (1,N,d)
    logvar1 = logvar.unsqueeze(1)  # (N,1,d)
    logvar2 = logvar.unsqueeze(0)  # (1,N,d)
    # M = 0.5*(P+Q): mean, var
    m_mu = 0.5 * (mu1 + mu2)
    m_var = 0.5 * (logvar1.exp() + logvar2.exp())
    m_logvar = m_var.log()
    # KL(P||M)
    kl_pm = kl_gauss(mu1, logvar1, m_mu, m_logvar)
    # KL(Q||M)
    kl_qm = kl_gauss(mu2, logvar2, m_mu, m_logvar)
    jsd = 0.5 * (kl_pm + kl_qm)  # (N,N)
    return jsd

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
    """
    Compute VAE loss = nll_coef * mean(nll) 
                   + kl_coef  * mean(kl) 
                   + coop_coef * pairwise_coop_kl(mu, logvar)
    Returns (loss_vae, metrics_vae)
    """
    nll_mean = nll.mean()
    kl_mean  = kl.mean()
    coop_kl  = pairwise_coop_kl(mu, logvar, n_agents).mean()
    loss_vae = nll_coef * nll_mean + kl_coef * kl_mean + coop_coef * coop_kl
    metrics = {
        'vae_nll': nll_mean,
        'vae_kl':  kl_mean,
        'coop_kl': coop_kl,
        'loss_vae': loss_vae
    }
    return loss_vae, metrics

def compute_rl_loss(logits: torch.Tensor,
                    actions: torch.Tensor,
                    adv: torch.Tensor,
                    values_pred: torch.Tensor,
                    returns: torch.Tensor,
                    ent_coef: float,
                    value_coef: float):
    # 1) Policy loss
    logp = F.log_softmax(logits, dim=-1)
    sel_logp = logp.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
    policy_loss = -(sel_logp * adv).mean()

    # 2) Value loss
    value_loss = F.mse_loss(values_pred, returns)
    
    # 3) Entropy bonus
    probs   = torch.softmax(logits, dim=-1)
    entropy = -(probs * logp).sum(-1).mean()

    loss_rl = policy_loss + value_coef * value_loss - ent_coef * entropy
    metrics_rl = {
        'policy_loss': policy_loss,
        'value_loss':  value_loss,
        'entropy':     entropy,
        'loss_rl':     loss_rl
    }
    return loss_rl, metrics_rl

def compute_dpo_loss(
    logp: torch.Tensor,         # log π(a|s) (현재 policy)
    old_logp: torch.Tensor,     # log π_old(a|s) (reference/old policy)
    adv: torch.Tensor,          # advantage
    kl_coef: float              # KL penalty 계수
):
    policy_loss = -(adv * logp).mean()
    kl = (logp - old_logp).mean()
    loss = policy_loss + kl_coef * kl
    return loss, {'policy_loss': policy_loss, 'kl': kl, 'loss': loss}

def compute_rl_ppo_loss(
    logits: torch.Tensor,
    old_logits: torch.Tensor,
    actions: torch.Tensor,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    clip_eps: float,
    ent_coef: float,
    value_coef: float,
    values_pred: torch.Tensor
) -> tuple[torch.Tensor, dict]:
    """
    PPO loss with clipping.
    - logits: new policy logits (B, A)
    - old_logits: old policy logits (B, A)
    - actions: taken actions (B,)
    - advantages: GAE advantages (B,)
    - returns: discounted returns (B,)
    - values_pred: value predictions (B,)
    """
    # 로그확률 계산
    logp      = F.log_softmax(logits, dim=-1)
    old_logp  = F.log_softmax(old_logits, dim=-1)
    sel_logp      = logp.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
    old_sel_logp  = old_logp.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
    ratio     = (sel_logp - old_sel_logp).exp()
    # 클리핑
    clipped_ratio = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
    policy_loss   = -torch.mean(torch.min(ratio * advantages, clipped_ratio * advantages))
    # 가치함수 손실
    value_loss  = F.mse_loss(values_pred, returns)
    # 엔트로피 보너스
    probs   = torch.softmax(logits, dim=-1)
    entropy = -(probs * logp).sum(-1).mean()

    loss = policy_loss + value_coef * value_loss - ent_coef * entropy
    metrics = {
        'policy_loss': policy_loss,
        'value_loss':  value_loss,
        'entropy':     entropy,
        'loss_rl':     loss
    }
    return loss, metrics

def compute_comm_loss(zs: torch.Tensor, comm_recons: torch.Tensor, n_agents: int) -> torch.Tensor:
    """
    Communication loss를 0으로 설정 (GAT만 사용하는 경우)
    """
    # Communication을 제거했으므로 loss도 0
    return zs.new_zeros(())

def adaptive_loss_coefficients(vae_loss, rl_loss, comm_loss):
    total_loss = vae_loss + rl_loss + comm_loss
    # 각 loss의 비율에 따라 coefficient 조정
    vae_coef = 1.0 / (vae_loss / total_loss + 1e-8)
    rl_coef = 1.0 / (rl_loss / total_loss + 1e-8)
    comm_coef = 1.0 / (comm_loss / total_loss + 1e-8)
    return vae_coef, rl_coef, comm_coef
# ---------------------------------------------------------------------------
# 6.  Config‑driven instantiation example
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import yaml, argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=r"C:\Users\Chamjin Joo\Desktop\cham\Etri_project\MADP\configs.yaml")
    parser.add_argument("--env",    type=str, default="dectiger")
    args = parser.parse_args()

    cfg  = yaml.safe_load(Path(args.config).read_text())
    task = args.env              
    task_cfg = cfg[task]     

    obs_dim    = task_cfg["obs_dim"]
    hidden_dim = task_cfg.get("hidden_dim", 2 * obs_dim)
    z_dim      = task_cfg.get("z_dim",      obs_dim // 2)
    gat_dim    = task_cfg.get("gat_dim",    2 * hidden_dim)

    model = VRNNGATA2C(
        obs_dim  = obs_dim,
        act_dim  = task_cfg["act_dim"],
        hidden_dim = hidden_dim,
        z_dim      = z_dim,
        gat_dim    = gat_dim,
        n_agents   = task_cfg["n_agents"],
        use_gat    = True,  # GAT ablation
        use_causal_gat = True,  # Causal GAT 사용 옵션
    )
    print(model)

# ---------------------------------------------------------------------------
# 2.5. Hierarchical Reasoning Network
# ---------------------------------------------------------------------------

class HierarchicalReasoningNetwork(nn.Module):
    """Hierarchical reasoning for cooperative decision making."""
    def __init__(self, obs_dim: int, hidden_dim: int, n_agents: int):
        super().__init__()
        self.n_agents = n_agents
        
        # Level 1: Individual reasoning
        self.individual_reasoner = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Level 2: Local group reasoning
        self.local_reasoner = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Level 3: Global coordination reasoning
        self.global_reasoner = nn.Sequential(
            nn.Linear(hidden_dim * n_agents, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Reasoning fusion
        self.reasoning_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def forward(self, obs: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        obs: (N, obs_dim)
        adj: (N, N) adjacency matrix
        return: (N, hidden_dim) reasoning features
        """
        N = obs.shape[0]
        
        # Level 1: Individual reasoning
        individual_features = self.individual_reasoner(obs)  # (N, hidden_dim)
        
        # Level 2: Local group reasoning
        local_features = []
        for i in range(N):
            # 이웃들과의 local reasoning
            neighbors = []
            for j in range(N):
                if adj[i, j] > 0 and i != j:
                    neighbors.append(individual_features[j])
            
            if len(neighbors) > 0:
                # 이웃들과의 pairwise reasoning
                neighbor_feats = torch.stack(neighbors)  # (num_neighbors, hidden_dim)
                local_input = torch.cat([individual_features[i], neighbor_feats.mean(dim=0)], dim=-1)
                local_feat = self.local_reasoner(local_input)
            else:
                local_feat = individual_features[i]
            
            local_features.append(local_feat)
        local_features = torch.stack(local_features)  # (N, hidden_dim)
        
        # Level 3: Global coordination reasoning
        global_input = individual_features.flatten()  # (N * hidden_dim)
        global_features = self.global_reasoner(global_input)  # (hidden_dim)
        global_features = global_features.unsqueeze(0).expand(N, -1)  # (N, hidden_dim)
        
        # Fusion of all reasoning levels
        fused_input = torch.cat([individual_features, local_features, global_features], dim=-1)
        reasoning_features = self.reasoning_fusion(fused_input)  # (N, hidden_dim)
        
        return reasoning_features
