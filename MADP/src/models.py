from __future__ import annotations
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
# ---------------------------------------------------------------------------
# 1.  VRNN building blocks
# ---------------------------------------------------------------------------
class VRNNCell(nn.Module):
    """Single‑step Variational RNN cell (Chung et al., 2015).

    Prior, encoder, decoder are simple MLPs. Replace with CNNs/TCNs for images.
    """
    LOGVAR_CLAMP = 10.0

    def __init__(
        self, 
        x_dim: int,
        a_dim: int, 
        h_dim: int, 
        z_dim: int, 
    ) -> None:
        super().__init__()
        self.x_dim = x_dim
        self.z_dim = z_dim
        # RNN core
        self.rnn = nn.GRUCell(x_dim + a_dim + z_dim, h_dim)
        
        # Prior p(z_t | h_{t-1})
        self.prior_net = nn.Sequential(
            nn.Linear(h_dim, h_dim), nn.ReLU(), 
            nn.Linear(h_dim, h_dim), nn.ReLU(), 
            nn.Linear(h_dim, z_dim * 2)
        )
        # Encoder q(z_t | x_t, h_{t-1})
        self.enc_net = nn.Sequential(
            nn.Linear(x_dim + h_dim, h_dim), nn.ReLU(), 
            nn.Linear(h_dim, h_dim), nn.ReLU(),
            nn.Linear(h_dim, z_dim * 2)
        )
        # Decoder p(x_t | z_t, h_{t-1}) — predicts μ and log σ² (diagonal)
        out_dim = x_dim * 2
        self.dec_net = nn.Sequential(
            nn.Linear(z_dim + h_dim, h_dim), nn.ReLU(),
            nn.Linear(h_dim, h_dim), nn.ReLU(),
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
        enc_stats = self.enc_net(torch.cat([x_t, h_prev], dim=-1))
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
        inv_var = (-logvar_x).exp() # TODO: clamping
        nll = 0.5 * ( 
            ((x_t - mu_x).pow(2) * inv_var + logvar_x + math.log(2 * math.pi))
        ).sum(dim=-1)
        
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
        rnn_in = [x_t, z_t, a_prev]
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
  
    def forward(self, V: torch.Tensor, adj: torch.Tensor | None = None) -> torch.Tensor:
        """
        V: (N, in_dim)
        adj: (N, N) adjacency mask (1 for edge, 0 for no edge). If provided, blocks attention.
        """
        N = V.size(0)
        # --- 1st Attention Layer ---
        Wh1 = self.W1(V)  # (N, hid_dim)
        # compute attention scores via source and dest projections
        src1 = self.a_src1(Wh1)  # (N,1)
        dst1 = self.a_dst1(Wh1)  # (N,1)
        e1 = self.leaky_relu(src1 + dst1.transpose(0, 1))  # (N,N)
        if adj is not None:
            e1 = e1.masked_fill(adj == 0, float('-inf'))
        alpha1 = self.softmax(e1)  # (N,N)
        alpha1 = self.dropout(alpha1)
        H1 = alpha1 @ Wh1          # (N, hid_dim)
        H1 = F.elu(H1)             # (N, hid_dim)

        # --- 2nd Attention Layer ---
        Wh2 = self.W2(H1)  # (N, out_dim)
        src2 = self.a_src2(Wh2)  # (N,1)
        dst2 = self.a_dst2(Wh2)  # (N,1)
        e2 = self.leaky_relu(src2 + dst2.transpose(0, 1))  # (N,N)
        if adj is not None:
            e2 = e2.masked_fill(adj == 0, float('-inf'))
        alpha2 = self.softmax(e2)  # (N,N)
        alpha2 = self.dropout(alpha2)
        H2 = alpha2 @ Wh2          # (N, out_dim)
        H2 = F.elu(H2)             # (N, out_dim)

        return H2
# ---------------------------------------------------------------------------
# 3.  Actor‑Critic heads
# ---------------------------------------------------------------------------

class ActorCriticHead(nn.Module):
    def __init__(self, in_dim: int, act_dim: int) -> None:
        super().__init__()
        self.actor = nn.Sequential(nn.Linear(in_dim, in_dim), nn.ReLU(), 
                                   nn.Linear(in_dim, in_dim), nn.ReLU(), 
                                   nn.Linear(in_dim, act_dim))
        self.critic = nn.Sequential(nn.Linear(in_dim, in_dim), nn.ReLU(), 
                                    nn.Linear(in_dim, in_dim), nn.ReLU(),
                                    nn.Linear(in_dim, 1))

    def forward(self, feats: torch.Tensor):
        logits = self.actor(feats)
        value = self.critic(feats).squeeze(-1)
        return logits, value

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
    ) -> None:
        super().__init__()
        self.nagents = n_agents
        
        self.act_dim = act_dim
        self.hidden_dim = hidden_dim
        self.z_dim = z_dim
        self.gat_dim = gat_dim

        self.vrnn_cells = nn.ModuleList(
            [VRNNCell(obs_dim, act_dim, hidden_dim, z_dim) for _ in range(n_agents)]
        )
     
        self.gat = GATLayer(hidden_dim + z_dim + 1, gat_dim, gat_dim)  # last dim = δ scalar
        self.policy_heads = nn.ModuleList(
            # [ActorCriticHead(gat_dim + obs_dim, act_dim) for _ in range(n_agents)]
            # [ActorCriticHead(gat_dim + hidden_dim, act_dim) for _ in range(n_agents)]  # option 2. 𝜋(a|V,h)
            [ActorCriticHead(gat_dim + z_dim, act_dim) for _ in range(n_agents)]  # option 3. 𝜋(a|V,z)
        )
    def forward_step(
        self,
        obs: torch.Tensor,      # (N, obs_dim)
        a_prev: torch.Tensor,   # (N, act_dim)
        h_prev: torch.Tensor,   # (N, hidden_dim)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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

        # Build node features and apply GAT
        V_nodes = torch.cat([h_next, zs, deltas], dim=-1)
        V_gat = self.gat(V_nodes)
        


        logits_list, values_list = [], []
        for i in range(self.nagents):
            # pi_in = torch.cat([V_gat[i], obs[i]], dim=-1)       # option 1. 𝜋(a|V,o)
            # pi_in = torch.cat([V_gat[i], h_next[i]], dim=-1)  # option 2. 𝜋(a|V,h)
            pi_in = torch.cat([V_gat[i], zs[i]], dim=-1)      # option 3. 𝜋(a|V,z)
            logits, value = self.policy_heads[i](pi_in)
            logits_list.append(logits)
            values_list.append(value)
        logits = torch.stack(logits_list)
        values = torch.stack(values_list)

        return h_next, nlls, kls, logits, values, mus, logvars

# ---------------------------------------------------------------------------
# 5.  Compute loss
# ---------------------------------------------------------------------------
def pairwise_coop_kl(mu: torch.Tensor, logvar: torch.Tensor, n_agents: int) -> torch.Tensor:
    """
    Compute per-timestep pairwise KL divergences for diagonal Gaussian posteriors.
    """
    B, d = mu.shape
    T = B // n_agents
    # (T, N, d)
    mu_t     = mu.view(T, n_agents, d)
    logvar_t = logvar.view(T, n_agents, d)
    var_t    = torch.exp(logvar_t)

    kl_mats = []
    for t in range(T):
        m = mu_t[t]       # (N, d)
        lv = logvar_t[t]  # (N, d)
        v = var_t[t]      # (N, d)
        # expand for pairwise
        m_i = m.unsqueeze(1)      # (N,1,d)
        m_j = m.unsqueeze(0)      # (1,N,d)
        v_i = v.unsqueeze(1)      # (N,1,d)
        v_j = v.unsqueeze(0)      # (1,N,d)
        lv_i = lv.unsqueeze(1)
        lv_j = lv.unsqueeze(0)

        term1 = (v_i / v_j).sum(-1)               # (N,N)
        term2 = ((m_j - m_i).pow(2) / v_j).sum(-1)
        term3 = (lv_j - lv_i).sum(-1)
        kl_mat = 0.5 * (term1 + term2 + term3 - d) # (N,N)
        kl_mats.append(kl_mat)
    # stack into (T, N, N)
    return torch.stack(kl_mats, dim=0)

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
    task_cfg = cfg[task]           # 해당 블록 딕셔너리

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
    )

    print(model)