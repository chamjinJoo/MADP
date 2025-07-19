from __future__ import annotations
from typing import Optional, Dict, Any
import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
# ACD 모듈 경로를 sys.path에 추가
acd_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../ACD_repo/causal-marl/ACD/model"))
if acd_path not in sys.path:
    sys.path.append(acd_path)
# src 경로를 sys.path에 추가
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if src_path not in sys.path:
    sys.path.append(src_path)
from src.forward_pass_and_eval import forward_pass_and_eval
from ACDrepo.ACD.model.MLPEncoder import MLPEncoder
from ACDrepo.ACD.model.MLPDecoder import MLPDecoder

import numpy as np

# --- SimpleRNNEncoder ---
class SimpleRNNEncoder(nn.Module):
    def __init__(self, n_in, n_hid, n_out, rel_rec, rel_send):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(n_in * 2, n_hid),
            nn.ReLU(),
            nn.Linear(n_hid, n_out)
        )
        self.rel_rec = rel_rec
        self.rel_send = rel_send

    def node2edge(self, x):
        # x: [batch, n_agents, n_in]
        receivers = torch.matmul(self.rel_rec, x)
        senders = torch.matmul(self.rel_send, x)
        edges = torch.cat([senders, receivers], dim=-1)
        return edges

    def forward(self, x):
        # x: [batch, n_agents, n_in]
        edges = self.node2edge(x)
        out = self.mlp(edges)
        return out  # [batch, n_edges, n_out]

# --- SimpleRNNDecoder ---
class SimpleRNNDecoder(nn.Module):
    def __init__(self, n_in, n_hid, n_out, rel_rec, rel_send):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(n_in * 2, n_hid),
            nn.ReLU(),
            nn.Linear(n_hid, n_out)
        )
        self.rel_rec = rel_rec
        self.rel_send = rel_send

    def node2edge(self, x):
        receivers = torch.matmul(self.rel_rec, x)
        senders = torch.matmul(self.rel_send, x)
        edges = torch.cat([senders, receivers], dim=-1)
        return edges

    def edge2node(self, edges):
        # edges: [batch, n_agents, n_out]
        agg = torch.matmul(self.rel_rec.t(), edges)  # [batch, n_agents, n_out]
        norm = self.rel_rec.sum(0).unsqueeze(0).unsqueeze(-1)  # [1, n_agents, 1]
        return agg / (norm + 1e-6)

    def forward(self, x, rel_type):
        # x: [batch, n_agents, n_in]
        # rel_type: [batch, n_edges, edge_types]
        edges = self.node2edge(x)
        # rel_type softmax 결과를 가중치로 곱해 causal message를 만들 수도 있음
        msg = self.mlp(edges)
        agg = self.edge2node(msg)
        return agg  # [batch, n_agents, n_out]

# ---------------------------------------------------------------------------
# 1. ACD 기반 SCM 래퍼
# ---------------------------------------------------------------------------
class ACD_SCM(nn.Module):
    def __init__(self, args, n_agents, obs_dim, rnn_hidden_dim, scm_hidden_dim, edge_types, do_prob=0.0, factor=True):
        super().__init__()
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.rnn_hidden_dim = rnn_hidden_dim
        self.scm_hidden_dim = scm_hidden_dim
        self.edge_types = edge_types * 4
        self.rel_rec, self.rel_send = self.create_rel_matrices(n_agents)
        self.encoder = SimpleRNNEncoder(rnn_hidden_dim, scm_hidden_dim, self.edge_types, self.rel_rec, self.rel_send)
        self.decoder = SimpleRNNDecoder(rnn_hidden_dim, scm_hidden_dim, obs_dim + 1, self.rel_rec, self.rel_send)

    def create_rel_matrices(self, n_nodes):
        # n_nodes = n_agents (각 agent가 obs+reward를 포함)
        off_diag = np.ones([n_nodes, n_nodes]) - np.eye(n_nodes)
        rel_rec = np.array(self.encode_onehot(np.where(off_diag)[0], n_nodes))
        rel_send = np.array(self.encode_onehot(np.where(off_diag)[1], n_nodes))
        return torch.FloatTensor(rel_rec), torch.FloatTensor(rel_send)

    def encode_onehot(self, idx, n):
        onehot = np.zeros((len(idx), n))
        onehot[np.arange(len(idx)), idx] = 1
        return onehot

    def forward(self, hidden):
        # hidden: [batch, n_agents, rnn_hidden_dim]
        rel_type_logits = self.encoder(hidden)
        rel_type = torch.softmax(rel_type_logits, dim=-1)
        pred = self.decoder(hidden, rel_type)
        return pred, rel_type

    # def get_causal_structure(self, in_series):
    #     # in_series: [batch, n_agents, timesteps, obs_dim+1]
    #     batch_size, n_agents, timesteps, features = in_series.shape
        
    #     rel_type_logits = self.encoder(in_series, self.rel_rec, self.rel_send)
    #     rel_type = torch.softmax(rel_type_logits, dim=-1)
    #     return rel_type

# ---------------------------------------------------------------------------
# 2. Multi-Agent Actor-Critic (with ACD_SCM & GAT)
# ---------------------------------------------------------------------------
class MultiAgentActorCritic(nn.Module):
    def __init__(self, args, obs_dim, action_dim, num_agents, scm_hidden_dim, rnn_hidden_dim, edge_types=2, gat_type='none', gat_dim=32, num_heads=4, dropout=0.1):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.num_agents = num_agents
        self.gat_type = gat_type
        self.gat_dim = gat_dim
        self.n_hid = scm_hidden_dim
        self.edge_types = edge_types * 4
        self.hidden_size = rnn_hidden_dim
        # SCM (ACD 기반)
        self.rnn = nn.GRU(input_size=obs_dim+1, hidden_size=rnn_hidden_dim, batch_first=True)
        self.scm = ACD_SCM(args, num_agents, obs_dim, rnn_hidden_dim, scm_hidden_dim, edge_types)
        # Actor/Critic 네트워크
        input_dim = 2*obs_dim # obs + message
        self.actor_networks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, input_dim),
                nn.ReLU(),
                nn.Linear(input_dim, action_dim)
            ) for _ in range(num_agents)
        ])
        self.critic_networks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, input_dim),
                nn.ReLU(),
                nn.Linear(input_dim, 1)
            ) for _ in range(num_agents)
        ])
        # attention-like message MLPs
        self.message_mlps = nn.ModuleList([
            nn.Linear(obs_dim, obs_dim) for _ in range(self.edge_types)
        ])

    def forward(self, obs, reward, prev_hidden):
        batch_size, n_agents, obs_dim = obs.shape
        x = torch.cat([obs, reward], dim=-1)  # [batch, n_agents, obs_dim+1]
        x = x.view(batch_size * n_agents, 1, x.shape[-1])  # [batch*n_agents, 1, obs_dim+1]
        rnn_out, h_out = self.rnn(x, prev_hidden)
        hidden = rnn_out.squeeze(1).view(batch_size, n_agents, -1)  # [batch, n_agents, rnn_hidden_dim]
        scm_pred, causal_structure = self.scm(hidden)
        # message passing (attention-like weighted sum)
        causal_matrix = torch.zeros(batch_size, n_agents, n_agents, self.edge_types, device=causal_structure.device)
        edge_idx = 0
        for i in range(n_agents):
            for j in range(n_agents):
                if i != j:
                    causal_matrix[:, i, j, :] = causal_structure[:, edge_idx, :]
                    edge_idx += 1
        message_obs = torch.zeros(batch_size, n_agents, obs_dim, device=obs.device)
        for i in range(n_agents):
            for j in range(n_agents):
                if i == j:
                    continue
                obs_j = obs[:, j, :]  # [batch, obs_dim]
                for k in range(self.edge_types):
                    msg_k = self.message_mlps[k](obs_j)  # [batch, obs_dim]
                    weight = causal_matrix[:, i, j, k].unsqueeze(-1)  # [batch, 1]
                    message_obs[:, i, :] += msg_k * weight
        message = message_obs  # [batch, n_agents, obs_dim]
        current_obs = obs  # [batch, n_agents, obs_dim]
        actor_input = torch.cat([current_obs, message], dim=-1)  # [batch, n_agents, 2*obs_dim]
        actor_outputs = torch.stack([
            self.actor_networks[i](actor_input[:, i, :]) for i in range(self.num_agents)
        ], dim=1)  # [batch, n_agents, action_dim]
        critic_outputs = torch.stack([
            self.critic_networks[i](actor_input[:, i, :]) for i in range(self.num_agents)
        ], dim=1)  # [batch, n_agents, 1]
        return {
            'actor_outputs': actor_outputs,
            'critic_outputs': critic_outputs,
            'scm_predictions': scm_pred,
            'causal_structure': causal_structure,
            'hidden': h_out
        }

    def compute_a2c_loss(self, in_series, action_series, reward_series, adj_matrix, current_data): 
        pass


    def compute_ACD_loss(self, prob, output, target, nagents, logits=None, relations=None):
        import torch.nn.functional as F
        from ACDrepo.ACD.model.utils import kl_categorical
        losses = {}
        losses['loss_nll'] = F.mse_loss(output, target)
        edge_types = self.edge_types 
        log_prior = torch.log(torch.ones(edge_types) / edge_types).to(prob.device)
        predicted_atoms = nagents
        losses['loss_kl'] = kl_categorical(prob, log_prior, predicted_atoms)
        losses['loss'] = losses['loss_nll'] + losses['loss_kl']
        # # 기타 metric
        # if relations is not None:
        #     acc = (logits.argmax(-1) == relations.argmax(-1)).float().mean()
        # else:
        #     acc = (logits.argmax(-1) == logits.argmax(-1)).float().mean()
        # losses['acc'] = acc
        # losses['auroc'] = torch.tensor(0.0, device=logits.device)  # 필요시 구현
        return losses