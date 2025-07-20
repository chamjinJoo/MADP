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

import numpy as np

# --- Encoder ---
class Encoder(nn.Module):
    def __init__(self, n_in, n_hid, n_out, rel_rec, rel_send, factor=True):
        super().__init__()
        self.factor = factor
        
        # MLP layers following ACDrepo structure
        self.mlp1 = nn.Sequential(
            nn.Linear(n_in, n_hid),
            nn.ReLU(),
            nn.Linear(n_hid, n_hid)
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(n_hid * 2, n_hid),
            nn.ReLU(),
            nn.Linear(n_hid, n_hid)
        )
        self.mlp3 = nn.Sequential(
            nn.Linear(n_hid, n_hid),
            nn.ReLU(),
            nn.Linear(n_hid, n_hid)
        )
        
        if self.factor:
            self.mlp4 = nn.Sequential(
                nn.Linear(n_hid * 3, n_hid),
                nn.ReLU(),
                nn.Linear(n_hid, n_hid)
            )
            print("Using factor graph MLP encoder.")
        else:
            self.mlp4 = nn.Sequential(
                nn.Linear(n_hid * 2, n_hid),
                nn.ReLU(),
                nn.Linear(n_hid, n_hid)
            )
            print("Using MLP encoder.")
        
        self.fc_out = nn.Linear(n_hid, n_out)
        self.rel_rec = rel_rec
        self.rel_send = rel_send

    def node2edge(self, x):
        # x: [batch, n_agents, n_in]
        receivers = torch.matmul(self.rel_rec, x)
        senders = torch.matmul(self.rel_send, x)
        edges = torch.cat([senders, receivers], dim=-1)
        return edges

    def edge2node(self, edges):
        # edges: [batch, n_edges, n_hid] (for Encoder)
        agg = torch.matmul(self.rel_rec.t(), edges)  # [batch, n_agents, n_hid]
        norm = self.rel_rec.sum(0).unsqueeze(0).unsqueeze(-1)  # [1, n_agents, 1]
        return agg / (norm + 1e-6)

    def forward(self, x):
        # x: [batch, n_agents, n_in]
        x = self.mlp1(x)  # 2-layer ELU net per node
        x = self.node2edge(x)
        x = self.mlp2(x)
        x_skip = x

        if self.factor:
            x = self.edge2node(x)
            x = self.mlp3(x)
            x = self.node2edge(x)
            x = torch.cat((x, x_skip), dim=2)  # Skip connection
            x = self.mlp4(x)
        else:
            x = self.mlp3(x)
            x = torch.cat((x, x_skip), dim=2)  # Skip connection
            x = self.mlp4(x)

        return self.fc_out(x)  # [batch, n_edges, n_out]

# --- Decoder ---
class Decoder(nn.Module):
    def __init__(self, n_in, n_hid, n_out, rel_rec, rel_send, edge_types, skip_first=False):
        super().__init__()
        self.edge_types = edge_types
        self.skip_first_edge_type = skip_first
        
        # Separate MLPs for each edge type (following ACDrepo)
        self.msg_fc1 = nn.ModuleList([
            nn.Linear(n_in * 2, n_hid) for _ in range(edge_types)
        ])
        self.msg_fc2 = nn.ModuleList([
            nn.Linear(n_hid, n_out) for _ in range(edge_types)
        ])
        
        self.msg_out_shape = n_out
        
        # Output MLP
        self.out_fc1 = nn.Linear(n_in + n_out, n_hid)
        self.out_fc2 = nn.Linear(n_hid, n_hid)
        self.out_fc3 = nn.Linear(n_hid, n_out)  # Output obs_dim + 1 instead of n_in
        
        self.rel_rec = rel_rec
        self.rel_send = rel_send
        self.dropout_prob = 0.0

    def node2edge(self, x):
        receivers = torch.matmul(self.rel_rec, x)
        senders = torch.matmul(self.rel_send, x)
        edges = torch.cat([senders, receivers], dim=-1)
        return edges

    def edge2node(self, edges):
        # edges: [batch, n_edges, msg_out_shape] (for Decoder)
        # Aggregate all msgs to receiver following ACDrepo's approach
        agg_msgs = torch.matmul(self.rel_rec.t(), edges)  # [batch, n_agents, msg_out_shape]
        return agg_msgs

    def forward(self, x, rel_type):
        # x: [batch, n_agents, n_in]
        # rel_type: [batch, n_edges, edge_types]
        
        # Node2edge
        pre_msg = self.node2edge(x)
        
        all_msgs = torch.zeros(
            pre_msg.size(0), pre_msg.size(1), self.msg_out_shape,
            device=pre_msg.device
        )

        if self.skip_first_edge_type:
            start_idx = 1
        else:
            start_idx = 0

        # Run separate MLP for every edge type (following ACDrepo)
        for i in range(start_idx, len(self.msg_fc2)):
            msg = F.relu(self.msg_fc1[i](pre_msg))
            msg = F.dropout(msg, p=self.dropout_prob, training=self.training)
            msg = F.relu(self.msg_fc2[i](msg))  # [batch, n_edges, n_out]
            # Fix: Expand rel_type to match msg dimensions
            weight = rel_type[:, :, i : i + 1]  # [batch, n_edges, 1]
            msg = msg * weight  # Apply relationship weights
            all_msgs += msg

        # Aggregate all msgs to receiver
        agg_msgs = self.edge2node(all_msgs)

        # Skip connection
        aug_inputs = torch.cat([x, agg_msgs], dim=-1)

        # Output MLP
        pred = F.dropout(F.relu(self.out_fc1(aug_inputs)), p=self.dropout_prob, training=self.training)
        pred = F.dropout(F.relu(self.out_fc2(pred)), p=self.dropout_prob, training=self.training)
        pred = self.out_fc3(pred)

        # Return predicted obs + reward directly
        return pred  # [batch, n_agents, obs_dim + 1]

# ---------------------------------------------------------------------------
# 1. ACD 기반 VAE
# ---------------------------------------------------------------------------
class ACD_VAE(nn.Module):
    def __init__(self, args, n_agents, obs_dim, rnn_hidden_dim, scm_hidden_dim, edge_types, do_prob=0.0, factor=True):
        super().__init__()
        self.n_agents = n_agents  # 실제 agent 수 N
        self.obs_dim = obs_dim
        self.rnn_hidden_dim = rnn_hidden_dim
        self.scm_hidden_dim = scm_hidden_dim
        self.edge_types = edge_types  # 인과 edge type 개수
        self.total_nodes = 2 * n_agents  # 관측 노드 + 보상 노드
        self.rel_rec, self.rel_send = self.create_rel_matrices(self.total_nodes)
        self.encoder = Encoder(rnn_hidden_dim,      # n_in
                                scm_hidden_dim,     # n_hid
                                self.edge_types,    # n_out
                                self.rel_rec,
                                self.rel_send,
                                factor=factor)
        self.decoder = Decoder(rnn_hidden_dim,      # n_in (각 노드별 hidden)
                                scm_hidden_dim,     # n_hid
                                obs_dim + 1,        # n_out
                                self.rel_rec,
                                self.rel_send,
                                self.edge_types)
        # 관측/보상 RNN 분리
        self.obs_rnn = nn.GRU(input_size=obs_dim, hidden_size=rnn_hidden_dim, batch_first=True)
        self.rew_rnn = nn.GRU(input_size=1, hidden_size=rnn_hidden_dim, batch_first=True)

    def create_rel_matrices(self, n_nodes):
        off_diag = np.ones([n_nodes, n_nodes]) - np.eye(n_nodes)
        rel_rec = np.array(self.encode_onehot(np.where(off_diag)[0], n_nodes))
        rel_send = np.array(self.encode_onehot(np.where(off_diag)[1], n_nodes))
        return torch.FloatTensor(rel_rec), torch.FloatTensor(rel_send)

    def encode_onehot(self, idx, n):
        onehot = np.zeros((len(idx), n))
        onehot[np.arange(len(idx)), idx] = 1
        return onehot

    def forward(self, obs, reward, prev_hidden_obs, prev_hidden_rew):
        # obs: [batch, N, obs_dim], reward: [batch, N, 1]
        batch_size, N, obs_dim = obs.shape
        # 관측/보상 RNN 분리 (입력과 hidden을 [N, ...]로 flatten)
        obs_flat = obs.view(-1, 1, obs_dim)  # [N, 1, obs_dim]
        prev_hidden_obs_flat = prev_hidden_obs  # [1, N, rnn_hidden_dim]
        rnn_out_obs, h_out_obs = self.obs_rnn(obs_flat, prev_hidden_obs_flat)  # rnn_out_obs: [N, 1, rnn_hidden_dim]
        rnn_out_obs = rnn_out_obs.squeeze(1)               # (batch_size*N, hidden)
        rnn_out_obs = rnn_out_obs.view(batch_size, N, -1)  # [1, N, rnn_hidden_dim]
        # 보상 RNN도 동일하게
        reward_flat = reward.view(-1, 1, 1)  # [N, 1, 1]
        prev_hidden_rew_flat = prev_hidden_rew  # [1, N, rnn_hidden_dim]
        rnn_out_rew, h_out_rew = self.rew_rnn(reward_flat, prev_hidden_rew_flat)  # [N, 1, rnn_hidden_dim]
        rnn_out_rew = rnn_out_rew.squeeze(1)
        rnn_out_rew = rnn_out_rew.view(batch_size, N, -1)  # [1, N, rnn_hidden_dim]
        # 2N 노드로 합치기
        hidden = torch.cat([rnn_out_obs, rnn_out_rew], dim=1)  # [batch, 2N, rnn_hidden_dim]
        rel_type_logits = self.encoder(hidden)
        rel_type = torch.softmax(rel_type_logits, dim=-1)
        print(rel_type)
        pred = self.decoder(hidden, rel_type)
        return pred, rel_type, h_out_obs, h_out_rew

# ---------------------------------------------------------------------------
# 2. Multi-Agent Actor-Critic (with ACD_VAE & GAT)
# ---------------------------------------------------------------------------
class ACD_VAE_A2C(nn.Module):
    def __init__(self, args, obs_dim, action_dim, num_agents, scm_hidden_dim, rnn_hidden_dim, edge_types=2, num_heads=4, dropout=0.1):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.num_agents = num_agents
        self.n_hid = scm_hidden_dim
        self.edge_types = edge_types
        self.hidden_size = rnn_hidden_dim
        self.total_nodes = 2 * num_agents
        # SCM (ACD 기반)
        self.scm = ACD_VAE(args, num_agents, obs_dim, rnn_hidden_dim, scm_hidden_dim, edge_types)
        # Actor/Critic 네트워크 (관측 노드만)
        input_dim = 2 * obs_dim
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
        # 메시지 MLPs: 관측/보상 노드용 2개
        self.message_mlps = nn.ModuleList([
            nn.Linear(rnn_hidden_dim, obs_dim),  # 관측 노드에서 온 메시지
            nn.Linear(rnn_hidden_dim, obs_dim),  # 보상 노드에서 온 메시지
        ])
        # 관측/보상 RNN 분리
        self.obs_rnn = nn.GRU(input_size=obs_dim, hidden_size=rnn_hidden_dim, batch_first=True)
        self.rew_rnn = nn.GRU(input_size=1, hidden_size=rnn_hidden_dim, batch_first=True)

    def forward(self, obs, reward, prev_hidden_obs, prev_hidden_rew):
        batch_size, N, obs_dim = obs.shape
        # 관측/보상 RNN 분리 (입력과 hidden을 [N, ...]로 flatten)
        obs_flat = obs.view(-1, 1, obs_dim)  # [N, 1, obs_dim]
        prev_hidden_obs_flat = prev_hidden_obs  # [1, N, rnn_hidden_dim]
        rnn_out_obs, h_out_obs = self.obs_rnn(obs_flat, prev_hidden_obs_flat)  # [N, 1, rnn_hidden_dim]
        rnn_out_obs = rnn_out_obs.view(batch_size, N, -1)  # [1, N, rnn_hidden_dim]
        reward_flat = reward.view(-1, 1, 1)  # [N, 1, 1]
        prev_hidden_rew_flat = prev_hidden_rew  # [1, N, rnn_hidden_dim]
        rnn_out_rew, h_out_rew = self.rew_rnn(reward_flat, prev_hidden_rew_flat)  # [N, 1, rnn_hidden_dim]
        rnn_out_rew = rnn_out_rew.view(batch_size, N, -1)  # [1, N, rnn_hidden_dim]
        # 2N 노드로 합치기
        hidden = torch.cat([rnn_out_obs, rnn_out_rew], dim=1)  # [batch, 2N, rnn_hidden_dim]
        scm_pred, causal_structure, _, _ = self.scm.forward(obs, reward, prev_hidden_obs, prev_hidden_rew)
        causal_structure_detached = causal_structure.detach()
        # causal_structure: [batch, n_edges, edge_types]
        # causal_matrix: [batch, 2N, 2N, edge_types]
        causal_matrix = torch.zeros(batch_size, 2*N, 2*N, self.edge_types, device=causal_structure.device)
        edge_idx = 0
        for i in range(2*N):
            for j in range(2*N):
                if i != j:
                    causal_matrix[:, i, j, :] = causal_structure_detached[:, edge_idx, :]
                    edge_idx += 1
        # 메시지 패싱 (관측 노드에만 메시지 집계)
        message_obs = torch.zeros(batch_size, N, self.obs_dim, device=obs.device)
        for tgt in range(N):  # 관측 노드만
            for src in range(2*N):
                if src == tgt:
                    continue
                src_type = 0 if src < N else 1  # 0: 관측, 1: 보상
                src_hidden = hidden[:, src, :]  # [batch, rnn_hidden_dim]
                edge_weight = causal_matrix[:, src, tgt, 1].unsqueeze(-1)  # edge 존재 확률
                msg = self.message_mlps[src_type](src_hidden)
                message_obs[:, tgt, :] += msg * edge_weight
        # actor/critic 입력
        actor_input = torch.cat([obs, message_obs], dim=-1)  # [batch, N, 2*obs_dim]
        actor_outputs = torch.stack([
            self.actor_networks[i](actor_input[:, i, :]) for i in range(self.num_agents)
        ], dim=1)
        critic_outputs = torch.stack([
            self.critic_networks[i](actor_input[:, i, :]) for i in range(self.num_agents)
        ], dim=1)
        return {
            'actor_outputs': actor_outputs,
            'critic_outputs': critic_outputs,
            'scm_predictions': scm_pred,
            'causal_structure': causal_structure,
            'hidden_obs': h_out_obs,
            'hidden_rew': h_out_rew
        }

    def forward_scm_only(self, obs, reward, prev_hidden):
        """Separate forward pass for SCM training only"""
        batch_size, n_agents, obs_dim = obs.shape
        x = torch.cat([obs, reward], dim=-1)  # [batch, n_agents, obs_dim+1]
        x = x.view(batch_size * n_agents, 1, x.shape[-1])  # [batch*n_agents, 1, obs_dim+1]
        rnn_out, h_out = self.rnn(x, prev_hidden)
        hidden = rnn_out.squeeze(1).view(batch_size, n_agents, -1)  # [batch, n_agents, rnn_hidden_dim]
        scm_pred, causal_structure = self.scm(hidden)
        
        return {
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
        output = output[..., :nagents, :]  # 관측 노드만 loss 계산
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