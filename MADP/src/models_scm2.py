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
from ACDrepo.ACD.model.MLPEncoder import MLPEncoder
from ACDrepo.ACD.model.MLPDecoder import MLPDecoder

import numpy as np

# ---------------------------------------------------------------------------
# 1. ACD 기반 SCM 래퍼
# ---------------------------------------------------------------------------
class ACD_SCM(nn.Module):
    def __init__(self, args, n_agents, obs_dim, n_hid, edge_types, episode_length=10, do_prob=0.0, factor=True):
        super().__init__()
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.n_hid = n_hid
        self.edge_types = edge_types
        self.episode_length = episode_length
        n_in = (episode_length + 1) * (obs_dim + 1)

        # 2*n_agents 간의 edge_types로 확장
        edge_types_expanded = edge_types * 4  # obs-obs, obs-reward, reward-obs, reward-reward 관계
        self.encoder = MLPEncoder(args, n_in=n_in, n_hid=n_hid, n_out=edge_types_expanded, do_prob=do_prob, factor=factor)
        self.decoder = MLPDecoder(args, n_in_node=obs_dim+1, edge_types=edge_types_expanded, msg_hid=n_hid, msg_out=n_hid, n_hid=n_hid, do_prob=do_prob)
        self.rel_rec, self.rel_send = self.create_rel_matrices(n_agents)  # n_agents로 수정

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

    def forward(self, in_series):
        # in_series: [batch, n_agents, timesteps, obs_dim+1]
        # rollout/학습 모두에서 in_series는 마지막 timestep을 제외한 시계열로 입력해야 함
        batch_size, n_agents, timesteps, features = in_series.shape
        rel_type_logits = self.encoder(in_series, self.rel_rec, self.rel_send)
        '''
        What is the rel_type_logits? 
                  j=0: O→O(x)   j=1: O→O(o)   j=2: O→R(x)   j=3: O→R(o)   j=4: R→O(x)   j=5: R→O(o)   j=6: R→R(x)   j=7: R→R(o)  
        
        i=0: 0→1  logit(0,0)    logit(0,1)    logit(0,2)    logit(0,3)    logit(0,4)    logit(0,5)    logit(0,6)    logit(0,7)

        i=1: 1→0  logit(1,0)    logit(1,1)    logit(1,2)    logit(1,3)    logit(1,4)    logit(1,5)    logit(1,6)    logit(1,7)
        '''
        rel_type = torch.softmax(rel_type_logits, dim=-1)
        pred = self.decoder(in_series, rel_type, self.rel_rec, self.rel_send)
        return pred, rel_type

    def get_causal_structure(self, in_series):
        # in_series: [batch, n_agents, timesteps, obs_dim+1]
        batch_size, n_agents, timesteps, features = in_series.shape
        
        rel_type_logits = self.encoder(in_series, self.rel_rec, self.rel_send)
        rel_type = torch.softmax(rel_type_logits, dim=-1)
        return rel_type

# ---------------------------------------------------------------------------
# 2. Multi-Agent Actor-Critic (with ACD_SCM & GAT)
# ---------------------------------------------------------------------------
class MultiAgentActorCritic(nn.Module):
    def __init__(self, args, obs_dim, action_dim, num_agents, hidden_dim, edge_types=2, episode_length=10, gat_type='none', gat_dim=32, num_heads=4, dropout=0.1):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.num_agents = num_agents
        self.gat_type = gat_type
        self.gat_dim = gat_dim
        self.n_hid = hidden_dim
        self.edge_types = edge_types * 4  # obs-obs, obs-reward, reward-obs, reward-reward 관계
        self.episode_length = episode_length
        # SCM (ACD 기반)
        self.scm = ACD_SCM(args, num_agents, obs_dim, hidden_dim, edge_types, episode_length)
        # Actor/Critic 네트워크 (기존 구조 유지)
        self.actor_networks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(obs_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_dim)
            ) for _ in range(num_agents)
        ])
        self.critic_networks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(obs_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            ) for _ in range(num_agents)
        ])

    def forward(self, in_series):
        # in_series: [batch, n_agents, timesteps, obs_dim+1]
        scm_pred, causal_structure = self.scm(in_series)
        # 가장 마지막 timestep의 obs만 actor/critic에 사용 (예시)
        obs_last = in_series[:, :, -1, :-1]  # [batch, n_agents, obs_dim]
        actor_outputs = torch.stack([
            self.actor_networks[i](obs_last[:, i, :]) for i in range(self.num_agents)
        ], dim=1)  # [batch, n_agents, action_dim]
        critic_outputs = torch.stack([
            self.critic_networks[i](obs_last[:, i, :]) for i in range(self.num_agents)
        ], dim=1)  # [batch, n_agents, 1]
        return {
            'actor_outputs': actor_outputs,
            'critic_outputs': critic_outputs,
            'scm_predictions': scm_pred,
            'causal_structure': causal_structure
        }

    def compute_a2c_loss(self, in_series, action_series, reward_series, adj_matrix, current_data): 
        pass