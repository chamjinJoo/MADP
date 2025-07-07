from __future__ import annotations
from typing import Optional, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# 1. Structural Causal Model (SCM)
# ---------------------------------------------------------------------------
class SCM(nn.Module):
    """
    Multi-agent 환경을 위한 Structural Causal Model.
    각 agent의 관측/행동 간 인과관계를 모델링.
    """
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 64, 
                 num_agents: int = 2, use_causal_prior: bool = True):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.num_agents = num_agents
        self.use_causal_prior = use_causal_prior
        # 인과구조 행렬 (학습/고정)
        if use_causal_prior:
            self.causal_matrix = nn.Parameter(torch.eye(num_agents), requires_grad=True)
        else:
            self.causal_matrix = nn.Parameter(torch.randn(num_agents, num_agents), requires_grad=True)
        # 각 agent별 인과 메커니즘
        self.causal_mechanisms = nn.ModuleList([
            nn.Sequential(
                nn.Linear(obs_dim + action_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, obs_dim)
            ) for _ in range(num_agents)
        ])
        # 각 agent별 노이즈 모델
        self.noise_models = nn.ModuleList([
            nn.Sequential(
                nn.Linear(obs_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, obs_dim)
            ) for _ in range(num_agents)
        ])

    def forward(self, observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            observations: (batch, agents, obs_dim)
            actions: (batch, agents) or (batch, agents, action_dim)
        Returns:
            predicted_observations: (batch, agents, obs_dim)
        """
        batch_size = observations.shape[0]
        # actions shape 보정
        if actions.dim() == 2:
            actions = F.one_hot(actions, num_classes=self.action_dim).float()
        # 인과구조 softmax
        causal_weights = F.softmax(self.causal_matrix, dim=-1)
        # obs+action 결합
        combined_input = torch.cat([observations, actions], dim=-1)
        # 인과 메커니즘 적용
        causal_effects = torch.stack([
            self.causal_mechanisms[i](combined_input[:, i]) 
            for i in range(self.num_agents)
        ], dim=1)
        # 인과구조 행렬 적용
        structured_effects = torch.bmm(
            causal_weights.unsqueeze(0).expand(batch_size, -1, -1),
            causal_effects
        )
        # 노이즈 추가
        noise = torch.stack([
            self.noise_models[i](observations[:, i])
            for i in range(self.num_agents)
        ], dim=1)
        predicted_observations = structured_effects + noise
        return predicted_observations

    def get_causal_structure(self) -> torch.Tensor:
        """학습된 인과구조 행렬 반환"""
        return F.softmax(self.causal_matrix, dim=-1)

# ---------------------------------------------------------------------------
# 2. Graph Attention Layer (GAT)
# ---------------------------------------------------------------------------
class GraphAttentionLayer(nn.Module):
    """
    GAT 논문 기반 그래프 어텐션 레이어
    """
    def __init__(self, input_dim: int, output_dim: int, num_heads: int = 4, 
                 dropout: float = 0.1, alpha: float = 0.2, concat: bool = True):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat
        # 각 head별 선형변환
        self.W = nn.Parameter(torch.Tensor(num_heads, input_dim, output_dim))
        nn.init.xavier_uniform_(self.W)
        # 어텐션 메커니즘
        self.attention = nn.Parameter(torch.Tensor(num_heads, 2 * output_dim, 1))
        nn.init.xavier_uniform_(self.attention)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        batch_size, num_nodes, _ = x.shape
        if adj.dim() == 2:
            adj = adj.unsqueeze(0).expand(batch_size, -1, -1)
        if adj.shape[1] != num_nodes or adj.shape[2] != num_nodes:
            raise ValueError(f"adj shape {adj.shape} does not match num_nodes {num_nodes} (x.shape={x.shape})")
        # 선형변환
        x_transformed = torch.stack([
            torch.matmul(x, self.W[i]) for i in range(self.num_heads)
        ], dim=1)  # (batch, heads, nodes, output_dim)
        # 어텐션 입력 준비
        x_i = x_transformed.unsqueeze(3).expand(-1, -1, -1, num_nodes, -1)
        x_j = x_transformed.unsqueeze(2).expand(-1, -1, num_nodes, -1, -1)
        alpha_input = torch.cat([x_i, x_j], dim=-1)
        # 어텐션 스코어 계산
        attention_scores = torch.einsum(
            'bhijc,hc->bhij',
            alpha_input,
            self.attention.squeeze(-1)
        )
        attention_scores = F.leaky_relu(attention_scores, self.alpha)
        # 마스킹
        adj_expanded = adj.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
        attention_scores = attention_scores.masked_fill(adj_expanded == 0, float('-inf'))
        # softmax
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout_layer(attention_weights)
        # 어텐션 적용
        output = torch.matmul(attention_weights, x_transformed)
        if self.concat:
            output = output.transpose(1, 2).contiguous().view(
                batch_size, num_nodes, self.num_heads * self.output_dim
            )
        else:
            output = output.mean(dim=1)
        return output

# ---------------------------------------------------------------------------
# 3. Causal Graph Attention Network (CausalGAT)
# ---------------------------------------------------------------------------
class CausalGAT(nn.Module):
    """
    인과구조를 통합한 Causal Graph Attention Network
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, 
                 num_heads: int = 4, num_layers: int = 2, dropout: float = 0.1,
                 use_causal_mask: bool = True):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_causal_mask = use_causal_mask
        # GAT layers
        self.gat_layers = nn.ModuleList()
        self.gat_layers.append(
            GraphAttentionLayer(input_dim, hidden_dim, num_heads, dropout, concat=True)
        )
        for _ in range(num_layers - 2):
            self.gat_layers.append(
                GraphAttentionLayer(hidden_dim * num_heads, hidden_dim, num_heads, dropout, concat=True)
            )
        self.gat_layers.append(
            GraphAttentionLayer(hidden_dim * num_heads, output_dim, num_heads, dropout, concat=False)
        )
        # 인과구조 학습 파라미터
        self.causal_structure = nn.Parameter(torch.eye(input_dim), requires_grad=True)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        if self.use_causal_mask:
            causal_mask = F.softmax(self.causal_structure, dim=-1)
            x = torch.matmul(x, causal_mask)
        for i, layer in enumerate(self.gat_layers):
            x = layer(x, adj)
            if i < len(self.gat_layers) - 1:
                x = F.elu(x)
                x = F.dropout(x, self.dropout, training=self.training)
        return x

    def get_causal_structure(self) -> torch.Tensor:
        return F.softmax(self.causal_structure, dim=-1)

# ---------------------------------------------------------------------------
# 4. Multi-Agent Actor-Critic (with SCM & GAT)
# ---------------------------------------------------------------------------
class MultiAgentActorCritic(nn.Module):
    """
    SCM, GAT, CausalGAT을 통합한 multi-agent actor-critic 네트워크
    """
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 64,
                 num_agents: int = 2, use_gat: bool = True, use_causal_gat: bool = True,
                 gat_dim: int = 32, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.num_agents = num_agents
        self.use_gat = use_gat
        self.use_causal_gat = use_causal_gat
        self.gat_dim = gat_dim
        # SCM
        self.scm = SCM(obs_dim, action_dim, hidden_dim, num_agents)
        # GAT
        if use_gat:
            self.gat = GraphAttentionLayer(obs_dim, gat_dim, num_heads, dropout, concat=False)
        if use_causal_gat:
            self.causal_gat = CausalGAT(obs_dim, gat_dim, gat_dim, num_heads, 2, dropout)
        # Actor
        self.actor_networks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(obs_dim + (gat_dim if use_gat else 0), hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_dim)
            ) for _ in range(num_agents)
        ])
        # Critic
        self.critic_networks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(obs_dim + (gat_dim if use_gat else 0), hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            ) for _ in range(num_agents)
        ])
        # 중앙집중 critic (MADDPG 스타일)
        self.centralized_critic = nn.Sequential(
            nn.Linear(num_agents * (obs_dim + action_dim), hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_agents)
        )

    def forward(self, observations: torch.Tensor, actions: Optional[torch.Tensor] = None,
                adj_matrix: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        batch_size, num_agents, _ = observations.shape
        if adj_matrix is None:
            adj_matrix = torch.ones(batch_size, num_agents, num_agents, device=observations.device)
        # GAT 통신
        communication_features = None
        if self.use_gat:
            gat_features = self.gat(observations, adj_matrix)
            communication_features = gat_features
        if self.use_causal_gat and self.use_gat:
            causal_features = self.causal_gat(observations, adj_matrix)
            if communication_features is not None:
                communication_features = communication_features + causal_features
            else:
                communication_features = causal_features
        # Actor
        actor_outputs = []
        for i in range(self.num_agents):
            if communication_features is not None:
                actor_input = torch.cat([observations[:, i], communication_features[:, i]], dim=-1)
            else:
                actor_input = observations[:, i]
            actor_output = self.actor_networks[i](actor_input)
            actor_outputs.append(actor_output)
        actor_outputs = torch.stack(actor_outputs, dim=1)
        # Critic
        critic_outputs = []
        for i in range(self.num_agents):
            if communication_features is not None:
                critic_input = torch.cat([observations[:, i], communication_features[:, i]], dim=-1)
            else:
                critic_input = observations[:, i]
            critic_output = self.critic_networks[i](critic_input)
            critic_outputs.append(critic_output)
        critic_outputs = torch.stack(critic_outputs, dim=1)
        # 중앙집중 critic
        centralized_critic_output = None
        if actions is not None:
            all_inputs = torch.cat([observations, actions], dim=-1)
            all_inputs_flat = all_inputs.view(batch_size, -1)
            centralized_critic_output = self.centralized_critic(all_inputs_flat)
        # SCM 예측
        scm_predictions = None
        if actions is not None:
            scm_predictions = self.scm(observations, actions)
        return {
            'actor_outputs': actor_outputs,
            'critic_outputs': critic_outputs,
            'centralized_critic_output': centralized_critic_output,
            'communication_features': communication_features,
            'scm_predictions': scm_predictions,
            'causal_structure': self.scm.get_causal_structure()
        }

    def get_action_probs(self, observations: torch.Tensor, 
                        adj_matrix: Optional[torch.Tensor] = None) -> torch.Tensor:
        outputs = self.forward(observations, adj_matrix=adj_matrix)
        action_logits = outputs['actor_outputs']
        return F.softmax(action_logits, dim=-1)

    def get_value(self, observations: torch.Tensor, actions: torch.Tensor,
                  adj_matrix: Optional[torch.Tensor] = None) -> torch.Tensor:
        outputs = self.forward(observations, actions, adj_matrix)
        return outputs['centralized_critic_output']

    # -----------------------------------------------------------------------
    # Loss functions (SCM, Causal Consistency)
    # -----------------------------------------------------------------------
    def compute_scm_loss(self, observations: torch.Tensor, actions: torch.Tensor, 
                        next_observations: torch.Tensor) -> torch.Tensor:
        """
        SCM 예측값과 실제 next_observations 간의 MSE loss 계산
        Args:
            observations: (batch, agents, obs_dim)
            actions: (batch, agents, action_dim)
            next_observations: (batch, agents, obs_dim)
        Returns:
            SCM loss (MSE)
        """
        scm_predictions = self.scm(observations, actions)
        scm_loss = F.mse_loss(scm_predictions, next_observations)
        return scm_loss

    def compute_causal_consistency_loss(self, causal_structure: torch.Tensor) -> torch.Tensor:
        """
        인과구조 행렬의 sparsity 및 identity 유도 loss (L1 + identity)
        Args:
            causal_structure: (num_agents, num_agents)
        Returns:
            Causal consistency loss
        """
        sparsity_loss = torch.norm(causal_structure, p=1)
        identity = torch.eye(causal_structure.shape[0], device=causal_structure.device)
        identity_loss = F.mse_loss(causal_structure, identity)
        return sparsity_loss + identity_loss
