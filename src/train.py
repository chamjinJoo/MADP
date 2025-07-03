# train.py — Training script using Trainer and utils (with VAE pretraining and full RL unpack)

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Any, Callable
from collections import deque

from src.env_wrapper import MPEGymWrapper, RWAREWrapper
from src.models import (
    VRNNGATA2C,
    compute_rl_loss,
    compute_vae_loss,
    compute_dpo_loss,
    compute_comm_loss,
    adaptive_loss_coefficients,
)
from src.utils import (
    create_progress_bar,
    init_history,
    update_history,
    plot_history,
    plot_episode_returns,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sample_actions(logits: torch.Tensor) -> torch.Tensor:
    return torch.distributions.Categorical(logits=logits).sample()

def compute_gae_multi(
    rews: np.ndarray, vals: np.ndarray,
    dones: np.ndarray, gamma: float=0.99,
    lam: float=0.95
) -> Tuple[np.ndarray, np.ndarray]:
    T, N = vals.shape
    adv = np.zeros((T, N), dtype=np.float32)
    next_val = np.zeros(N, dtype=np.float32)
    next_adv = np.zeros(N, dtype=np.float32)
    for t in reversed(range(T)):
        mask = 1.0 - dones[t]
        delta = rews[t] + (gamma * next_val * mask) - vals[t]
        adv[t] = delta + (gamma * lam * next_adv * mask)
        next_val, next_adv = vals[t], adv[t]
    return adv, adv + vals
def unpack_trajectory(traj: list[tuple]) -> tuple[dict[str, list], dict[str, Any]]:
    """trajectory 리스트에서 각 항목을 dict로 한 번에 추출, numpy 변환은 별도 반환"""
    keys = ['obs', 'acts', 'rews', 'vals', 'dones']
    data = {k: [] for k in keys}
    for t in traj:
        for i, k in enumerate(keys):
            data[k].append(t[i])
    # numpy 변환
    np_data = {
        'obs': data['obs'],
        'acts': np.stack(data['acts'], axis=0),
        'rews': np.array(data['rews'], dtype=np.float32),
        'vals': np.stack(data['vals'], axis=0),
        'dones': np.array(data['dones'], dtype=np.float32),
    }
    return data, np_data

def to_torch(arr, device, flatten=False, dtype=None):
    t = torch.from_numpy(arr)
    if dtype is not None:
        t = t.type(dtype)
    if flatten:
        t = t.flatten()
    return t.to(device)

def get_onehot_actions(acts_np, act_dim, device):
    acts = torch.from_numpy(acts_np).long().to(device)
    return F.one_hot(acts, num_classes=act_dim).float()

def forward_sequence(model, obs_raw, acts_onehot, h_init, device, preprocess_obs_fn):
    logits_l, vals_l, nlls_l, kls_l, mus_l, logvars_l = [],[],[],[],[],[]
    h_re = h_init.clone()
    for t, o in enumerate(obs_raw):
        o_t = preprocess_obs_fn(o)
        a_prev = torch.zeros(model.nagents, model.act_dim, device=device) if t==0 else acts_onehot[t-1]
        h_re, nll_s, kl_s, logits_s, _, vals_s, mu_s, logvar_s = \
            model.forward_step(o_t, a_prev, h_re)
        logits_l.append(logits_s); vals_l.append(vals_s)
        nlls_l.append(nll_s); kls_l.append(kl_s)
        mus_l.append(mu_s); logvars_l.append(logvar_s)
    return logits_l, vals_l, nlls_l, kls_l, mus_l, logvars_l

def close_env_and_figures(env):
    # 환경 닫기
    try:
        if hasattr(env, 'close'):
            env.close()
        elif hasattr(env, 'env') and hasattr(env.env, 'close'):
            env.env.close()
        elif hasattr(env, 'unwrapped') and hasattr(env.unwrapped, 'close'):
            env.unwrapped.close()
        # matplotlib 창들도 닫기
        plt.close('all')
        # 잠시 대기하여 렌더링 창이 완전히 닫히도록 함
        import time
        time.sleep(0.5)
    except Exception as e:
        print(f"환경 닫기 중 오류 발생 (무시됨): {e}")

def plot_success_rate(success_list, save_path=None, window=20, task_name=None):
    import matplotlib.pyplot as plt
    arr = np.array(success_list, dtype=np.float32)
    ma = np.convolve(arr, np.ones(window)/window, mode='valid')
    plt.figure(figsize=(8,4))
    plt.plot(arr, label='Success (per episode)', alpha=0.3)
    plt.plot(np.arange(window-1, len(arr)), ma, label=f'Moving Avg (window={window})', color='orange')
    plt.ylim(-0.05, 1.05)
    plt.xlabel('Episode')
    plt.ylabel('Success')
    plt.title(f'Success Rate ({task_name})' if task_name else 'Success Rate')
    plt.legend()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_batch_success_rate(batch_success_rates, save_path=None, task_name=None):
    import matplotlib.pyplot as plt
    arr = np.array(batch_success_rates, dtype=np.float32)
    plt.figure(figsize=(8,4))
    plt.plot(arr, marker='o', label='Batch Success Rate')
    plt.ylim(-0.05, 1.05)
    plt.xlabel('Batch')
    plt.ylabel('Success Rate')
    plt.title(f'Batch Success Rate ({task_name})' if task_name else 'Batch Success Rate')
    plt.legend()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def save_training_plots(history, episode_returns, output_dir, env_name, episode_successes=None, batch_success_rates=None):
    # 그래프 저장
    plot_history(history, save_path=f"{output_dir}/training_history.png", task_name=env_name)
    plot_episode_returns(episode_returns, save_path=f"{output_dir}/episode_returns.png", task_name=env_name)
    if episode_successes is not None:
        plot_success_rate(episode_successes, save_path=f"{output_dir}/success_rate.png", task_name=env_name)
    if batch_success_rates is not None:
        plot_batch_success_rate(batch_success_rates, save_path=f"{output_dir}/batch_success_rate.png", task_name=env_name)

# ---------------------------------------------------------------------------
# train.py — Training script using Trainer and utils (with VAE pretraining and full RL unpack)
# ---------------------------------------------------------------------------

class Trainer:
    """Trainer orchestrates rollout, VAE pretraining, RL update, logging, and plotting."""
    def __init__(
        self,
        env,
        model: VRNNGATA2C,
        cfg,
        device: str | None = None,
        log_fn: Callable[[Dict[str, float], int], None] | None = None,
    ) -> None:
        # Device selection
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # Components
        self.env = env
        self.model = model.to(self.device)
        self.nagents = model.nagents
        self.cfg = cfg

        # Separate VAE vs RL parameters
        self.vae_params = list(self.model.vrnn_cells.parameters())
        vae_ids = {id(p) for p in self.vae_params}
        self.rl_params  = [p for p in self.model.parameters() if id(p) not in vae_ids]

        # Optimizers
        self.opt = torch.optim.Adam(self.model.parameters(), lr=cfg.lr)
        self.opt_vae = torch.optim.Adam(self.vae_params, lr=cfg.lr_vae)
        self.opt_rl  = torch.optim.Adam(self.rl_params,  lr=cfg.lr)

        # Logging
        self.log_fn = log_fn or (lambda metrics, step: None)
        self.history = init_history([
            'vae_nll','vae_kl','coop_kl','loss_vae',
            'policy_loss','value_loss','entropy','loss_rl','total_loss',
            'grad_norm_vae','grad_norm_rl'
        ])
        self.pbar = create_progress_bar(cfg.total_steps)
        self.episode_returns: List[float] = []
        self.batch_returns: List[np.ndarray] = []
        self.episode_successes: List[bool] = []
        self.batch_success_rates: List[float] = []
        # Hidden state
        hdim = self.model.hidden_dim
        self.h = torch.zeros(self.nagents, hdim, device=self.device)
        # --- rolling error buffer 추가 (K=10)
        self.K = 10
        self.rolling_errors = [deque(maxlen=self.K) for _ in range(self.nagents)]
        
        # --- 학습 안정성을 위한 gradient clipping 설정
        self.max_grad_norm = getattr(cfg, 'max_grad_norm', 1.0)
        
        # --- loss balancing을 위한 exponential moving average
        self.loss_ema = {'vae': 1.0, 'rl': 1.0, 'comm': 1.0}
        self.ema_alpha = getattr(cfg, 'ema_alpha', 0.99)
        
        # --- Hierarchical reasoning을 위한 추가 설정
        self.use_hierarchical_reasoning = getattr(cfg, 'use_hierarchical_reasoning', True)
        if self.use_hierarchical_reasoning:
            from src.models import HierarchicalReasoningNetwork
            self.reasoning_net = HierarchicalReasoningNetwork(
                obs_dim=getattr(cfg, 'obs_dim', 16),
                hidden_dim=self.model.hidden_dim,
                n_agents=self.nagents
            ).to(self.device)
        
        # --- Causal GAT 사용 확인
        self.use_causal_gat = getattr(cfg, 'use_causal_gat', False)
        if self.use_causal_gat and not hasattr(self.model, 'use_causal_gat'):
            print("Warning: CausalGATLayer가 모델에 설정되지 않았습니다. configs.yaml에서 use_causal_gat: true로 설정하세요.")
        
        # --- 모델 정보 출력
        print(f"Model Configuration:")
        print(f"  - Use GAT: {self.model.use_gat}")
        print(f"  - Use Causal GAT: {getattr(self.model, 'use_causal_gat', False)}")
        print(f"  - Use Hierarchical Reasoning: {self.use_hierarchical_reasoning}")
        print(f"  - Hidden Dim: {self.model.hidden_dim}")
        print(f"  - GAT Dim: {self.model.gat_dim}")
        print(f"  - Z Dim: {self.model.z_dim}")
        print(f"  - Number of Agents: {self.nagents}")

    def _update_rolling_errors(self, nlls: torch.Tensor):
        # nlls: (N,) tensor
        for i in range(self.nagents):
            self.rolling_errors[i].append(float(nlls[i].item()))

    def get_rolling_mean_errors(self):
        # 각 agent별 rolling mean 반환 (N,) torch tensor
        means = []
        for i in range(self.nagents):
            if len(self.rolling_errors[i]) > 0:
                means.append(sum(self.rolling_errors[i]) / len(self.rolling_errors[i]))
            else:
                means.append(0.0)
        return torch.tensor(means, dtype=torch.float32, device=self.device)

    def _rollout(self, render_freq: int = 0) -> List[Tuple]:
        """
        Rollout with optional real-time rendering
        
        Parameters:
        -----------
        render_freq : int
            Render every N steps (0 = no rendering)
        """
        traj: List[Tuple] = []
        episodes = 0
        step_count = 0
        # --- 배치 내 에피소드별 return을 임시 저장
        batch_episode_returns = []
        batch_episode_successes = []
        
        while episodes < self.cfg.ep_num:
            self.h.zero_()
            obs = self.env.reset()
            done = False
            if not hasattr(self, 'last_action'):
                self.last_action = torch.zeros(
                    self.nagents, self.model.act_dim, device=self.device
                )
            
            # --- rollout 시작 시 rolling error 초기화
            self.rolling_errors = [deque(maxlen=self.K) for _ in range(self.nagents)]
            
            # --- Causal GAT를 위한 이전 GAT 출력 초기화
            if hasattr(self.model, 'prev_gat_output'):
                self.model.prev_gat_output = None
            episode_success = False
            while not done:
                self.model.eval()
                
                if not self.env.name == 'dectiger':
                    # Render if enabled and at the right frequency
                    if render_freq > 0 and step_count % render_freq == 0:
                        self.env.render(mode='human')
                
                with torch.no_grad():
                    obs_t = self.preprocess_obs(obs)
                    # rolling mean error 계산
                    rolling_mean_error = self.get_rolling_mean_errors()
                    
                    # Hierarchical reasoning 적용 (선택적)
                    if self.use_hierarchical_reasoning:
                        # JSD 기반 adjacency matrix 생성 (간단한 버전)
                        adj_matrix = torch.ones(self.nagents, self.nagents, device=self.device)
                        # reasoning_features = self.reasoning_net(obs_t, adj_matrix)
                        # reasoning_features를 모델에 전달 (향후 모델 수정 필요)
                    
                    h_new, nlls, kls, logits, ref_logits, values, mu, logvar, zs, V_gat, comm_recons = \
                        self.model.forward_step(obs_t, self.last_action, self.h, rolling_mean_error=rolling_mean_error)
                    self.h = h_new.detach()
                    acts = sample_actions(logits)
                self.last_action = F.one_hot(acts, num_classes=self.model.act_dim).float()
                next_obs, reward, done, truncated, info = self.env.step(tuple(acts.cpu().tolist()))
                # --- nlls를 rolling error에 저장
                self._update_rolling_errors(nlls)
                # --- success 여부 기록
                if done and isinstance(info, dict) and 'success' in info:
                    episode_success = bool(info['success'])
                traj.append((
                    obs,
                    acts.cpu().numpy(),
                    np.asarray(reward, np.float32),
                    values.cpu().numpy(),
                    done,
                ))
                obs = next_obs
                step_count += 1
                
            # 에이전트별 리턴 벡터로 기록
            rewards = []
            for _,_,r,_,done in traj:
                rewards.append(r)
                if done:
                    break
            rewards = np.array(rewards)
            agent_returns = rewards.sum(axis=0)  # (n_agents,)
            # --- 배치 내 에피소드별 return 저장
            batch_episode_returns.append(agent_returns)
            batch_episode_successes.append(episode_success)
            episodes += 1
        
        # --- 배치가 끝나면 평균 계산하여 저장
        batch_avg_returns = np.mean(batch_episode_returns, axis=0)  # (n_agents,)
        self.batch_returns.append(batch_avg_returns)
        self.episode_successes.extend(batch_episode_successes)
        # --- 배치 success rate 계산 및 저장
        batch_success_rate = float(np.mean(batch_episode_successes)) if batch_episode_successes else 0.0
        self.batch_success_rates.append(batch_success_rate)
        return traj

    def log_metrics(self, vae_metrics, rl_metrics, total_loss, comm_loss=None, grad_norms=None):
        metrics = {**vae_metrics, **rl_metrics, 'total_loss': total_loss.item()}
        if comm_loss is not None:
            metrics['comm_loss'] = comm_loss.item()
        if grad_norms is not None:
            metrics['grad_norm_vae'] = grad_norms['vae']
            metrics['grad_norm_rl'] = grad_norms['rl']
        update_history(self.history, metrics)
        self.pbar.update()

    def train(self):
        """
        Train the model with optional real-time rendering
        
        Parameters:
        -----------
        render_freq : int
            Render every N steps during rollout (0 = no rendering)
        """
        self.model.train()
        h_initial = self.h.clone()
        for global_step in range(self.cfg.total_steps):
            traj = self._rollout(render_freq=0)
            # trajectory unpack
            _, data = unpack_trajectory(traj)
            # Advantage & returns
            adv_np, gae_ret_np = compute_gae_multi(
                data['rews'], data['vals'], data['dones'],
                self.cfg.gamma, self.cfg.gae_lambda
            )
            # Flatten & to tensor
            act_t = to_torch(data['acts'], self.device, flatten=True, dtype=torch.long)
            adv_t = to_torch(adv_np, self.device, flatten=True, dtype=torch.float)
            gae_ret_t = to_torch(gae_ret_np, self.device, flatten=True, dtype=torch.float)
            # Re-evaluate sequence under grad (모든 학습용 값은 여기서 모델로 얻음)
            obs_raw = data['obs']
            acts_np = data['acts']
            acts = torch.from_numpy(acts_np).long().to(self.device)
            acts_onehot = F.one_hot(acts, num_classes=self.model.act_dim).float()
            logits_l, vals_l, nlls_l, kls_l, mus_l, logvars_l, zs_l, Vgat_l, comm_recons_l = [],[],[],[],[],[],[],[],[]
            
            self.model.train()
            h_re = self.h.clone()
            for t, o in enumerate(obs_raw):
                o_t = self.preprocess_obs(o)
                a_prev = torch.zeros(self.nagents, self.model.act_dim, device=self.device) if t==0 else acts_onehot[t-1]
                # --- rolling mean error 계산
                rolling_mean_error = self.get_rolling_mean_errors()
                h_re, nll_s, kl_s, logits_s, _, vals_s, mu_s, logvar_s, zs_s, V_gat_s, comm_recons_s = \
                    self.model.forward_step(o_t, a_prev, h_re, rolling_mean_error=rolling_mean_error)
                # --- nlls를 rolling error에 저장
                self._update_rolling_errors(nll_s)
                logits_l.append(logits_s); vals_l.append(vals_s)
                nlls_l.append(nll_s); kls_l.append(kl_s)
                mus_l.append(mu_s); logvars_l.append(logvar_s)
                zs_l.append(zs_s); Vgat_l.append(V_gat_s); comm_recons_l.append(comm_recons_s)

            logits_t = torch.vstack(logits_l)
            val_t    = torch.cat(vals_l, dim=0).squeeze()
            nll_t    = torch.cat([x.unsqueeze(1) for x in nlls_l], dim=0).squeeze()
            kl_t     = torch.cat([x.unsqueeze(1) for x in kls_l], dim=0).squeeze()
            mu_t     = torch.vstack(mus_l)
            logvar_t = torch.vstack(logvars_l)
            
            # ActorCriticHead를 사용한 표준 RL loss 계산
            loss_rl, rl_metrics = compute_rl_loss(
                logits_t, act_t, adv_t, val_t, gae_ret_t, 
                ent_coef=self.cfg.ent_coef, 
                value_coef=self.cfg.value_coef
            )
            
            # VAE loss
            mu_t = mu_t.view(-1, mu_t.shape[-1])
            logvar_t = logvar_t.view(-1, logvar_t.shape[-1])
            loss_vae, vae_metrics = compute_vae_loss(
                nll_t, kl_t, mu_t, logvar_t,
                nll_coef=self.cfg.nll_coef,
                kl_coef=self.cfg.kl_coef,
                coop_coef=self.cfg.coop_coef,
                n_agents=self.nagents
            )
            # Communication loss 계산 (현재는 0)
            zs_arr = torch.stack(zs_l)  # (T, N, z_dim)
            comm_recons_arr = torch.stack(comm_recons_l)  # (T, N, N, z_dim)
            comm_loss = compute_comm_loss(zs_arr, comm_recons_arr, self.nagents)
           
            # total_loss = vae_coef * loss_vae + rl_coef * loss_rl + comm_coef * comm_loss
            total_loss =  loss_vae + loss_rl
            # 1) 그래디언트 초기화
            self.opt_vae.zero_grad()
            self.opt_rl.zero_grad()
            # 2) 역전파
            total_loss.backward()
            # 3) Gradient clipping 적용 및 norm 계산
            grad_norm_vae = torch.nn.utils.clip_grad_norm_(self.vae_params, self.max_grad_norm)
            grad_norm_rl = torch.nn.utils.clip_grad_norm_(self.rl_params, self.max_grad_norm)
            # 4) 파라미터 업데이트
            self.opt_vae.step()
            self.opt_rl.step()

            # self.opt.zero_grad()
            # total_loss.backward()
            # self.opt.step()
            # Logging with additional metrics
            self.log_metrics(
                vae_metrics, rl_metrics, total_loss, comm_loss,
                grad_norms={'vae': grad_norm_vae, 'rl': grad_norm_rl}
            )
        self.pbar.close()
        
        # 환경 렌더링 창 닫기 (다양한 환경 래퍼 지원)
        close_env_and_figures(self.env)
        # 그래프를 outputs 폴더에 저장
        import os
        from datetime import datetime
        # 현재 시간으로 폴더 생성
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_dir = f"outputs/{timestamp}"
        os.makedirs(output_dir, exist_ok=True)
        # 그래프 저장 (환경 이름 포함)
        env_name = getattr(self.env, 'name', 'unknown_env')
        save_training_plots(
            self.history, self.batch_returns, output_dir, env_name,
            episode_successes=self.episode_successes,
            batch_success_rates=self.batch_success_rates
        )
        print(f"\n모든 그래프가 {output_dir} 폴더에 저장되었습니다.")

    def preprocess_obs(self, obs) -> torch.Tensor:
        if isinstance(obs, np.ndarray):
            return torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        if isinstance(obs, (tuple, list)):
            arr = np.stack([np.asarray(o, dtype=np.float32) for o in obs], axis=0)
            return torch.as_tensor(arr, dtype=torch.float32, device=self.device)
        raise TypeError(f"Unsupported obs type: {type(obs)}")