import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Any
from collections import deque
import gc
import time
from contextlib import nullcontext
import os
from datetime import datetime

from src.env_wrapper import MPEGymWrapper, RWAREWrapper
from src.models_scm2 import ACD_VAE_A2C
from src.utils import (
    create_progress_bar,
    init_history,
    update_history,
    save_all_results,
    plot_causal_structure_evolution,
    init_wandb,
    log_gradients,
    log_metrics,
    finish_wandb,
    ask_wandb_logging,
)

# ---------------------------------------------------------------------------
# GPU 최적화 헬퍼 함수들
# ---------------------------------------------------------------------------
def get_device(cfg) -> torch.device:
    if cfg['params']['cuda'] and torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.cuda.empty_cache()
        print(f"GPU 사용: {torch.cuda.get_device_name()}")
        print(f"GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device("cpu")
        print("CPU 사용")
    return device

def optimize_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

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

def unpack_trajectory(traj: list[dict]) -> tuple[dict[str, list], dict[str, Any]]:
    keys = ['obs', 'acts', 'rews', 'next_obs', 'vals', 'dones']
    data = {k: [] for k in keys}
    for t in traj:
        for k in keys:
            data[k].append(t[k])
    np_data = {
        'obs': data['obs'],
        'acts': np.stack(data['acts'], axis=0),
        'rews': np.array(data['rews'], dtype=np.float32),
        'next_obs': data['next_obs'],
        'vals': np.stack(data['vals'], axis=0),
        'dones': np.array(data['dones'], dtype=np.float32),
    }
    return data, np_data

def to_torch(arr, device, flatten=False, dtype=None):
    if isinstance(arr, np.ndarray):
        t = torch.from_numpy(arr)
    else:
        t = torch.tensor(arr)
    if dtype is not None:
        t = t.type(dtype)
    if flatten:
        t = t.flatten()
    if device.type == 'cuda' and t.is_contiguous():
        t = t.pin_memory().to(device, non_blocking=True)
    else:
        t = t.to(device)
    return t

def get_onehot_actions(acts_np, act_dim, device):
    acts = torch.from_numpy(acts_np).long().to(device)
    return F.one_hot(acts, num_classes=act_dim).float()

def close_env_and_figures(env):
    try:
        if hasattr(env, 'close'):
            env.close()
        elif hasattr(env, 'env') and hasattr(env.env, 'close'):
            env.env.close()
        elif hasattr(env, 'unwrapped') and hasattr(env.unwrapped, 'close'):
            env.unwrapped.close()
        plt.close('all')
        time.sleep(0.5)
    except Exception as e:
        print(f"환경 닫기 중 오류 발생 (무시됨): {e}")

# ---------------------------------------------------------------------------
# SCM/GAT 기반 Multi-Agent Trainer
# ---------------------------------------------------------------------------
class SCMTrainer2:
    def __init__(self, env, model: ACD_VAE_A2C, cfg, device: str | None = None):
        self.device = get_device(cfg) if device is None else torch.device(device)
        self.env = env
        self.model = model.to(self.device)
        self.nagents = model.num_agents
        self.cfg = cfg
        self.ep_num = cfg.ep_num
        self.history = init_history([
            'scm_loss', 'policy_loss', 'value_loss', 'entropy', 'total_loss', 'grad_norm'
        ])
        self.episode_returns: List[float] = []
        self.batch_returns: List[np.ndarray] = []
        self.h = None
        self.max_grad_norm = getattr(cfg, 'clip_grad', 1.0)
        self.loss_history = {'scm': [], 'policy': [], 'value': [], 'entropy': [], 'total': []}
        self.causal_structure_list = []
        
        # Separate optimizers for SCM and RL components
        scm_params = list(self.model.scm.parameters())
        scm_param_ids = {id(p) for p in scm_params}
        self.rl_params = [p for p in self.model.parameters() if id(p) not in scm_param_ids]
        self.scm_opt = torch.optim.Adam(scm_params, lr=cfg.lr)
        self.rl_opt = torch.optim.Adam(self.rl_params, lr=cfg.lr)
        
        env_name = getattr(self.env, 'name', 'unknown_env')
        print(f"\n Environment: {self.env.name}")
        print(f" Operation on {self.device}")
        print(f" Model Configuration:")
        print(f"  - Hidden Dim: {self.model.n_hid}")
        print(f"  - Number of Agents: {self.nagents}")
        self.pbar = create_progress_bar(cfg.total_steps)

    def _rollout(self, render_freq: int = 0) -> tuple[list[dict], np.ndarray]:
        traj: list[dict] = []
        episode_return = np.zeros((self.nagents,), dtype=np.float32)
        obs = self.env.reset()
        done = False
        t = 0
        batch_size = 1
        obs = np.asarray(obs, dtype=np.float32).reshape(batch_size, self.nagents, -1)
        reward = np.zeros((batch_size, self.nagents, 1), dtype=np.float32)
        # RNN hidden state 초기화 (관측/보상 분리)
        prev_hidden_obs = torch.zeros(1, batch_size * self.nagents, self.model.hidden_size, device=self.device)
        prev_hidden_rew = torch.zeros(1, batch_size * self.nagents, self.model.hidden_size, device=self.device) 
        while not done:
            self.model.eval()
            if render_freq > 0 and t % render_freq == 0:
                if hasattr(self.env, 'render'):
                    try:
                        self.env.render(mode='human')
                    except Exception as e:
                        print(f"렌더링 중 오류 발생: {e}")
            obs_torch = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
            reward_torch = torch.as_tensor(reward, dtype=torch.float32, device=self.device).reshape(1, self.nagents, -1)
            with torch.no_grad():
                outputs = self.model(obs_torch, reward_torch, prev_hidden_obs, prev_hidden_rew)
                logits = outputs['actor_outputs'].squeeze(0)
                values = outputs['critic_outputs'].squeeze(0)
                acts = sample_actions(logits)
                next_hidden_obs = outputs['hidden_obs']
                next_hidden_rew = outputs['hidden_rew']
            acts_np = acts.cpu().numpy()
            next_obs, reward, done, truncated, info = self.env.step(tuple(acts_np.tolist()))
            next_obs = np.asarray(next_obs, dtype=np.float32)
            traj.append({
                'obs': obs.copy(),
                'acts': acts_np.copy(),
                'rews': np.asarray(reward, np.float32),
                'vals': values.cpu().numpy(),
                'next_obs': next_obs.copy(),
                'dones': done,
                'hidden_obs': prev_hidden_obs.detach().cpu().numpy().copy(),
                'hidden_rew': prev_hidden_rew.detach().cpu().numpy().copy(),
            })
            obs = next_obs.reshape(1, self.nagents, -1)
            reward_np = np.asarray(reward, dtype=np.float32)
            prev_hidden_obs = next_hidden_obs.detach()
            prev_hidden_rew = next_hidden_rew.detach()
            episode_return += reward_np
            t += 1
        return traj, episode_return

    def train(self, render_freq: int = 0):
        self.model.train()
        for global_step in range(self.cfg.total_steps):
            if global_step % 10 == 0:
                optimize_memory()
            total_loss_accum = torch.tensor(0., device=self.device)
            batch_episode_returns = []
            for ep in range(self.ep_num):
                self.scm_opt.zero_grad()
                self.rl_opt.zero_grad()
                traj, episode_return = self._rollout(render_freq=render_freq)
                _, data = unpack_trajectory(traj)
                obs_arr = np.array(data['obs'])
                action_arr = np.array(data['acts'])
                reward_arr = np.array(data['rews'])
                value_arr = np.array(data['vals'])
                done_arr = np.array(data['dones'])
                T = obs_arr.shape[0]
                nagents = self.nagents
                obs_torch = torch.as_tensor(obs_arr, dtype=torch.float32, device=self.device)
                reward_torch = torch.as_tensor(reward_arr, dtype=torch.float32, device=self.device)
                action_torch = torch.as_tensor(action_arr, dtype=torch.long, device=self.device)
                value_arr = np.array(data['vals'])
                if value_arr.ndim == 3 and value_arr.shape[2] == 1:
                    value_arr = value_arr.squeeze(-1)
                reward_arr = np.array(data['rews'])
                if reward_arr.ndim == 3 and reward_arr.shape[2] == 1:
                    reward_arr = reward_arr.squeeze(-1)
                done_arr = np.array(data['dones'])
                if done_arr.ndim == 3 and done_arr.shape[2] == 1:
                    done_arr = done_arr.squeeze(-1)
                value_torch = torch.as_tensor(value_arr, dtype=torch.float32, device=self.device)
                done_torch = torch.as_tensor(done_arr, dtype=torch.float32, device=self.device)
                prev_hidden_obs = torch.zeros(1, nagents, self.model.hidden_size, device=self.device)
                prev_hidden_rew = torch.zeros(1, nagents, self.model.hidden_size, device=self.device)
                scm_preds, rel_types, actor_outputs, critic_outputs = [], [], [], []
                for t in range(T):
                    obs_t = obs_torch[t]
                    if t == 0:
                        reward_t = torch.zeros(1, nagents, 1, device=self.device)
                    else:
                        reward_t = reward_torch[t].reshape(1, self.nagents, 1)
                    outputs = self.model(obs_t, reward_t, prev_hidden_obs, prev_hidden_rew)
                    scm_preds.append(outputs['scm_predictions'])
                    rel_types.append(outputs['causal_structure'])
                    actor_outputs.append(outputs['actor_outputs'])
                    critic_outputs.append(outputs['critic_outputs'])
                    prev_hidden_obs = outputs['hidden_obs'].detach()
                    prev_hidden_rew = outputs['hidden_rew'].detach()
                scm_preds = torch.stack(scm_preds, dim=0)
                rel_types = torch.stack(rel_types, dim=0)
                actor_outputs = torch.stack(actor_outputs, dim=0)
                critic_outputs = torch.stack(critic_outputs, dim=0)
        
                # --- 4. SCM(ACD) Loss trajectory 평균 (분리된 계산) ---
                next_obs_arr = np.array(data['next_obs'])      # (T, n_agents, obs_dim)
                reward_arr = np.array(data['rews'])            # (T, n_agents) or (T, n_agents, 1)
                if reward_arr.ndim == 2:
                    reward_arr = reward_arr[..., None]         # (T, n_agents, 1)
                target = np.concatenate([next_obs_arr, reward_arr], axis=-1)  # (T, n_agents, obs_dim+1)
                target_torch = torch.as_tensor(target, dtype=torch.float32, device=self.device).unsqueeze(1)  # [T, 1, n_agents, obs_dim+1]
                
                # SCM loss 계산 (SCM 파라미터에만 영향)
                acd_losses = self.model.compute_ACD_loss(rel_types, scm_preds, target_torch, nagents)
                scm_loss = acd_losses['loss']

                # --- 5. RL Loss trajectory 평균 (SCM과 분리된 계산) ---
                adv, returns = compute_gae_multi(
                    rews=reward_arr.squeeze(-1),
                    vals=value_arr,
                    dones=done_arr,
                    gamma=0.99,
                    lam=0.95,
                )
                adv_torch = torch.as_tensor(adv, dtype=torch.float32, device=self.device)
                returns_torch = torch.as_tensor(returns, dtype=torch.float32, device=self.device)
                acts = action_torch
                logits = actor_outputs.squeeze(1)  # [T, n_agents, action_dim]
                values = critic_outputs.squeeze(1).squeeze(-1)  # [T, n_agents]
                logp = torch.log_softmax(logits, dim=-1)
                logp_act = logp.gather(-1, acts.unsqueeze(-1)).squeeze(-1)  # [T, n_agents]
                policy_loss = - (logp_act * adv_torch).mean()
                value_loss = F.mse_loss(values, returns_torch)
                entropy = (- (logp * torch.exp(logp)).sum(-1).mean())
                
                # --- 6. 분리된 loss 계산 및 역전파 ---
                # SCM loss와 RL loss를 완전히 분리하여 계산
                # SCM loss는 SCM 파라미터에만 영향을 주고, RL loss는 RL 파라미터에만 영향을 줌
                
                # SCM loss 계산 및 역전파
                scm_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.scm.parameters(), self.max_grad_norm)
                
                # RL loss 계산 및 역전파
                rl_loss = policy_loss + value_loss - 0.01 * entropy
                rl_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.rl_params, self.max_grad_norm)
                
                # 전체 loss는 로깅용으로만 사용
                total_loss = scm_loss + rl_loss
                total_loss_accum = total_loss_accum + total_loss
                batch_episode_returns.append(episode_return)
                self.causal_structure_list.append(rel_types[-1].detach().cpu().numpy().squeeze(0))
            total_loss_accum /= self.ep_num
            self.scm_opt.step()
            self.rl_opt.step()
            batch_avg_returns = np.mean(batch_episode_returns, axis=0)
            self.batch_returns.append(batch_avg_returns)
            self.loss_history['scm'].append(float(scm_loss.item()))
            self.loss_history['policy'].append(float(policy_loss.item()))
            self.loss_history['value'].append(float(value_loss.item()))
            self.loss_history['entropy'].append(float(entropy.item()))
            self.loss_history['total'].append(float(total_loss.item()))
            metrics = {
                'scm_loss': float(np.mean(self.loss_history['scm'][-self.ep_num:])),
                'policy_loss': float(np.mean(self.loss_history['policy'][-self.ep_num:])),
                'value_loss': float(np.mean(self.loss_history['value'][-self.ep_num:])),
                'entropy': float(np.mean(self.loss_history['entropy'][-self.ep_num:])),
                'total_loss': float(np.mean(self.loss_history['total'][-self.ep_num:])),
                'grad_norm': float(np.mean([p.grad.norm().item() for p in self.model.parameters() if p.grad is not None])),
                'global_step': global_step
            }
            update_history(self.history, metrics)
            self.pbar.update()
        self.pbar.close()
        close_env_and_figures(self.env)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        env_name = getattr(self.env, 'name', 'unknown_env')
        output_dir = f"outputs_scm2/{env_name}/{timestamp}"
        os.makedirs(output_dir, exist_ok=True)
        if self.history is not None:
            save_all_results(
                history=self.history,
                episode_returns=self.batch_returns,
                save_dir=output_dir,
                task_name=env_name,
                config=None,
                episode_counts=None,
                success_counts=None,
                phase_names=None
            )
        try:
            plot_causal_structure_evolution(self.causal_structure_list, output_dir)
        except Exception as e:
            print(f"Causal structure 시각화 오류: {e}")

    def preprocess_obs(self, obs) -> torch.Tensor:
        if isinstance(obs, torch.Tensor):
            return obs.to(self.device)
        if isinstance(obs, np.ndarray):
            return torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        if isinstance(obs, (tuple, list)):
            arr = np.stack([np.asarray(o, dtype=np.float32) for o in obs], axis=0)
            return torch.as_tensor(arr, dtype=torch.float32, device=self.device)
        raise TypeError(f"Unsupported obs type: {type(obs)}")

# ---------------------------------------------------------------------------
# 메인 실행 예시 (실제 실험에서는 configs.yaml 등에서 파라미터 로드 필요)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import yaml
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs.yaml')
    args = parser.parse_args()
    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    # 환경 생성 (예시)
    env = MPEGymWrapper(cfg['env'])
    # 모델 생성
    obs_dim = env.obs_dim
    action_dim = env.act_dim
    num_agents = getattr(env, 'n_agents', 2)
    # ACD 관련 인자 추출
    acd_args = cfg.get('acd_args', {})
    edge_types = cfg['model'].get('edge_types', 2)
    scm_hidden_dim = cfg['model']['scm_hidden_dim']
    rnn_hidden_dim = cfg['model']['rnn_hidden_dim']
    model = ACD_VAE_A2C(
        acd_args,
        obs_dim=obs_dim,
        action_dim=action_dim,
        num_agents=num_agents,
        scm_hidden_dim=scm_hidden_dim,
        rnn_hidden_dim=rnn_hidden_dim,
        edge_types=edge_types,
        gat_type=cfg['model'].get('gat_type', 'none'),
        gat_dim=cfg['model'].get('gat_dim', 32),
        num_heads=cfg['model'].get('num_heads', 4),
        dropout=cfg['model'].get('dropout', 0.1),
    )
    trainer = SCMTrainer2(env, model, cfg['params'])
    trainer.train() 