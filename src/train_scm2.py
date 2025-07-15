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
from src.models_scm2 import MultiAgentActorCritic
from src.utils import (
    create_progress_bar,
    init_history,
    update_history,
    save_all_results,
    init_wandb,
    log_gradients,
    log_metrics,
    finish_wandb,
    plot_causal_structure_evolution,
    ask_wandb_logging,
)
from src.forward_pass_and_eval import forward_pass_and_eval

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

def unpack_trajectory(traj: list[tuple]) -> tuple[dict[str, list], dict[str, Any]]:
    keys = ['obs', 'acts', 'rews', 'vals', 'dones', 'series_data']
    data = {k: [] for k in keys}
    for t in traj:
        for i, k in enumerate(keys):
            data[k].append(t[i])
    np_data = {
        'obs': data['obs'],
        'acts': np.stack(data['acts'], axis=0),
        'rews': np.array(data['rews'], dtype=np.float32),
        'vals': np.stack(data['vals'], axis=0),
        'dones': np.array(data['dones'], dtype=np.float32),
        'series_data': np.stack(data['series_data'], axis=0),  # (batch, episode_length, 2N*series_dim)
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
    def __init__(self, env, model: MultiAgentActorCritic, cfg, device: str | None = None):
        self.device = get_device(cfg) if device is None else torch.device(device)
        self.env = env
        self.model = model.to(self.device)
        self.nagents = model.num_agents
        self.cfg = cfg
        self.history = init_history([
            'scm_loss', 'causal_consistency_loss',
            'policy_loss', 'value_loss', 'entropy', 'loss_rl', 'total_loss', 'grad_norm'
        ])
        self.episode_returns: List[float] = []
        self.batch_returns: List[np.ndarray] = []
        self.h = None
        self.max_grad_norm = getattr(cfg, 'clip_grad', 1.0)
        self.episode_length = getattr(self.env, 'episode_length', getattr(self.cfg, 'episode_length', 10))
        self.opt = torch.optim.Adam(self.model.parameters(), lr=cfg.lr)
        # --- causal structure 변화 저장용 리스트
        self.causal_structure_list = []
        

        # Wandb 초기화
        env_name = getattr(self.env, 'name', 'unknown_env')
        self.wandb_enabled = ask_wandb_logging()
        if self.wandb_enabled:
            init_wandb(env_name, self.model, self.cfg, str(self.device))
        
        print(f"\n Environment: {self.env.name}")
        print(f" Operation on {self.device}")
        print(f" Model Configuration:")
        print(f"  - Hidden Dim: {self.model.n_hid}")
        print(f"  - Number of Agents: {self.nagents}")
        self.pbar = create_progress_bar(cfg.total_steps)

    def _rollout(self, render_freq: int = 0) -> list[dict]:
        batch_trajs: list[dict] = []
        episodes = 0
        step_count = 0
        batch_episode_returns = []
        while episodes < self.cfg.ep_num:
            obs_series = np.zeros((self.episode_length, self.nagents, self.model.obs_dim))
            reward_series = np.zeros((self.episode_length, self.nagents, 1))
            obs = self.env.reset()
            obs_series[:] = obs
            # 시계열 저장용 리스트
            obs_list = [obs.copy()]
            action_list = []
            reward_list = []  # step마다 reward만 저장
            done_list = []
            done = False
            while not done:
                self.model.eval()
                with torch.no_grad():
                    obs_reward_series = np.concatenate([obs_series, reward_series], axis=-1)
                    obs_reward_series = np.transpose(obs_reward_series, (1,0,2))
                    obs_reward_series = obs_reward_series[None, ...]
                    x_t = torch.as_tensor(obs_reward_series, dtype=torch.float32)
                    outputs = self.model(x_t)
                    logits = outputs['actor_outputs'].squeeze(0)  # [n_agents, action_dim]
                    acts = sample_actions(logits)
                acts_np = acts.cpu().numpy()
                next_obs, reward, done, truncated, info = self.env.step(tuple(acts_np.tolist()))
                obs_series = np.roll(obs_series, -1, axis=0)
                reward_series = np.roll(reward_series, -1, axis=0)
                obs_series[-1] = next_obs
                reward_series[-1] = reward[:, None]
                obs_list.append(next_obs.copy())
                action_list.append(acts_np)
                reward_list.append(np.asarray(reward, np.float32).copy())
                done_list.append(done)
                obs = next_obs
                step_count += 1
            # numpy array로 변환 및 정렬
            obs_arr = np.stack(obs_list, axis=1)         # (n_agents, T+1, obs_dim)
            action_arr = np.stack(action_list, axis=1)   # (n_agents, T)
            reward_arr = np.stack(reward_list, axis=1)   # (n_agents, T)
            done_arr = np.array(done_list, dtype=np.float32)  # (T,)
            # zero-padding: 모든 시퀀스를 max_steps+1/max_steps 길이로 맞춤
            max_steps = self.episode_length
            pad_len_obs = max_steps + 1 - obs_arr.shape[1]
            if pad_len_obs > 0:
                obs_arr = np.pad(obs_arr, ((0,0),(0,pad_len_obs),(0,0)), mode='constant')
            pad_len_rew = max_steps - reward_arr.shape[1]
            if pad_len_rew > 0:
                reward_arr = np.pad(reward_arr, ((0,0),(0,pad_len_rew)), mode='constant')
            pad_len_act = max_steps - action_arr.shape[1]
            if pad_len_act > 0:
                action_arr = np.pad(action_arr, ((0,0),(0,pad_len_act)), mode='constant')
            pad_len_done = max_steps - done_arr.shape[0]
            if pad_len_done > 0:
                done_arr = np.pad(done_arr, (0,pad_len_done), mode='constant')
            batch_trajs.append({
                'obs': obs_arr,
                'acts': action_arr,
                'rews': reward_arr,
                'dones': done_arr,
            })
            # reward_arr shape: (n_agents, max_steps)
            batch_episode_returns.append(reward_arr)  # (n_agents, max_steps)
            episodes += 1
        # batch_episode_returns shape: (batch, n_agents, max_steps)
        batch_episode_returns = np.stack(batch_episode_returns, axis=0)
        batch_avg_returns = np.mean(batch_episode_returns, axis=0)
        self.batch_returns.append(batch_avg_returns)
        return batch_trajs

    def train(self, render_freq: int = 0):
        self.model.train()
        for global_step in range(self.cfg.total_steps):
            if global_step % 10 == 0:
                optimize_memory()
            batch_trajs = self._rollout(render_freq=render_freq)
            total_loss_list = []  # 에피소드별 total_loss 저장
            for ep_traj in batch_trajs:
                acts = ep_traj['acts']
                rews = ep_traj['rews']
                dones = ep_traj['dones']
                obs_arr = ep_traj['obs']  # (T+1, n_agents, obs_dim)
                rews_arr = ep_traj['rews']  # (T+1, n_agents)
                # using actual trajectory w/o zero-padding
                # obs와 rews를 마지막 차원 기준으로 concat: (T+1, n_agents, obs_dim+1)
                obs_reward_arr = np.concatenate([obs_arr[:, 1:, :], rews_arr[..., None]], axis=-1)  
                # (1, n_agents, episode_length, obs_dim+1)
                obs_reward_torch = torch.as_tensor(obs_reward_arr[None, ...], dtype=torch.float32, device=self.device)
                # --- SCM loss 등은 전체 시퀀스(혹은 마지막 시퀀스)로 한 번만 계산 ---
                target_seq = obs_reward_torch[:, :, 1:, :]
                input_seq = obs_reward_torch[:, :, :-1, :]
                batch, n_agents, timesteps, feat = input_seq.shape
                input_seq_flat = input_seq.reshape(batch, n_agents, -1)  # [1, n_agents, timesteps * (obs_dim+1)]
                losses, output, edges = forward_pass_and_eval(
                    args=self.model.scm.encoder.args,
                    encoder=self.model.scm.encoder,
                    decoder=self.model.scm.decoder,
                    data=input_seq_flat,
                    relations=None,
                    rel_rec=self.model.scm.rel_rec,
                    rel_send=self.model.scm.rel_send,
                    hard=False,
                    data_encoder=input_seq_flat,
                    data_decoder=target_seq,
                    log_prior=None
                )
                scm_loss = losses['loss']
                recon_loss = losses['loss_nll']
                kl_loss = losses['loss_kl']
                edge_acc = losses.get('acc', torch.tensor(0.0))
                auroc = losses.get('auroc', torch.tensor(0.0))
                # --- 각 timestep마다 시계열 데이터를 zero-padding하여 모델에 넣고 logits, values 시퀀스 생성 ---
                T = self.episode_length  # max_steps
                logits_seq = []
                values_seq = []
                for t in range(1, T+1):
                    # obs_arr: (n_agents, max_steps+1, obs_dim)
                    # reward_arr: (n_agents, max_steps)
                    # 각 timestep마다 obs, reward를 맞춰서 concat
                    obs_slice = obs_arr[:, :t+1, :]  # (n_agents, t+1, obs_dim)
                    rew_slice = rews_arr[:, :t+1]   # (n_agents, t+1)
                    # obs와 reward를 마지막 차원 기준으로 concat
                    obs_rew_slice = np.concatenate([obs_slice, rew_slice[..., None]], axis=-1)  # (n_agents, t+1, obs_dim+1)
                    pad_len = (T+1) - (t+1)
                    if pad_len > 0:
                        obs_rew_slice = np.pad(obs_rew_slice, ((0,0),(pad_len,0),(0,0)), mode='constant')
                    input_slice = obs_rew_slice[None, ...]  # (1, n_agents, max_steps+1, obs_dim+1)
                    outputs = self.model(torch.as_tensor(input_slice, dtype=torch.float32, device=self.device))
                    logits = outputs['actor_outputs'].squeeze(0)
                    values = outputs['critic_outputs'].squeeze(0)
                    logits_seq.append(logits)
                    values_seq.append(values.squeeze(-1))
                logits_seq = torch.stack(logits_seq, dim=0)
                values_seq = torch.stack(values_seq, dim=0)
                # --- 이후 RL loss 계산에 사용 ---
                logits_flat = logits_seq.reshape(-1, self.model.action_dim)
                values_flat = values_seq.flatten()
                # GAE 계산을 위해 shape 맞추기
                rews_for_gae = rews[1:].T  # (T, n_agents)
                vals_for_gae = values_seq.cpu().detach().numpy()  # (T, n_agents)
                dones_for_gae = dones  # (T,)
                if dones_for_gae.ndim == 1:
                    if dones_for_gae.shape[0] == rews_for_gae.shape[0]:
                        dones_for_gae = np.broadcast_to(dones_for_gae[:, None], rews_for_gae.shape)
                    elif dones_for_gae.shape[0] == rews_for_gae.shape[1]:
                        dones_for_gae = np.broadcast_to(dones_for_gae[None, :], rews_for_gae.shape)
                    else:
                        raise ValueError(f'dones_for_gae shape {dones_for_gae.shape} cannot be broadcast to {rews_for_gae.shape}')
                adv_np, gae_ret_np = compute_gae_multi(
                    rews_for_gae, vals_for_gae, dones_for_gae,
                    self.cfg.gamma, self.cfg.gae_lambda
                )
                act_t = to_torch(acts, self.device, flatten=True, dtype=torch.long)
                adv_t = to_torch(adv_np, self.device, flatten=True, dtype=torch.float)
                gae_ret_t = to_torch(gae_ret_np, self.device, flatten=True, dtype=torch.float)
                policy_loss = -(F.log_softmax(logits_flat, dim=-1).gather(1, act_t.unsqueeze(1)).squeeze(1) * adv_t).mean()
                value_loss = F.mse_loss(values_flat, gae_ret_t)
                probs = F.softmax(logits_flat, dim=-1)
                entropy = -(probs * F.log_softmax(logits_flat, dim=-1)).sum(dim=-1).mean()
                loss_rl = policy_loss + self.cfg.value_coef * value_loss - self.cfg.ent_coef * entropy
                total_loss = scm_loss + loss_rl
                total_loss_list.append(total_loss)
                metrics = {
                    'scm_loss': scm_loss.item(),
                    'recon_loss': recon_loss.item(),
                    'kl_loss': kl_loss.item(),
                    'edge_acc': edge_acc.item() if hasattr(edge_acc, 'item') else float(edge_acc),
                    'auroc': auroc.item() if hasattr(auroc, 'item') else float(auroc),
                    'policy_loss': policy_loss.item(),
                    'value_loss': value_loss.item(),
                    'entropy': entropy.item(),
                    'loss_rl': loss_rl.item(),
                    'total_loss': total_loss.item(),
                    'global_step': global_step
                }
                log_metrics(metrics, self.wandb_enabled)
                update_history(self.history, metrics)
            mean_total_loss = torch.stack(total_loss_list).mean()
            self.opt.zero_grad()
            mean_total_loss.backward()
            if global_step % 10 == 0:
                grad_info = log_gradients(self.model, self.wandb_enabled)
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.opt.step()
            self.pbar.update()
        self.pbar.close()
        close_env_and_figures(self.env)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        env_name = getattr(self.env, 'name', 'unknown_env')
        output_dir = f"outputs_scm/{env_name}/{timestamp}"
        os.makedirs(output_dir, exist_ok=True)
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
        finish_wandb(self.wandb_enabled)

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
    model = MultiAgentActorCritic(
        acd_args,
        obs_dim=obs_dim,
        action_dim=action_dim,
        num_agents=num_agents,
        hidden_dim=cfg['model']['hidden_dim'],
        edge_types=edge_types,
        gat_type=cfg['model'].get('gat_type', 'none'),
        gat_dim=cfg['model'].get('gat_dim', 32),
        num_heads=cfg['model'].get('num_heads', 4),
        dropout=cfg['model'].get('dropout', 0.1),
    )
    trainer = SCMTrainer2(env, model, cfg['params'])
    trainer.train() 