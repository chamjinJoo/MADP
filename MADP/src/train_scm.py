import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Any
from collections import deque
import gc
import time
from contextlib import nullcontext

from src.env_wrapper import MPEGymWrapper, RWAREWrapper
from src.models_scm import MultiAgentActorCritic
from src.utils import (
    create_progress_bar,
    init_history,
    update_history,
    plot_history,
    plot_episode_returns,
    save_all_results,
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

def unpack_trajectory(traj: list[tuple]) -> tuple[dict[str, list], dict[str, Any]]:
    keys = ['obs', 'acts', 'rews', 'vals', 'dones']
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
class SCMTrainer:
    def __init__(self, env, model: MultiAgentActorCritic, cfg, device: str | None = None):
        self.device = get_device(cfg) if device is None else torch.device(device)
        self.env = env
        self.model = model.to(self.device)
        self.nagents = model.num_agents
        self.cfg = cfg
        self.history = init_history([
            'scm_loss', 'causal_consistency_loss', 'do_loss', 'cf_loss',
            'policy_loss', 'value_loss', 'entropy', 'loss_rl', 'total_loss', 'grad_norm'
        ])
        self.episode_returns: List[float] = []
        self.batch_returns: List[np.ndarray] = []
        self.h = None
        self.max_grad_norm = getattr(cfg, 'clip_grad', 1.0)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=cfg.lr)
        # --- causal structure 변화 저장용 리스트
        self.causal_structure_list = []
        print(f"\n Environment: {self.env.name}")
        print(f" Operation on {self.device}")
        print(f" Model Configuration:")
        print(f"  - Use GAT: {self.model.use_gat}")
        print(f"  - Use Causal GAT: {getattr(self.model, 'use_causal_gat', False)}")
        print(f"  - Hidden Dim: {self.model.hidden_dim}")
        print(f"  - GAT Dim: {self.model.gat_dim}")
        print(f"  - Number of Agents: {self.nagents}")
        self.pbar = create_progress_bar(cfg.total_steps)

    def _rollout(self, render_freq: int = 0) -> List[Tuple]:
        traj: List[Tuple] = []
        episodes = 0
        step_count = 0
        batch_episode_returns = []
        while episodes < self.cfg.ep_num:
            obs = self.env.reset()
            done = False
            while not done:
                self.model.eval()
                if render_freq > 0 and step_count % render_freq == 0:
                    if hasattr(self.env, 'render'):
                        try:
                            self.env.render(mode='human')
                        except Exception as e:
                            print(f"렌더링 중 오류 발생: {e}")
                with torch.no_grad():
                    obs_t = self.preprocess_obs(obs)
                    outputs = self.model(obs_t.unsqueeze(0))
                    logits = outputs['actor_outputs'].squeeze(0)
                    values = outputs['critic_outputs'].squeeze(0)
                    acts = sample_actions(logits)
                acts_onehot = F.one_hot(acts, num_classes=self.model.action_dim).float()
                next_obs, reward, done, truncated, info = self.env.step(tuple(acts.cpu().tolist()))
                traj.append((obs, acts.cpu().numpy(), np.asarray(reward, np.float32), values.cpu().numpy(), done))
                obs = next_obs
                step_count += 1
            rewards = []
            for _,_,r,_,done in traj:
                rewards.append(r)
                if done:
                    break
            rewards = np.array(rewards)
            agent_returns = rewards.sum(axis=0)
            batch_episode_returns.append(agent_returns)
            episodes += 1
        batch_avg_returns = np.mean(batch_episode_returns, axis=0)
        self.batch_returns.append(batch_avg_returns)
        return traj

    def log_metrics(self, metrics: Dict[str, float]):
        update_history(self.history, metrics)
        self.pbar.update()

    def train(self, render_freq: int = 0): # render_freq: 0: 렌더링 안함
        self.model.train()
        for global_step in range(self.cfg.total_steps):
            if global_step % 10 == 0:
                optimize_memory()
            traj = self._rollout(render_freq=render_freq)
            _, data = unpack_trajectory(traj)
            vals = np.array(data['vals'])
            rews = np.array(data['rews'])
            dones = np.array(data['dones'])
            if vals.ndim == 3 and vals.shape[2] == 1:
                vals = vals.squeeze(-1)
            if rews.ndim == 3 and rews.shape[2] == 1:
                rews = rews.squeeze(-1)
            if dones.ndim == 3 and dones.shape[2] == 1:
                dones = dones.squeeze(-1)
            adv_np, gae_ret_np = compute_gae_multi(
                rews, vals, dones,
                self.cfg.gamma, self.cfg.gae_lambda
            )
            act_t = to_torch(data['acts'], self.device, flatten=True, dtype=torch.long)
            adv_t = to_torch(adv_np, self.device, flatten=True, dtype=torch.float)
            gae_ret_t = to_torch(gae_ret_np, self.device, flatten=True, dtype=torch.float)
            obs_raw = data['obs']
            acts_np = data['acts']
            acts = torch.from_numpy(acts_np).long().to(self.device)
            acts_onehot = F.one_hot(acts, num_classes=self.model.action_dim).float()
            # SCM loss, causal consistency loss, RL loss 계산
            obs_torch = self.preprocess_obs(np.stack(obs_raw, axis=0))
            next_obs_torch = self.preprocess_obs(np.stack(obs_raw[1:] + [obs_raw[-1]], axis=0))
            actions_torch = acts_onehot
            scm_loss = self.model.compute_scm_loss(obs_torch, actions_torch, next_obs_torch)
            causal_structure = self.model.scm.get_causal_structure()
            causal_consistency_loss = self.model.compute_causal_consistency_loss(causal_structure)
            # --- causal structure 저장 (detach, cpu, numpy)
            self.causal_structure_list.append(causal_structure.detach().cpu().numpy())

            # do-연산 및 counterfactual loss 추가 (모든 agent)
            # 각 agent별로 무작위 action을 선택하여 intervention/counterfactual로 사용
            num_agents = actions_torch.shape[1]
            action_dim = actions_torch.shape[2]
            do_action_tensor = torch.zeros(num_agents, action_dim, device=actions_torch.device)
            cf_action_tensor = torch.zeros(num_agents, action_dim, device=actions_torch.device)
            for i in range(num_agents):
                do_idx = int(torch.randint(0, action_dim, (1,)))
                cf_idx = int(torch.randint(0, action_dim, (1,)))
                do_action_tensor[i, do_idx] = 1.0
                cf_action_tensor[i, cf_idx] = 1.0
            do_loss = self.model.compute_do_intervention_loss_all(obs_torch, actions_torch, next_obs_torch, do_action_tensor)
            cf_loss = self.model.compute_counterfactual_loss_all(obs_torch, actions_torch, next_obs_torch, cf_action_tensor)
            lambda_do = 1.0  # 필요시 조정
            lambda_cf = 1.0  # 필요시 조정

            # RL loss (actor/critic)
            outputs = self.model(obs_torch)
            logits = outputs['actor_outputs'].view(-1, self.model.action_dim)
            values = outputs['critic_outputs'].view(-1)
            policy_loss = -(F.log_softmax(logits, dim=-1).gather(1, act_t.unsqueeze(1)).squeeze(1) * adv_t).mean()
            value_loss = F.mse_loss(values, gae_ret_t)
            probs = F.softmax(logits, dim=-1)
            entropy = -(probs * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
            loss_rl = policy_loss + self.cfg.value_coef * value_loss - self.cfg.ent_coef * entropy
            total_loss = scm_loss + causal_consistency_loss + loss_rl + lambda_do * do_loss + lambda_cf * cf_loss
            total_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.opt.step()
            self.opt.zero_grad()
            metrics = {
                'scm_loss': scm_loss.item(),
                'causal_consistency_loss': causal_consistency_loss.item(),
                'do_loss': float(do_loss),
                'cf_loss': float(cf_loss),
                'policy_loss': policy_loss.item(),
                'value_loss': value_loss.item(),
                'entropy': entropy.item(),
                'loss_rl': loss_rl.item(),
                'total_loss': total_loss.item(),
                'grad_norm': grad_norm.item() if hasattr(grad_norm, 'item') else float(grad_norm)
            }
            self.log_metrics(metrics)
        self.pbar.close()
        close_env_and_figures(self.env)
        import os
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_dir = f"outputs_scm/{timestamp}"
        os.makedirs(output_dir, exist_ok=True)
        env_name = getattr(self.env, 'name', 'unknown_env')
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
        # ---------------------- 결과 출력 및 시각화 ----------------------
        print("\n===== 학습 결과 요약 =====")
        if len(self.history['total_loss']) > 0:
            print(f"최종 total_loss: {self.history['total_loss'][-1]:.4f}")
        if len(self.history['scm_loss']) > 0:
            print(f"최종 SCM loss: {self.history['scm_loss'][-1]:.4f}")
        if len(self.history['causal_consistency_loss']) > 0:
            print(f"최종 Causal Consistency loss: {self.history['causal_consistency_loss'][-1]:.4f}")
        if len(self.history['loss_rl']) > 0:
            print(f"최종 RL loss: {self.history['loss_rl'][-1]:.4f}")
        if len(self.batch_returns) > 0:
            print(f"최종 평균 리턴: {self.batch_returns[-1]}")
        # --- causal structure 변화 시각화 ---
        try:
            self.plot_causal_structure_evolution(output_dir)
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

    def plot_causal_structure_evolution(self, output_dir):
        """
        학습 과정에서의 causal structure 행렬 변화 시각화 (heatmap 시퀀스)
        """
        import matplotlib.pyplot as plt
        import numpy as np
        arr = np.stack(self.causal_structure_list, axis=0)  # (steps, N, N)
        steps, N, _ = arr.shape
        # (1) 마지막 causal structure heatmap
        plt.figure(figsize=(4,4))
        plt.title(f'Causal Structure (Last, step={steps})')
        plt.imshow(arr[-1], cmap='viridis', vmin=0, vmax=1)
        plt.colorbar()
        plt.xlabel('Cause')
        plt.ylabel('Effect')
        plt.savefig(f"{output_dir}/causal_structure_last.png")
        plt.show()
        # (2) 각 entry별 변화 라인플롯
        plt.figure(figsize=(8,6))
        for i in range(N):
            for j in range(N):
                plt.plot(arr[:,i,j], label=f'{i}->{j}')
        plt.title('Causal Structure Entry Evolution')
        plt.xlabel('Step')
        plt.ylabel('Softmax Weight')
        plt.legend(bbox_to_anchor=(1.05,1), loc='upper left', fontsize='small')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/causal_structure_evolution.png")
        plt.show()

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
    model = MultiAgentActorCritic(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dim=cfg['model']['hidden_dim'],
        num_agents=num_agents,
        use_gat=cfg['model'].get('use_gat', True),
        use_causal_gat=cfg['model'].get('use_causal_gat', True),
        gat_dim=cfg['model'].get('gat_dim', 32),
        num_heads=cfg['model'].get('num_heads', 4),
        dropout=cfg['model'].get('dropout', 0.1)
    )
    trainer = SCMTrainer(env, model, cfg['params'])
    trainer.train() 