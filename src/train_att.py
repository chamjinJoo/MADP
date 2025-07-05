# train_att.py — Training script for VRNNDecentralizedA2C model

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Any, Callable
from collections import deque
import gc
import time
from contextlib import nullcontext
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler

# Mixed precision 지원
try:
    from torch.amp.autocast_mode import autocast
    MIXED_PRECISION_AVAILABLE = True
except ImportError:
    MIXED_PRECISION_AVAILABLE = False
    print("Warning: Mixed precision not available. Install PyTorch >= 1.6.0")

from src.env_wrapper import MPEGymWrapper, RWAREWrapper
from src.models_att import (
    VRNNDecentralizedA2C,
    compute_rl_loss,
    compute_vae_loss
)
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
    """GPU 디바이스 설정 및 최적화"""
    if cfg['params']['cuda'] and torch.cuda.is_available():
        device = torch.device("cuda")
        # GPU 메모리 최적화 설정
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        # GPU 메모리 캐시 정리
        torch.cuda.empty_cache()
        print(f"GPU 사용: {torch.cuda.get_device_name()}")
        print(f"GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device("cpu")
        print("CPU 사용")
    return device

def optimize_memory():
    """메모리 최적화"""
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
    """GPU로 데이터 전송 최적화"""
    if isinstance(arr, np.ndarray):
        t = torch.from_numpy(arr)
    else:
        t = torch.tensor(arr, dtype=dtype if dtype else torch.float32)
    
    if dtype is not None and t.dtype != dtype:
        t = t.to(dtype)
    if flatten:
        t = t.flatten()
    
    # GPU로 전송 시 pin_memory 사용 (contiguous한 경우에만)
    if device.type == 'cuda' and t.is_contiguous():
        t = t.pin_memory().to(device, non_blocking=True)
    else:
        t = t.to(device)
    
    return t

def get_onehot_actions(acts_np, act_dim, device):
    acts = torch.from_numpy(acts_np).long().to(device)
    return F.one_hot(acts, num_classes=act_dim).to(torch.float32)

def forward_sequence(model, obs_raw, acts_onehot, h_init, device, preprocess_obs_fn):
    logits_l, vals_l, nlls_l, kls_l, mus_l, logvars_l, zs_l, attn_l = [],[],[],[],[],[],[],[]
    h_re = h_init.clone()
    for t, o in enumerate(obs_raw):
        o_t = preprocess_obs_fn(o)
        a_prev = torch.zeros(model.nagents, model.act_dim, device=device) if t==0 else acts_onehot[t-1]
        h_re, nll_s, kl_s, logits_s, _, vals_s, mu_s, logvar_s, zs_s, attn_s, infer_loss = \
            model.forward_step(o_t, a_prev, h_re)
        logits_l.append(logits_s); vals_l.append(vals_s)
        nlls_l.append(nll_s); kls_l.append(kl_s)
        mus_l.append(mu_s); logvars_l.append(logvar_s)
        zs_l.append(zs_s); attn_l.append(attn_s)
    return logits_l, vals_l, nlls_l, kls_l, mus_l, logvars_l, zs_l, attn_l

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
# GPU 최적화된 Trainer 클래스
# ---------------------------------------------------------------------------

class TrainerAtt:
    """GPU 최적화된 Trainer for VRNNDecentralizedA2C - rollout, VAE pretraining, RL update, logging, plotting을 조율"""
    def __init__(
        self,
        env,
        model: VRNNDecentralizedA2C,
        cfg,
        device: str | None = None,
        log_fn: Callable[[Dict[str, float], int], None] | None = None,
        experiment_config: Dict[str, Any] | None = None,
    ) -> None:
        # Device selection with optimization
        if device is None:
            self.device = get_device(cfg)
        else:
            self.device = torch.device(device)

        # Components
        self.env = env
        self.model = model.to(self.device)
        self.nagents = model.nagents
        self.cfg = cfg
        self.experiment_config = experiment_config

        # Mixed precision 설정
        self.use_mixed_precision = getattr(cfg, 'mixed_precision', False) and MIXED_PRECISION_AVAILABLE
        if self.use_mixed_precision:
            self.scaler = GradScaler('cuda')
            print("Mixed precision 활성화")
        else:
            self.scaler = None

        # Separate VAE vs RL parameters
        self.vae_params = list(self.model.vrnn_cells.parameters())
        vae_ids = {id(p) for p in self.vae_params}
        self.rl_params  = [p for p in self.model.parameters() if id(p) not in vae_ids]

        # Optimizers with different learning rates
        self.opt = torch.optim.Adam(self.model.parameters(), lr=cfg.lr)
        self.opt_vae = torch.optim.Adam(self.vae_params, lr=cfg.lr_vae)
        self.opt_rl  = torch.optim.Adam(self.rl_params,  lr=cfg.lr)

        # Logging
        self.log_fn = log_fn or (lambda metrics, step: None)
        self.history = init_history([
            'vae_nll','vae_kl','loss_vae',
            'policy_loss','value_loss','entropy','loss_rl','total_loss',
            'grad_norm_vae','grad_norm_rl','infer_loss'
        ])
        self.episode_returns: List[float] = []
        self.batch_returns: List[np.ndarray] = []
        self.episode_successes: List[bool] = []
        self.batch_success_rates: List[float] = []
        
        # Hidden state
        hdim = self.model.hidden_dim
        self.h = torch.zeros(self.nagents, hdim, device=self.device)
        
        # --- 학습 안정성을 위한 gradient clipping 설정
        self.max_grad_norm = getattr(cfg, 'clip_grad', 1.0)
        
        # --- 모델 정보 출력
        print(f"\n Environment: {self.env.name}")
        print(f" Operation on {self.device}")
        print(f" Mixed Precision: {self.use_mixed_precision}")
        print(f" Model Configuration:")
        print(f"  - Use RNN: {self.model.use_rnn}")
        print(f"  - Use Attention: {self.model.use_attention}")
        print(f"  - Hidden Dim: {self.model.hidden_dim}")
        print(f"  - D Model: {self.model.d_model}")
        print(f"  - Z Dim: {self.model.z_dim}")
        print(f"  - Number of Agents: {self.nagents}")
        
        self.pbar = create_progress_bar(cfg.total_steps)

    def _rollout(self, render_freq: int = 0) -> List[Tuple]:
        """
        GPU 최적화된 rollout with optional real-time rendering
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
            

            
            episode_success = False
            
            while not done:
                self.model.eval()
                
                if not self.env.name == 'dectiger':
                    # Render if enabled and at the right frequency
                    if render_freq > 0 and step_count % render_freq == 0:
                        self.env.render(mode='human')
                
                with torch.no_grad():
                    obs_t = self.preprocess_obs(obs)
                    h_new, nlls, kls, logits, ref_logits, values, mu, logvar, zs, attention_output, infer_loss = \
                        self.model.forward_step(obs_t, self.last_action, self.h)
                    self.h = h_new.detach()
                    acts = sample_actions(logits)

                self.last_action = F.one_hot(acts, num_classes=self.model.act_dim).to(torch.float32)
                next_obs, reward, done, truncated, info = self.env.step(tuple(acts.cpu().tolist()))
                # --- success 여부 기록
                if done and reward > 1:
                    episode_success = True
                
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

    def log_metrics(self, vae_metrics, rl_metrics, total_loss, grad_norms=None, infer_loss=None):
        metrics = {**vae_metrics, **rl_metrics, 'total_loss': total_loss.item()}
        if grad_norms is not None:
            metrics['grad_norm_vae'] = grad_norms['vae']
            metrics['grad_norm_rl'] = grad_norms['rl']
        if infer_loss is not None:
            metrics['infer_loss'] = infer_loss.item()
        update_history(self.history, metrics)
        self.pbar.update()

    def train(self):
        """
        GPU 최적화된 훈련 루프
        """
        self.model.train()
        h_initial = self.h.clone()
        
        for global_step in range(self.cfg.total_steps):
            # 메모리 최적화
            if global_step % 10 == 0:
                optimize_memory()
            
            traj = self._rollout(render_freq=0)
            # trajectory unpack
            _, data = unpack_trajectory(traj)
            
            # GAE 계산 전 shape 보정
            vals = np.array(data['vals'])
            rews = np.array(data['rews'])
            dones = np.array(data['dones'])

            # (T, N, 1) -> (T, N)
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
            
            # Flatten & to tensor (GPU 최적화)
            act_t = to_torch(data['acts'], self.device, flatten=True, dtype=torch.long)
            adv_t = to_torch(adv_np, self.device, flatten=True, dtype=torch.float)
            gae_ret_t = to_torch(gae_ret_np, self.device, flatten=True, dtype=torch.float)
            
            # Re-evaluate sequence under grad (모든 학습용 값은 여기서 모델로 얻음)
            obs_raw = data['obs']
            acts_np = data['acts']
            acts = torch.from_numpy(acts_np).long().to(self.device)
            acts_onehot = F.one_hot(acts, num_classes=self.model.act_dim).to(torch.float32)
            logits_l, vals_l, nlls_l, kls_l, mus_l, logvars_l, zs_l, attn_l, infer_losses = [],[],[],[],[],[],[],[],[]
            
            self.model.train()
            h_re = self.h.clone()
            
            # Mixed precision context
            autocast_context = autocast('cuda') if self.use_mixed_precision else nullcontext()
            
            with autocast_context:
                for t, o in enumerate(obs_raw):
                    o_t = self.preprocess_obs(o)
                    a_prev = torch.zeros(self.nagents, self.model.act_dim, device=self.device) if t==0 else acts_onehot[t-1]
                    h_re, nll_s, kl_s, logits_s, _, vals_s, mu_s, logvar_s, zs_s, attn_s, infer_loss = \
                        self.model.forward_step(o_t, a_prev, h_re)
                    logits_l.append(logits_s); vals_l.append(vals_s)
                    nlls_l.append(nll_s); kls_l.append(kl_s)
                    mus_l.append(mu_s); logvars_l.append(logvar_s)
                    zs_l.append(zs_s); attn_l.append(attn_s)
                    infer_losses.append(infer_loss)

                logits_t = torch.vstack(logits_l)
                val_t    = torch.cat(vals_l, dim=0).squeeze()
                nll_t    = torch.cat([x.unsqueeze(1) for x in nlls_l], dim=0).squeeze()
                kl_t     = torch.cat([x.unsqueeze(1) for x in kls_l], dim=0).squeeze()
                mu_t     = torch.vstack(mus_l)
                logvar_t = torch.vstack(logvars_l)
                
                # Average inference loss
                avg_infer_loss = torch.stack(infer_losses).mean()
                
                # ActorCriticHead를 사용한 표준 RL loss 계산
                loss_rl, rl_metrics = compute_rl_loss(
                    logits_t, act_t, adv_t, val_t, gae_ret_t, 
                    ent_coef=self.cfg.ent_coef, 
                    value_coef=self.cfg.value_coef
                )
                
                # VAE loss
                loss_vae, vae_metrics = compute_vae_loss(
                    nll_t, kl_t,
                    nll_coef=getattr(self.cfg, 'nll_coef', 1.0),
                    kl_coef=getattr(self.cfg, 'kl_coef', 1.0)
                )
                
                # Total loss with inference loss
                infer_coef = getattr(self.cfg, 'infer_coef', 0.1)
                total_loss = loss_vae + loss_rl + infer_coef * avg_infer_loss

            # Mixed precision backward pass
            if self.use_mixed_precision and self.scaler is not None:
                self.scaler.scale(total_loss).backward()
                self.scaler.unscale_(self.opt_vae)
                self.scaler.unscale_(self.opt_rl)
            else:
                total_loss.backward()

            # Gradient clipping 적용 및 norm 계산
            grad_norm_vae = torch.nn.utils.clip_grad_norm_(self.vae_params, self.max_grad_norm)
            grad_norm_rl = torch.nn.utils.clip_grad_norm_(self.rl_params, self.max_grad_norm)
            
            # Mixed precision optimizer step
            if self.use_mixed_precision and self.scaler is not None:
                self.scaler.step(self.opt_vae)
                self.scaler.step(self.opt_rl)
                self.scaler.update()
            else:
                self.opt_vae.step()
                self.opt_rl.step()

            # 그래디언트 초기화
            self.opt_vae.zero_grad()
            self.opt_rl.zero_grad()

            # Logging with additional metrics
            self.log_metrics(
                vae_metrics, rl_metrics, total_loss,
                grad_norms={'vae': grad_norm_vae, 'rl': grad_norm_rl},
                infer_loss=avg_infer_loss
            )
            
        self.pbar.close()
        
        # 환경 렌더링 창 닫기 (다양한 환경 래퍼 지원)
        close_env_and_figures(self.env)
        
        # 결과를 outputs 폴더에 저장
        import os
        from datetime import datetime
        from src.utils import save_all_results
        
        # 현재 시간으로 폴더 생성
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_dir = f"outputs/{timestamp}"
        os.makedirs(output_dir, exist_ok=True)
        
        # 환경 이름 가져오기
        env_name = getattr(self.env, 'name', 'unknown_env')
        
        # 모든 결과 저장 (설정 포함)
        save_all_results(
            history=self.history,
            episode_returns=self.batch_returns,
            save_dir=output_dir,
            task_name=env_name,
            config=self.experiment_config,
            episode_counts=None,  # 페이즈별 데이터가 있는 경우 추가
            success_counts=None,  # 페이즈별 데이터가 있는 경우 추가
            phase_names=None      # 페이즈별 데이터가 있는 경우 추가
        )

    def preprocess_obs(self, obs) -> torch.Tensor:
        """GPU 최적화된 관찰 전처리"""
        if isinstance(obs, torch.Tensor):
            # 이미 올바른 디바이스와 dtype인지 확인
            if obs.device != self.device:
                obs = obs.to(self.device)
            if obs.dtype != torch.float32:
                obs = obs.to(torch.float32)
            return obs
        if isinstance(obs, np.ndarray):
            return torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        if isinstance(obs, (tuple, list)):
            arr = np.stack([np.asarray(o, dtype=np.float32) for o in obs], axis=0)
            return torch.as_tensor(arr, dtype=torch.float32, device=self.device)
        raise TypeError(f"Unsupported obs type: {type(obs)}") 