"""
train.py — Training script using Trainer and utils
================================================
* Pure algorithm: uses DecTigerEnv, VRNNGATA2C model
* Delegates logging, progress bar, and plotting to utils.py

Usage:
    from src.train import Trainer, TrainConfig
    from src.models import VRNNGATA2C
    from src.envs import DecTigerEnv

    env = DecTigerEnv(proj_dim=64)
    model = VRNNGATA2C(obs_dim=64, act_dim=3, hidden_dim=128, z_dim=32, n_agents=2, gat_dim=256)
    cfg = TrainConfig(total_steps=50000, batch_length=16, lr=3e-4)
    trainer = Trainer(env, model, cfg)
    trainer.train()
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Tuple, Dict, Any, Callable

from src.models import VRNNGATA2C, compute_rl_loss, compute_vae_loss, compute_rl_ppo_loss
from src.utils import (
    create_progress_bar, init_history,
    update_history, plot_history,
    plot_phase_success,
    plot_episode_returns
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sample_actions(logits: torch.Tensor) -> torch.Tensor:
    """Categorical sampling per agent."""
    return torch.distributions.Categorical(logits=logits).sample()


def compute_gae_multi(
    rews: np.ndarray, vals: np.ndarray,
    dones: np.ndarray, gamma: float=0.99,
    lam: float=0.95
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute GAE for shared scalar reward and per-agent value predictions.
    rews: (T,N), vals: (T,N), dones: (T,)
    returns adv, ret arrays of shape (T,N).
    """
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

# ---------------------------------------------------------------------------
class Trainer:
    """Trainer orchestrates rollout, update, logging, and plotting."""
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

        # Separate VAE vs RL parameters by id
        self.vae_params = list(self.model.vrnn_cells.parameters())  # VRNN encoder/decoder만
        vae_param_ids = {id(p) for p in self.vae_params}
        self.rl_params = [p for p in self.model.parameters() if id(p) not in vae_param_ids]
        
        # Optimizers
        self.opt_vae = torch.optim.Adam(self.vae_params, lr=self.cfg.lr)
        self.opt_rl  = torch.optim.Adam(self.rl_params,  lr=self.cfg.lr)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=cfg.lr)

        self.log_fn = log_fn or (lambda metrics, step: None)
        # History and progress bar
        self.history = init_history([
            'vae_nll','vae_kl','coop_kl','loss_vae',
            'policy_loss','value_loss','entropy','loss_rl',
            'total_loss'
        ])
        self.pbar = create_progress_bar(cfg.total_steps)

        # success tracking per phase
        self.num_phases = 10
        self.phase_names = []
        for i in range(self.num_phases):
            start = int(i     / self.num_phases * 100)
            end   = int((i+1) / self.num_phases * 100)
            self.phase_names.append(f"{start}-{end}%")
        self.episode_returns: List[float] = []
        self.episode_counts = [0] * self.num_phases
        self.success_counts = [0] * self.num_phases
         
        # Hidden state
        hdim = model.vrnn_cells[0].rnn.hidden_size
        self.h = torch.zeros(self.nagents, hdim, device=self.device)
        

    def preprocess_obs(self, obs) -> torch.Tensor:
        """
        Env 관측을 (N, D) torch Tensor로 변환한다.
        • obs가 ((arr0, arr1), info) 같이 중첩 튜플일 때도 지원
        • obs가 1-D numpy 배열일 경우 → N 에이전트로 broadcast
        """
        # 1. (obs, info) 형태라면 info 버리고 obs만 남김
        if isinstance(obs, tuple) and len(obs) == 2 and isinstance(obs[1], dict):
            obs, _ = obs                           # obs ← (arr0, arr1)

        # 2. 에이전트별 배열 튜플/리스트
        if isinstance(obs, (tuple, list)):
            # np.stack 전에 dtype·shape 정규화
            arr = np.stack([np.asarray(o, dtype=np.float32) for o in obs], axis=0)
            tensor = torch.as_tensor(arr, dtype=torch.float32, device=self.device)
            return tensor                          # shape (N, D)

        # 3. 단일 np.ndarray → 모든 에이전트에 동일 관측 broadcast
        if isinstance(obs, np.ndarray):
            arr = obs.astype(np.float32)
            if arr.ndim == 1:                      # (D,) → (1, D)
                arr = arr[None, :]
            tensor = torch.as_tensor(arr, dtype=torch.float32, device=self.device)
            # self.nagents 이 정의돼 있다면 필요 시 확장
            if hasattr(self, "N") and tensor.size(0) == 1 and self.nagents > 1:
                tensor = tensor.expand(self.nagents, -1)  # (N, D)
            return tensor

        raise TypeError(f"Unsupported obs type: {type(obs)}")

    def _rollout(self) -> Tuple[List[Tuple], List[Dict]]:
        """
        Collect a fixed number of episodes defined by cfg.batch_size.
        Returns trajectories and logs across all episodes.
        """
        traj: List[Tuple] = []
        log_l: List[Dict] = []
        episodes = 0
        
        while episodes < self.cfg.batch_size:
            # episode start
            self.h = torch.zeros_like(self.h)  # reset hidden state
            obs = self.env.reset()
            
            done = False
            if not hasattr(self, 'last_action'):
                self.last_action = torch.zeros(self.nagents, self.model.act_dim, device=self.device)
            ep_ret = np.zeros(self.nagents, dtype=np.float32)

            while not done:
                self.env.render()
                # inference
                self.model.eval()
                with torch.no_grad():
                    obs_t = self.preprocess_obs(obs)
                    h_new, nlls, kls, logits, values, _, _ \
                        = self.model.forward_step(obs_t, self.last_action, self.h)
                    self.h = h_new.detach()
                acts = sample_actions(logits)
                
                self.last_action = F.one_hot(acts, num_classes=self.model.act_dim).float()
                next_obs, reward, done, truncated, info = self.env.step(tuple(acts.cpu().tolist()))
                
                rew_arr = np.asarray(reward, dtype=np.float32)
                ep_ret += rew_arr 
                # store transition
                traj.append((
                    obs, 
                    acts.cpu().numpy(), 
                    rew_arr,
                    values.cpu().numpy(), 
                    nlls.cpu().numpy(), 
                    kls.cpu().numpy(),
                    done, 
                    logits.cpu().numpy()  
                ))
                # # log for diagnostics
                # log = {
                #     'state': info['state'],
                #     'action': info['action'],
                #     'reward': reward,
                #     'obs': obs,
                #     'next_obs': info['obs0'],
                # }
                # log_l.append(log)
                obs = next_obs
            self.episode_returns.append(ep_ret)
            episodes += 1

        return traj, log_l

    def train(self):
        self.model.train()
        self.last_obs = self.env.reset()

        # Save initial hidden to re-evaluate sequence
        h_initial = self.h.clone()
        phase_size = int(self.cfg.total_steps / len(self.episode_counts))
        
        # Main training loop
        for global_step in range(self.cfg.total_steps):
            # Collect rollout
            traj, log_l = self._rollout()

            # Unpack trajectory
            obs_raw  = [s[0] for s in traj] # one timestep
            acts_np  = np.stack([s[1] for s in traj], axis=0)
            rews_np  = np.array([s[2] for s in traj], dtype=np.float32)
            vals_np  = np.stack([s[3] for s in traj], axis=0)
            nlls_np  = np.stack([s[4] for s in traj], axis=0)
            kls_np   = np.stack([s[5] for s in traj], axis=0)
            dones_np = np.array([s[6] for s in traj], dtype=np.float32)
            old_logits_np = np.stack([s[7] for s in traj], axis=0)  # prev logits 

            # if global_step > 0.95*self.cfg.total_steps:
            #     for log in log_l:
            #         print(log['state'])
            #         print(log['action'])
            #         pass
            # track success per phase
            # for done, rew in zip(dones_np, rews_np):
            #     if done:
            #         phase = min(int(global_step // phase_size), self.num_phases - 1)
            #         self.episode_counts[phase] += 1 
            #         if rew == 2.0:
            #             self.success_counts[phase] += 1

            # Advantage & returns
            adv_np, gae_ret_np = compute_gae_multi(
                rews_np, vals_np, dones_np,
                self.cfg.gamma, self.cfg.gae_lambda
            )
            # Flatten & Transform into tensor
            act_t = torch.from_numpy(acts_np.flatten()).long().to(self.device)
            adv_t = torch.from_numpy(adv_np.flatten()).float().to(self.device)
            gae_ret_t = torch.from_numpy(gae_ret_np.flatten()).float().to(self.device)
            # old_logits_np: shape (T, N, A) → reshape to (B=T*N, A)
            T, N, A = old_logits_np.shape
            old_logits_t = torch.from_numpy(old_logits_np.reshape(T * N, A)).float().to(self.device)
            
            # Re-evaluate under grad with RNN state progression
            self.model.train()
            logits_l, vals_l, nlls_l, kls_l, mus_l, logvars_l = [], [], [], [], [], []

            # 뒤에서 act_np를 one-hot 텐서로 변환
            acts = torch.from_numpy(acts_np).long().to(self.device)                         # (T, N)
            acts_onehot = F.one_hot(acts, num_classes=self.model.act_dim).float()          # (T, N, act_dim)

            # RNN 은닉 초기화
            h_re = h_initial.clone()

            # 시계열 재평가
            for t, o in enumerate(obs_raw):
                o_t = self.preprocess_obs(o)  # (N, obs_dim)
                # t=0일 땐 이전 액션이 없으므로 모두 0 벡터
                if t == 0:
                    a_prev = torch.zeros(self.nagents, self.model.act_dim, device=self.device)
                else:
                    a_prev = acts_onehot[t-1]   # (N, act_dim)

                # forward_step에 obs, a_prev, h_re 순서로 넘겨줌
                h_re, nll_s, kl_s, logits_s, vals_s, mu_s, logvar_s  = \
                    self.model.forward_step(o_t, a_prev, h_re)

                logits_l.append(logits_s)
                vals_l  .append(vals_s)
                nlls_l  .append(nll_s)
                kls_l   .append(kl_s)
                mus_l   .append(mu_s)
                logvars_l.append(logvar_s)
                
            # Stack into batch B = T * N
            logits_t = torch.vstack(logits_l)                    # (B, act_dim)
            val_t    = torch.cat( vals_l,   dim=0)               # (B,)
            nll_t    = torch.cat([x.unsqueeze(1) for x in nlls_l], dim=0).squeeze()  # (B,)
            kl_t     = torch.cat([x.unsqueeze(1) for x in kls_l],  dim=0).squeeze()  # (B,)
            mu_t     = torch.vstack(mus_l)                       # (B, d)
            logvar_t = torch.vstack(logvars_l)                   # (B, d)
            
            # Compute VAE loss with coop KL
            loss_vae, vae_metrics = compute_vae_loss(
                nll_t, kl_t,
                mu_t, logvar_t,
                nll_coef=self.cfg.nll_coef,
                kl_coef=self.cfg.kl_coef,
                coop_coef=self.cfg.coop_coef,
                n_agents=self.nagents
            )

            # # RL loss
            # loss_rl, rl_metrics = compute_rl_loss(
            #     logits_t, act_t, adv_t, val_t, gae_ret_t,
            #     ent_coef=self.cfg.ent_coef,
            #     value_coef=self.cfg.value_coef
            # )
            # PPO loss
            loss_rl, rl_metrics = compute_rl_ppo_loss(
                logits_t,            # new logits
                old_logits_t,        # old logits
                act_t,               # actions
                adv_t,               # advantages
                gae_ret_t,           # returns
                clip_eps=0.2,
                ent_coef=self.cfg.ent_coef,
                value_coef=self.cfg.value_coef,
                values_pred=val_t    # predicted values
            )

            rollout_return = float(np.sum(rews_np))
            rl_metrics['return'] = rollout_return

            # backward & step
            self.opt_vae.zero_grad()
            self.opt_rl.zero_grad()
            # self.opt.zero_grad()
            total_loss = loss_vae  + loss_rl # Loss_VAE : Loss_RL = 50 : 1
            total_loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.vae_params, self.cfg.clip_grad)
            # torch.nn.utils.clip_grad_norm_(self.rl_params, self.cfg.clip_grad)
            self.opt_vae.step()
            self.opt_rl.step()
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.clip_grad)
            # self.opt.step()
            
            # logging
            self.pbar.update()
            metrics = {**vae_metrics, **rl_metrics, 'total_loss': total_loss.item()}
            update_history(self.history, metrics)
            self.log_fn(metrics, global_step)

        self.pbar.close()
        # Plot all metrics
        plot_history(self.history)

        plot_episode_returns(self.episode_returns)
        # plot_phase_success(
        #     self.episode_counts,
        #     self.success_counts,
        #     phase_names=self.phase_names
        # )

