import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from src.train_scm import SCMTrainer, get_device
from src.models_scm import MultiAgentActorCritic
from src.utils import save_all_results
from pathlib import Path
from src.envs import DecTigerEnv
from src.env_wrapper import (
    DecTigerWrapper, RWAREWrapper, SMAXGymWrapper, MPEGymWrapper, SwitchWrapper, PPWrapper, LBForagingWrapper
)
import yaml, argparse
import torch
import random
import numpy as np
import gymnasium as gym

def set_seed(seed: int):
    os.environ['PYTHONHASHSEED']        = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

def create_env(env_name: str, env_cfg: dict, seed: int, device: torch.device | None = None):
    if env_name == 'dectiger':
        env = DecTigerWrapper(proj_dim=16, seed=seed,)
        obs_dim = env.obs_dim
        act_dim = env.act_dim
        nagents = env.n
    elif env_name == "rware":
        layout = env_cfg.get("layout", None)
        env_name_str = f"rware:rware-tiny-{env_cfg['nagents']}ag-v2"
        base_env = gym.make(env_name_str, layout=layout)
        env = RWAREWrapper(base_env,)
        obs_dim = env.obs_dim
        act_dim = env.act_dim
        nagents = env.n
    elif env_name == 'smax':
        env = SMAXGymWrapper(map_name="2s3z", seed=seed,)
        obs_dim = env.obs_dim
        act_dim = env.act_dim
        nagents = env.n
    elif env_name == 'mpe':
        env = MPEGymWrapper(
            n_agents=env_cfg.get("nagents", 3),
            local_ratio=env_cfg.get("local_ratio", 0.5),
            max_cycles=env_cfg.get("max_cycles", 25),
            continuous_actions=env_cfg.get("continuous_actions", False),
            seed=seed,
        )
        obs_dim = env.obs_dim
        act_dim = env.act_dim
        nagents = env.n
    elif env_name == 'switch':
        env = SwitchWrapper(
            step_cost = -0.1,
            n_agents = 3,
            max_steps = 50,
        )
        obs_dim = env.obs_dim
        act_dim = env.act_dim
        nagents = env.n
    elif env_name == 'pp':
        env = PPWrapper(n_agents=2, n_preys=1, max_steps=100, agent_view_mask=(5, 5),)
        obs_dim = env.obs_dim
        act_dim = env.act_dim
        nagents = env.n
    elif env_name == 'foraging':
        env = LBForagingWrapper(grid_size=8, n_agents=3, n_food=2, force_coop=True, sight=2, seed=seed,)
        obs_dim = env.obs_dim
        act_dim = env.act_dim
        nagents = env.n
    else:
        raise ValueError(f"Unknown environment: {env_name}. Available envs are (dectiger, rware, smax, mpe, switch, pp, foraging)")
    return env, obs_dim, act_dim, nagents

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=r"configs.yaml")
    parser.add_argument("--env",    type=str, default="dectiger")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))
    env_name = args.env
    env_cfg = cfg[env_name]
    SEED = cfg['seed']
    set_seed(SEED)
    device = get_device(cfg)
    print(f"Using device: {device}")
    env, obs_dim, act_dim, nagents = create_env(env_name, env_cfg, SEED, device)
    # 모델 생성 (SCM 모델용 설정)
    model_cfg = cfg['model_scm']
    model = MultiAgentActorCritic(
        obs_dim=obs_dim,
        action_dim=act_dim,
        hidden_dim=model_cfg['hidden_dim'],
        num_agents=nagents,
        gat_type=model_cfg.get('gat_type', 'basic'),
        gat_dim=model_cfg.get('gat_dim', 32),
        num_heads=model_cfg.get('num_heads', 4),
        dropout=model_cfg.get('dropout', 0.1)
    )
    from dataclasses import dataclass
    @dataclass
    class TrainConfig:
        ep_num: int
        total_steps: int
        lr: float
        gamma: float
        gae_lambda: float
        clip_grad: float
        ent_coef: float
        value_coef: float
        episode_length: int
        def __post_init__(self):
            pass
    train_config = TrainConfig(
        ep_num=cfg['training']['ep_num'],
        total_steps=cfg['training']['total_steps'],
        lr=float(cfg['training']['lr']),
        gamma=float(cfg['training']['gamma']),
        gae_lambda=float(cfg['training']['gae_lambda']),
        clip_grad=float(cfg['params']['max_grad_norm']),
        ent_coef=float(cfg['training']['ent_coef']),
        value_coef=float(cfg['training']['value_coef']),
        episode_length=int(env_cfg.get('episode_length', 10))
    )
    experiment_config = {
        'env_name': env_name,
        'env_config': env_cfg,
        'model_config': model_cfg,
        'training_config': {
            'ep_num': train_config.ep_num,
            'total_steps': train_config.total_steps,
            'lr': train_config.lr,
            'gamma': train_config.gamma,
            'gae_lambda': train_config.gae_lambda,
            'clip_grad': train_config.clip_grad,
            'ent_coef': train_config.ent_coef,
            'value_coef': train_config.value_coef,
            'episode_length': train_config.episode_length
        },
        'params': cfg['params'],
        'seed': SEED,
        'device': str(device)
    }
    experiment_config.update({
        'obs_dim': obs_dim,
        'act_dim': act_dim,
        'n_agents': nagents,
        'hidden_dim': model_cfg['hidden_dim'],
        'gat_dim': model_cfg.get('gat_dim', 32),
        'gat_type': model_cfg.get('gat_type', 'basic')
    })
    trainer = SCMTrainer(env, model, train_config, device=str(device))
    trainer.train()

if __name__ == "__main__":
    main() 