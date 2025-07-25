import os
# OpenMP 라이브러리 충돌 방지
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from src.train import Trainer, get_device
from src.models import VRNNGATA2C
from src.utils import save_all_results
from pathlib import Path
from src.envs import DecTigerEnv
from src.env_wrapper import (DecTigerWrapper, 
                             RWAREWrapper, 
                             SMAXGymWrapper, 
                             MPEGymWrapper, 
                             SwitchWrapper,
                             PPWrapper,
                             LBForagingWrapper)

import yaml, argparse
import torch
import random
import numpy as np
import os
import gymnasium as gym

def set_seed(seed: int):
    """Fix all RNG seeds for reproducibility, including CuBLAS determinism."""
    os.environ['PYTHONHASHSEED']        = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # or ':16:8'
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

def create_env(env_name: str, env_cfg: dict, seed: int, device: torch.device | None = None):
    """GPU 최적화된 환경 생성 함수"""
    if env_name == 'dectiger':
        env = DecTigerWrapper(proj_dim=16, seed=seed, )
        obs_dim = env.obs_dim
        act_dim = env.act_dim
        nagents = env.n

    elif env_name == "rware":
        layout = env_cfg.get("layout", None)
        env_name_str = f"rware:rware-tiny-{env_cfg['nagents']}ag-v2"
        base_env = gym.make(env_name_str, layout=layout)
        env = RWAREWrapper(base_env, )
        obs_dim = env.obs_dim
        act_dim = env.act_dim
        nagents = env.n
        
    elif env_name == 'smax':
        env = SMAXGymWrapper(map_name="2s3z", seed=seed, )
        obs_dim = env.obs_dim
        act_dim = env.act_dim
        nagents = env.n

    elif env_name == 'mpe': # simple-spread
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
            n_agents = 2,
            max_steps = 50,
            
            )
        obs_dim = env.obs_dim
        act_dim = env.act_dim
        nagents = env.n

    elif env_name == 'pp':
        env = PPWrapper(n_agents=2, 
                        n_preys=1, 
                        max_steps=100, 
                        agent_view_mask=(5, 5),
                        
        )
        obs_dim = env.obs_dim
        act_dim = env.act_dim
        nagents = env.n

    elif env_name == 'foraging':
        env = LBForagingWrapper(grid_size=8, 
                                n_agents=3,
                                n_food=2, 
                                force_coop=True, 
                                sight=2,
                                seed=seed,
                                
        )
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

    # 설정 파일 로드 (UTF-8 인코딩 명시)
    cfg = yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))
    env_name = args.env         
    env_cfg = cfg[env_name] 

    # 시드 설정
    SEED = cfg['seed']
    set_seed(SEED)

    # GPU 디바이스 설정
    device = get_device(cfg)
    print(f"Using device: {device}")

    # 환경 생성
    env, obs_dim, act_dim, nagents = create_env(env_name, env_cfg, SEED, device)

    # 모델 생성 (VRNN-GAT 모델용 설정)
    model_cfg = cfg['model_vrnn']
    model = VRNNGATA2C(
        obs_dim=obs_dim,
        act_dim=act_dim,
        hidden_dim=model_cfg['hidden_dim'],
        z_dim=model_cfg['z_dim'],
        gat_dim=model_cfg['gat_dim'],
        n_agents=nagents,
        use_gat=model_cfg['use_gat'],
        use_rnn=model_cfg['use_rnn'],
    )
    # 훈련 설정
    from dataclasses import dataclass
    @dataclass
    class TrainConfig:
        ep_num: int
        total_steps: int
        lr: float
        lr_vae: float
        gamma: float
        vae_loss_type: str
        use_kl_annealing: bool
        gae_lambda: float
        clip_grad: float
        coop_coef: float
        ent_coef: float
        kl_coef: float
        nll_coef: float
        value_coef: float
        mixed_precision: bool
        ema_alpha: float
        
        def __post_init__(self):
            pass

    # 훈련 실행
    train_config = TrainConfig(
        ep_num=cfg['training']['ep_num'],
        total_steps=cfg['training']['total_steps'],
        lr=float(cfg['training']['lr']),
        lr_vae=float(cfg['training']['lr_vae']),
        gamma=float(cfg['training']['gamma']),
        gae_lambda=float(cfg['training']['gae_lambda']),
        clip_grad=float(cfg['params']['max_grad_norm']),
        coop_coef=float(cfg['training']['coop_coef']),
        ent_coef=float(cfg['training']['ent_coef']),
        kl_coef=float(cfg['training']['kl_coef']),
        nll_coef=float(cfg['training']['nll_coef']),
        value_coef=float(cfg['training']['value_coef']),
        mixed_precision=cfg['params'].get('mixed_precision', False),
        ema_alpha=float(cfg['training'].get('ema_alpha', 0.99)),
        vae_loss_type=cfg['training']['vae_loss_type'],
        use_kl_annealing=cfg['training']['use_kl_annealing'],
    )
    
    # 실험 설정을 딕셔너리로 변환
    experiment_config = {
        'env_name': env_name,
        'env_config': env_cfg,
        'model_config': model_cfg,
        'training_config': {
            'ep_num': train_config.ep_num,
            'total_steps': train_config.total_steps,
            'lr': train_config.lr,
            'lr_vae': train_config.lr_vae,
            'gamma': train_config.gamma,
            'gae_lambda': train_config.gae_lambda,
            'clip_grad': train_config.clip_grad,
            'coop_coef': train_config.coop_coef,
            'ent_coef': train_config.ent_coef,
            'kl_coef': train_config.kl_coef,
            'nll_coef': train_config.nll_coef,
            'value_coef': train_config.value_coef,
            'mixed_precision': train_config.mixed_precision,
            'ema_alpha': train_config.ema_alpha,
            'vae_loss_type': train_config.vae_loss_type,
            'use_kl_annealing': train_config.use_kl_annealing,
        },
        'params': cfg['params'],
        'seed': SEED,
        'device': str(device)
    }
    
    # 모델 정보 추가
    experiment_config.update({
        'obs_dim': obs_dim,
        'act_dim': act_dim,
        'n_agents': nagents,
        'hidden_dim': model_cfg['hidden_dim'],
        'gat_dim': model_cfg['gat_dim'],
        'z_dim': model_cfg['z_dim'],
        'use_gat': model_cfg['use_gat'],
        'use_rnn': model_cfg['use_rnn'],
        'vae_loss_type': train_config.vae_loss_type,
        'use_kl_annealing': train_config.use_kl_annealing,
    })
    
    trainer = Trainer(env, model, train_config, device=str(device), 
                     experiment_config=experiment_config)
    trainer.train()

if __name__ == "__main__":
    main()