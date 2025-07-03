import os
# OpenMP 라이브러리 충돌 방지
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from src.train import Trainer
from src.models import VRNNGATA2C
from pathlib import Path
from src.envs import DecTigerEnv
from src.env_wrapper import (DecTigerWrapper, 
                             RWAREWrapper, 
                             SMAXGymWrapper, 
                             MPEGymWrapper, 
                             SwitchWrapper,
                             PPWrapper)

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

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default=r"configs.yaml")
parser.add_argument("--env",    type=str, default="dectiger")
args = parser.parse_args()

cfg  = yaml.safe_load(Path(args.config).read_text())
env = args.env         
env_cfg = cfg[env] 

SEED=cfg['seed'] # random seed for torch opeartion, not for env
set_seed(SEED)

if env == 'dectiger':
    env = DecTigerWrapper(proj_dim=16, seed=SEED)
    obs_dim = env.obs_dim
    act_dim = env.act_dim
    nagents = env.n

elif args.env == "rware":
    layout = env_cfg.get("layout", None)
    env_name = f"rware:rware-tiny-{env_cfg['nagents']}ag-v2"
    base_env = gym.make(env_name, layout=layout)
    env = RWAREWrapper(base_env)
    obs_dim = env.obs_dim
    act_dim = env.act_dim
    nagents = env.n
    
elif env == 'smax':
    env = SMAXGymWrapper(map_name="2s3z", seed=SEED)
    obs_dim = env.obs_dim
    act_dim = env.act_dim
    nagents = env.n

elif env == 'mpe': # simple-spread
    env = MPEGymWrapper(
        n_agents=env_cfg.get("nagents", 3),
        local_ratio=env_cfg.get("local_ratio", 0.5),
        max_cycles=env_cfg.get("max_cycles", 25),
        continuous_actions=env_cfg.get("continuous_actions", False),
        seed=SEED
        )
    obs_dim = env.obs_dim
    act_dim = env.act_dim
    nagents = env.n

elif env == 'switch':
    env = SwitchWrapper(
        step_cost = -0.1,
        n_agents = 2,
        max_steps = 50,
        )
    obs_dim = env.obs_dim
    act_dim = env.act_dim
    nagents = env.n

elif env == 'pp':
    env = PPWrapper(n_agents=2, 
                    n_preys=1, 
                    max_steps=100, 
                    agent_view_mask=(5, 5)
    )
    obs_dim = env.obs_dim
    act_dim = env.act_dim
    nagents = env.n

else:
    ValueError("There is no such! Available envs are (dectiger, ...)")

hidden_dim = env_cfg.get("hidden_dim")
z_dim      = env_cfg.get("z_dim")
gat_dim    = env_cfg.get("gat_dim")

model = VRNNGATA2C(
    obs_dim=obs_dim,
    act_dim=act_dim,
    hidden_dim=hidden_dim,
    z_dim=z_dim,
    gat_dim=gat_dim,
    n_agents=nagents,
    use_gat=True
)

from dataclasses import dataclass
@dataclass
class TrainConfig:
    ep_num: int = 3       # the number of episodes per batch
    total_steps: int = 2000
    lr: float = 5e-4
    lr_vae: float = 1e-4  
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_grad: float = 1.0
    coop_coef: float = 0.01
    ent_coef: float = 0.01
    kl_coef: float = 0.5
    nll_coef: float = 1.0
    value_coef: float = 1.0

device = "cuda" if cfg['params']['cuda'] else "cpu"

trainer = Trainer(env, model, TrainConfig(), device)
print('Operation on ', trainer.device)
trainer.train()

# config = TrainConfig(
#     total_steps=1000,
#     replay_buffer_size=5000,
#     replay_batch_size=64,
#     lr=1e-4,
#     lr_vae=1e-4,
#     gamma=0.99,
#     coop_coef=0.01,
#     kl_coef=1.0,
#     nll_coef=0.02,
# )
# trainer_V = Trainer_V(env, model, config)
# print('Operation on ', trainer_V.device)
# trainer_V.train()