from src.train import Trainer
from src.models import VRNNGATA2C
from pathlib import Path
from src.envs import DecTigerEnv
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
task = args.env         
task_cfg = cfg[task] 

SEED=cfg['seed'] # random seed for torch opeartion, not for env
set_seed(SEED)

if task == 'dectiger':
    env = DecTigerEnv(seed=SEED)
    env.seed(SEED) 
    env.action_space.seed(SEED)
    env.observation_space.seed(SEED)
elif task == 'rware':
    layout = """
    .......
    ...x...
    ..x.x..
    .x...x.
    ..x.x..
    ...x...
    .g...g.
    """
    env_name = "rware:rware-tiny-"+str(task_cfg["nagents"])+"ag-v2"
    env = gym.make(env_name, layout=layout)
    obs_dim =71
else:
    ValueError("There is no such! Available envs are (dectiger, ...)")

env.nagents = task_cfg["nagents"]

# obs_dim    = task_cfg["obs_dim"]
# hidden_dim = task_cfg.get("hidden_dim", obs_dim)
# z_dim      = task_cfg.get("z_dim",      obs_dim // 2)
# gat_dim    = task_cfg.get("gat_dim",    obs_dim // 2)

hidden_dim = task_cfg.get("hidden_dim")
z_dim      = task_cfg.get("z_dim")
gat_dim    = task_cfg.get("gat_dim")

model = VRNNGATA2C(
    obs_dim  = obs_dim,
    act_dim  = task_cfg["act_dim"],
    hidden_dim = hidden_dim,
    z_dim      = z_dim,
    gat_dim    = gat_dim,
    n_agents   = task_cfg["nagents"],
)

from dataclasses import dataclass
@dataclass
class TrainConfig:
    batch_size: int = 1       # the number of episodes per batch
    total_steps: int = 1000
    lr: float = 5e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_grad: float = 1.0
    coop_coef: float = 1.0
    ent_coef: float = 0.01
    kl_coef: float = 1.0
    nll_coef: float = 1.0
    value_coef: float = 1.0

trainer = Trainer(env, model, TrainConfig)
trainer.train()