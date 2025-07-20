import random
import numpy as np
import gym
from gym import spaces
from typing import Dict, Any, List, Tuple, Set
import itertools


class DecTigerEnv(gym.Env):
    """
    Gym environment for Dec-Tiger with per-agent Gaussian random feature projection.
    Each agent receives its own high-dimensional continuous observation.
    Extended to 20 steps with time pressure and information degradation.
    """
    metadata = {'render.modes': ['human']}
    
    def __init__(self, proj_dim=32, seed=0, max_steps=2):
        super().__init__()
        # Actions: 0:listen, 1:open-left, 2:open-right
        self._actions = ["listen", "open-left", "open-right"]
        self.action_space = spaces.MultiDiscrete([3, 3])
        self.maxtimestep = max_steps
        self.nagents = 2
        # Base discrete obs for each agent: hear-left(0), hear-right(1)
        # but we won't use base_obs_space directly in projection wrapper
        self.base_obs_space = spaces.MultiDiscrete([2, 2])
        
        # Projection parameters: separate weight matrices for each agent
        self.proj_dim = proj_dim

        # --- RNG 인스턴스화 (전역 시드 제거) ---
        self.py_random = random.Random(seed)
        
        # Define per-agent continuous observation space
        obs_box = spaces.Box(low=-np.inf, high=np.inf, shape=(proj_dim,), dtype=np.float32)
        self.observation_space = spaces.Tuple((obs_box, obs_box))
        
        self.states = ["tiger-left", "tiger-right"]
        
        # 시간 압박 관련 설정
        self._base_listen_penalty = -0.2
        self._time_pressure_factor = 0.1  # 시간당 추가 페널티
        self._information_decay_rate = 0.05  # 정보 노이즈 증가율
        self._opportunity_cost_factor = 0.05  # 기회 비용
        
        self.reset()

    def seed(self, seed: int):
        """Seed both Python random and NumPy RNG."""
        self.py_random.seed(seed)
        return [seed]
    
    def reset(self):
        # Randomize hidden state
        self.time = 0
        self.state = self.py_random.choice(self.states)
        p = 0.85 if self.state=="tiger-left" else 0.15
        o0 = 1 if self.py_random.random() < p else 0
        o1 = 1 if self.py_random.random() < p else 0
        base_obs = (o0, o1)
        
        # Return projected observations for both agents
        return self._project_obs(base_obs)
    
    def step(self, action):
        """
        action: iterable of two ints in {0,1,2}
        returns: (obs0, obs1), reward, done, info
        """
        done = False
        a0, a1 = int(action[0]), int(action[1])
        ja = (self._actions[a0], self._actions[a1])

        
        # Compute reward with time pressure
        if ja == ("listen", "listen"):
            # 시간이 지날수록 listen 페널티 증가
            reward = -0.2
        else:
            done = True
            if ja == ("open-left", "open-left"):
                base_reward = -10 if self.state=="tiger-left" else +2.0
                reward = base_reward 
            elif ja == ("open-right", "open-right"):
                base_reward = -10 if self.state=="tiger-right" else +2.0
                reward = base_reward 
            elif ja in [("open-left","open-right"), ("open-right","open-left")]:
                reward = -10.1 
            else:
                opener = ja[1] if ja[0]=="listen" else ja[0]
                if opener == "open-left":
                    base_reward = -10 if self.state=="tiger-left" else +0.3
                else:
                    base_reward = -10 if self.state=="tiger-right" else +0.3
                reward = base_reward 
        
        # Generate base discrete obs per agent
        if ja == ("listen", "listen"):
            p = 0.85 if self.state=="tiger-left" else 0.15
            o0 = 1 if self.py_random.random() < p else 0
            o1 = 1 if self.py_random.random() < p else 0
        else:
            p = 1 if self.state=="tiger-left" else 0
            o0 = 1 if self.py_random.random() < p else 0
            o1 = 1 if self.py_random.random() < p else 0
        
        # Project observations with increasing noise over time
        obs0, obs1 = self._project_obs((o0, o1))
        
        info = {
            'state': self.state, 
            'action': ja,
            'obs0': 'hear-right' if o0 else 'hear-left', 
            'obs1': 'hear-right' if o1 else 'hear-left',
            'time': self.time,
        }
        self.time += 1
        if self.time == self.maxtimestep:
            done = True
        
        return (obs0, obs1), reward, done, info
    
    def _project_obs(self, base_obs):
        """
        base_obs: tuple of ints (o0, o1), each 0 or 1
        Returns:
            z0: proj_dim float vector for agent0
            z1: proj_dim float vector for agent1
        """
        o0, o1 = base_obs
        x0 = np.ones(self.proj_dim, dtype=np.float32) * o0
        x1 = np.ones(self.proj_dim, dtype=np.float32) * o1

        # 시간이 지날수록 노이즈 증가 (정보 품질 저하)
        base_noise_std = 0.1
        time_noise_std = self.time * self._information_decay_rate
        total_noise_std = base_noise_std + time_noise_std
        
        noise0 = self.py_random.gauss(0.0, total_noise_std)
        noise1 = self.py_random.gauss(0.0, total_noise_std)
        
        # 노이즈를 전체 벡터에 적용
        noise0_vec = np.full(self.proj_dim, noise0, dtype=np.float32)
        noise1_vec = np.full(self.proj_dim, noise1, dtype=np.float32)
        
        o0_proj = x0 + noise0_vec
        o1_proj = x1 + noise1_vec

        return o0_proj, o1_proj
    
    def render(self, mode='human'):
        print(f"Hidden tiger state: {self.state}")
        print(f"Current time step: {self.time}/{self.maxtimestep}")