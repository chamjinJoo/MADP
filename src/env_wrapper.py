from __future__ import annotations
from typing import Tuple, Dict, List, Any
import numpy as np
import copy
from PIL import Image, ImageDraw, ImageFont

# Optional imports with fallbacks
try:
    import jax
    import jax.random as jr
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    print("Warning: JAX not available. SMAX and MPE environments will not work.")

try:
    from jaxmarl.environments.smax import map_name_to_scenario
    from jaxmarl import make as jaxmarl_make
    JAXMARL_AVAILABLE = True
except ImportError:
    JAXMARL_AVAILABLE = False
    print("Warning: JaxMARL not available. SMAX and MPE environments will not work.")

from src.envs import DecTigerEnv
from src.gymenvs.switch.switch_one_corridor import Switch
from src.gymenvs.predator_prey.predator_prey import PredatorPrey

__all__ = [
    "SMAXGymWrapper",  # env adapter
    "Trainer",         # main training loop (imported below)
    "TrainConfig",     # dataclass imported below
]

class RWAREWrapper:
    """Lightweight wrapper adapting RWARE to (N, obs_dim) numpy API."""

    def __init__(self, env):
        self.name = "dectiger"
        self.env = env
        # Infer basic properties
        self.n: int = getattr(env, "n_agents", len(env.action_space))
        self.obs_dim: int = env.observation_space[0].shape[0]
        self.act_dim: int = env.action_space[0].n

    # --------------- helpers ---------------
    def _stack_obs(self, obs_tuple: Tuple[Any, ...]) -> np.ndarray:
        return np.stack([np.asarray(o, dtype=np.float32) for o in obs_tuple], axis=0)

    # --------------- gym‑style API ---------------
    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            obss, info = self.env.reset(seed=seed)
        else:
            obss, info = self.env.reset()
        return self._stack_obs(obss)

    def step(self, acts: Tuple[int, ...]):
        obss, rewards, done, truncated, info = self.env.step(acts)
        obs_mat = self._stack_obs(obss)
        rew_vec = np.asarray(rewards, dtype=np.float32)
        return obs_mat, rew_vec, done, truncated, info

    # optional passthroughs
    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)

    def close(self):
        self.env.close()

class MPEGymWrapper:
    """Lightweight adapter exposing JaxMARL MPE Simple Spread as a Gym-like Multi-Agent env."""
    def __init__(
        self,
        n_agents: int = 3,
        local_ratio: float = 0.5,
        max_cycles: int = 25,
        continuous_actions: bool = False,
        seed: int | None = None,
    ):
        if not JAXMARL_AVAILABLE:
            raise ImportError("JaxMARL is required for MPEGymWrapper. Install with: pip install git+https://github.com/instadeepai/jaxmarl.git")
        self.name = "mpe"
        # RNG key
        self.key = jr.PRNGKey(0 if seed is None else seed)
        # Instantiate the environment
        # Registry name: "MPE_simple_spread_v3"
        self.env = jaxmarl_make(
            "MPE_simple_spread_v3",
            N=n_agents,
            local_ratio=local_ratio,
            max_cycles=max_cycles,
            continuous_actions=continuous_actions,
        )
        self.state: Any = None
        # List of agent identifiers (e.g. ["agent_0", "agent_1", ...])
        self.agents = self.env.agents
        self.n = len(self.agents)

        # Infer observation & action dimensions
        sample_obs = self.env.observation_space(self.agents[0])
        self.obs_dim: int = sample_obs.shape[0]
        self.act_dim: int = self.env.action_space(self.agents[0]).n

    def _vec_obs(self, one_obs: Any) -> np.ndarray:
        # Observation is already a flat vector for MPE; just cast to float32
        return np.asarray(one_obs, dtype=np.float32)

    def reset(self) -> np.ndarray:
        """Reset env and return obs matrix of shape (N, obs_dim)."""
        self.key, sub = jr.split(self.key)
        obs_dict, self.state = self.env.reset(sub)
        obs = np.stack([self._vec_obs(obs_dict[a]) for a in self.agents], axis=0)
        return obs  # (N, obs_dim)

    def step(
        self,
        act_tuple: Tuple[int, ...]
    ) -> Tuple[np.ndarray, np.ndarray, bool, bool, Dict]:
        """
        Step the env.

        Parameters
        ----------
        act_tuple : tuple of ints
            Discrete actions for each agent in order `self.agents`.

        Returns
        -------
        obs      : np.ndarray, shape (N, obs_dim)
        rew      : np.ndarray, shape (N,)
        done     : bool
        truncated: bool (always False for MPE)
        info     : dict (raw env info)
        """
        self.key, sub = jr.split(self.key)
        act_dict = {a: int(x) for a, x in zip(self.agents, act_tuple)}

        obs_dict, self.state, rew_dict, done_dict, info = self.env.step(
            sub, self.state, act_dict
        )
        obs = np.stack([self._vec_obs(obs_dict[a]) for a in self.agents], axis=0)
        rew = np.asarray([rew_dict[a] for a in self.agents], dtype=np.float32)
        done = bool(done_dict["__all__"])
        # 성공 기준: info에 'success'가 있으면 사용, 없으면 False
        info = info or {}
        info['success'] = bool(info.get('success', False))
        return obs, rew, done, False, info
    
class SMAXGymWrapper:
    """Lightweight adapter exposing SMAX as if it were a Gym Multi‑Agent env."""

    def __init__(self, map_name: str = "3s5z", seed: int | None = None):
        if not JAXMARL_AVAILABLE:
            raise ImportError("JaxMARL is required for SMAXGymWrapper. Install with: pip install git+https://github.com/instadeepai/jaxmarl.git")
        self.name = "smax"
        self.key = jr.PRNGKey(0 if seed is None else seed)
        self.env = jaxmarl_make("HeuristicEnemySMAX", enemy_shoots=True, scenario=map_name_to_scenario(map_name))
        self.state = None  # will hold JAX env‑state
        self.agents: List[str] = self.env.agents  # e.g. ["agent-0", …]
        self.n = len(self.agents)

        # Infer observation & action dimensions (drop world_state)
        sample_obs = self.env.observation_space(self.agents[0])
        obs_dim_full = sample_obs.shape[0]
        self._world_state_dim = getattr(self.env, "world_state_dim", 0)
        self.obs_dim: int = obs_dim_full - self._world_state_dim
        self.act_dim: int = self.env.action_space(self.agents[0]).n

    # ------------- internal helpers -------------
    def _vec(self, one_obs: Dict[str, Any]) -> np.ndarray:
        """Convert per‑agent obs dict → 1‑D np.float32 vector, dropping world_state."""
        if self._world_state_dim == 0:
            return np.asarray(one_obs, dtype=np.float32)
        # SMAX packs into dict("obs", "world_state"), else list[float]
        if isinstance(one_obs, dict) and "obs" in one_obs:
            return np.asarray(one_obs["obs"], dtype=np.float32)
        # fallback: assume last k dims are world_state
        return np.asarray(one_obs[:-self._world_state_dim], dtype=np.float32)

    # ------------- Gym‑style API -------------
    def reset(self) -> np.ndarray:
        self.key, sub = jr.split(self.key)
        obs_dict, self.state = self.env.reset(sub)
        obs = np.stack([self._vec(obs_dict[a]) for a in self.agents], axis=0)
        return obs  # shape (N, obs_dim)

    def step(self, act_tuple: Tuple[int, ...]):
        """Step the env.\n
        Parameters
        ----------
        act_tuple : tuple of int
            Actions for each agent in order `self.agents`.
        Returns
        -------
        obs : np.ndarray, shape (N, obs_dim)
        rew : np.ndarray, shape (N,)
        done : bool (True if episode terminates for *all* agents)
        truncated : bool (always False; SMAX has no time‑limit truncation)
        info : dict (raw env info)
        """
        self.key, sub = jr.split(self.key)
        act_dict = {a: int(x) for a, x in zip(self.agents, act_tuple)}
        obs_dict, self.state, rew_dict, done_dict, info = self.env.step(
            sub, self.state, act_dict
        )
        obs = np.stack([self._vec(obs_dict[a]) for a in self.agents], axis=0)
        rew = np.asarray([rew_dict[a] for a in self.agents], dtype=np.float32)
        done = bool(done_dict["__all__"])
        return obs, rew, done, False, info  # Gym 5‑tuple

    # ------------- convenience -------------
    # def render(self):
    #     """Optional: call SMAX visualiser (slow)."""
    #     try:
    #         from jaxmarl.environments.smax.viz import visualize_state
    #         visualize_state(self.state)
    #     except ImportError:
    #         print("[SMAXGymWrapper] Visualization requires extra deps; skipped.")

class DecTigerWrapper:
    """DecTigerEnv를 gym-like 인터페이스로 래핑"""
    def __init__(self, proj_dim=16, seed=0):
        self.name = "dectiger"
        self.env = DecTigerEnv(proj_dim=proj_dim, seed=seed)
        self.n = self.env.nagents
        self.obs_dim = self.env.observation_space[0].shape[0]
        self.act_dim = self.env.action_space.nvec[0]  # MultiDiscrete([3,3])

    def reset(self, seed: int | None = None):
        if seed is not None:
            self.env.seed(seed)
        obs = self.env.reset()
        return np.stack(obs, axis=0)  # (N, obs_dim)

    def step(self, acts):
        # acts: (N,) or (N,1) or tuple/list
        if isinstance(acts, np.ndarray):
            acts = acts.tolist()
        obs, reward, done, info = self.env.step(acts)
        obs = np.stack(obs, axis=0)
        reward = np.array(reward, dtype=np.float32) if isinstance(reward, (list, tuple, np.ndarray)) else np.array([reward]*self.n, dtype=np.float32)
        # 성공 기준: reward > 0인 에이전트가 있으면 success
        success = np.any(reward == 2)
        info = info or {}
        info['success'] = bool(success)
        return obs, reward, done, False, info

    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)

    def close(self):
        pass

class SwitchWrapper:
    """Wrapper for the custom Switch2Env in src/envs/switch."""
    def __init__(
        self,
        full_observable: bool = False,
        step_cost: float = 0.0,
        n_agents: int = 4,
        max_steps: int = 50,
        clock: bool = True,
    ):
        # 원본 Switch2Env 초기화
        self.env = Switch(
            full_observable=full_observable,
            step_cost=step_cost,
            n_agents=n_agents,
            max_steps=max_steps,
            clock=clock,
        )
        self.name = "switch"
        self.n = self.env.n_agents
        sample_obs = self.env.reset()
        self.obs_dim = np.asarray(sample_obs, dtype=np.float32).shape[1]
        self.act_dim = 5

    def reset(self, seed: int | None = None) -> np.ndarray:
        obs_list = self.env.reset()
        return np.asarray(obs_list, dtype=np.float32)

    def step(
        self,
        actions: Tuple[int, ...]
    ) -> Tuple[np.ndarray, np.ndarray, bool, bool, Dict]:
        obs_list, rewards, dones, info = self.env.step(actions)
        obs = np.asarray(obs_list, dtype=np.float32)
        rew = np.asarray(rewards, dtype=np.float32)
        done = bool(all(dones))
        # 성공 기준: 모든 에이전트가 목표 위치에 도달하면 success
        # (예시: info에 'all_arrived'가 있으면 사용, 없으면 모든 reward > 0)
        if 'all_arrived' in (info or {}):
            success = bool(info['all_arrived'])
        else:
            success = bool(np.all(rew > 0))
        info = info or {}
        info['success'] = success
        return obs, rew, done, False, info

    def render(self, mode: str = "human"):
        """Switch 환경을 직접 렌더링합니다."""
        try:
            return self.env.render(mode)
        except ImportError:
            # Fallback: 직접 렌더링 구현
            return self._render_switch(mode)
    
    def _render_switch(self, mode: str = "human"):
        """Switch 환경을 직접 렌더링하는 메서드"""
        if not hasattr(self, '_base_img'):
            self._create_base_image()
        
        # 현재 상태로 이미지 생성
        img = copy.copy(self._base_img)
        
        # 에이전트 위치에 원 그리기
        for agent_i in range(self.n):
            pos = self.env.agent_pos[agent_i]
            self._draw_agent(img, pos, agent_i)
        
        img_array = np.asarray(img)
        
        if mode == 'rgb_array':
            return img_array
        elif mode == 'human':
            try:
                # Use matplotlib for real-time display
                import matplotlib.pyplot as plt
                if not hasattr(self, '_fig') or not hasattr(self, '_ax'):
                    plt.ion()  # Interactive mode
                    self._fig, self._ax = plt.subplots(figsize=(8, 6))
                    self._ax.set_title(f'Switch Environment - {self.n} agents')
                
                self._ax.clear()
                self._ax.imshow(img_array)
                self._ax.axis('off')
                self._ax.set_title(f'Switch Environment - {self.n} agents')
                plt.draw()
                plt.pause(0.01)  # Shorter pause for smoother animation
                return True
            except ImportError:
                print("Matplotlib not available, returning image array")
                return img_array
        return img_array
    
    def _create_base_image(self):
        """기본 그리드 이미지 생성"""
        from src.gymenvs.utils.draw import draw_grid, fill_cell, draw_cell_outline
        
        # 원본 Switch 환경의 상수들
        CELL_SIZE = 30
        WALL_COLOR = 'black'
        AGENT_COLORS = {0: 'red', 1: 'blue', 2: 'green', 3: 'orange'}
        
        # 3x7 그리드 생성
        self._base_img = draw_grid(3, 7, cell_size=CELL_SIZE, fill='white')
        
        # 벽 그리기 (중간 행과 양 끝 열)
        for row in range(3):
            for col in range(7):
                if row == 1 or col in [0, 1, 5, 6]:  # 벽 위치
                    fill_cell(self._base_img, (row, col), cell_size=CELL_SIZE, fill=WALL_COLOR)
        
        # 목표 위치에 아웃라인 그리기
        final_positions = {0: [0, 6], 1: [0, 0], 2: [2, 6], 3: [2, 0]}
        for agent_i in range(self.n):
            if agent_i in final_positions:
                pos = final_positions[agent_i]
                draw_cell_outline(self._base_img, pos, cell_size=CELL_SIZE, fill=AGENT_COLORS[agent_i])
        
        self._cell_size = CELL_SIZE
        self._agent_colors = AGENT_COLORS
    
    def _draw_agent(self, img, pos, agent_id):
        """에이전트를 이미지에 그리기"""
        from src.gymenvs.utils.draw import draw_circle, write_cell_text
        
        color = self._agent_colors.get(agent_id, 'gray')
        draw_circle(img, pos, cell_size=self._cell_size, fill=color, radius=0.3)
        write_cell_text(img, text=str(agent_id + 1), pos=pos, cell_size=self._cell_size,
                       fill='white', margin=0.4)

    def close(self):
        # Close matplotlib window if exists
        if hasattr(self, '_fig') and self._fig is not None:
            import matplotlib.pyplot as plt
            plt.close(self._fig)
            self._fig = None
            self._ax = None
        return self.env.close()
    
class PPWrapper:
    """Wrapper for the custom PredatorPrey environment."""
    def __init__(
        self,
        grid_shape=(5, 5),
        n_agents=2,
        n_preys=1,
        prey_move_probs=(0.175, 0.175, 0.175, 0.175, 0.3),
        full_observable=False,
        penalty=-0.5,
        step_cost=-0.01,
        prey_capture_reward=5,
        max_steps=100,
        agent_view_mask=(5, 5),
        seed: int | None = None,
    ):
        # PredatorPrey 환경 초기화
        self.env = PredatorPrey(
            grid_shape=grid_shape,
            n_agents=n_agents,
            n_preys=n_preys,
            prey_move_probs=prey_move_probs,
            full_observable=full_observable,
            penalty=penalty,
            step_cost=step_cost,
            prey_capture_reward=prey_capture_reward,
            max_steps=max_steps,
            agent_view_mask=agent_view_mask,
        )
        
        # seed 설정
        if seed is not None:
            self.env.seed(seed)
            
        self.name = "pp"
        self.n = self.env.n_agents
        
        # 관찰 공간과 행동 공간 정보 추출
        sample_obs = self.env.reset()
        self.obs_dim = len(sample_obs[0]) if isinstance(sample_obs, list) else sample_obs.shape[1]
        self.act_dim = 5  # PredatorPrey는 5개 행동 (DOWN, LEFT, UP, RIGHT, NOOP)

    def reset(self, seed: int | None = None) -> np.ndarray:
        """환경을 리셋하고 관찰 행렬을 반환합니다."""
        if seed is not None:
            self.env.seed(seed)
        obs_list = self.env.reset()
        return np.asarray(obs_list, dtype=np.float32)

    def step(
        self,
        actions: Tuple[int, ...]
    ) -> Tuple[np.ndarray, np.ndarray, bool, bool, Dict]:
        """        
        Parameters
        ----------
        actions :  (0-4: DOWN, LEFT, UP, RIGHT, NOOP)

        Returns
        -------
        obs      : np.ndarray, shape (N, obs_dim)
        rew      : np.ndarray, shape (N,)
        done     : bool
        truncated: bool (항상 False)
        info     : dict
        """
        obs_list, rewards, dones, info = self.env.step(actions)
        obs = np.asarray(obs_list, dtype=np.float32)
        rew = np.asarray(rewards, dtype=np.float32)
        
        # 전체 에피소드 종료 여부
        done = bool(all(dones)) if dones is not None else False
        
        # 성공 기준: 모든 prey가 잡히면 success (info에 'all_prey_captured'가 있으면 사용, 없으면 reward > 0)
        if 'all_prey_captured' in (info or {}):
            success = bool(info['all_prey_captured'])
        else:
            success = bool(np.any(rew > 0))
        info = info or {}
        info['success'] = success
        return obs, rew, done, False, info

    def render(self, mode: str = "human"):
        """PredatorPrey 환경을 직접 렌더링합니다."""
        try:
            return self.env.render(mode)
        except ImportError:
            # Fallback: 직접 렌더링 구현
            return self._render_predator_prey(mode)
    
    def _render_predator_prey(self, mode: str = "human"):
        """PredatorPrey 환경을 직접 렌더링하는 메서드"""
        if not hasattr(self, '_base_img'):
            self._create_pp_base_image()
        
        # 현재 상태로 이미지 생성
        img = copy.copy(self._base_img)
        
        # 에이전트와 prey 그리기
        for agent_i in range(self.n):
            pos = self.env.agent_pos[agent_i]
            self._draw_predator(img, pos, agent_i)
        
        for prey_i in range(self.env.n_preys):
            if self.env._prey_alive[prey_i]:
                pos = self.env.prey_pos[prey_i]
                self._draw_prey(img, pos, prey_i)
        
        img_array = np.asarray(img)
        
        if mode == 'rgb_array':
            return img_array
        elif mode == 'human':
            try:
                # Use matplotlib for real-time display
                import matplotlib.pyplot as plt
                if not hasattr(self, '_fig') or not hasattr(self, '_ax'):
                    plt.ion()  # Interactive mode
                    self._fig, self._ax = plt.subplots(figsize=(8, 6))
                    self._ax.set_title(f'PredatorPrey Environment - {self.n} agents')
                
                self._ax.clear()
                self._ax.imshow(img_array)
                self._ax.axis('off')
                self._ax.set_title(f'PredatorPrey Environment - {self.n} agents')
                plt.draw()
                plt.pause(0.01)  # Shorter pause for smoother animation
                return True
            except ImportError:
                print("Matplotlib not available, returning image array")
                return img_array
        return img_array
    
    def _create_pp_base_image(self):
        """PredatorPrey 기본 그리드 이미지 생성"""
        from src.gymenvs.utils.draw import draw_grid
        
        # 원본 PredatorPrey 환경의 상수들
        CELL_SIZE = 35
        AGENT_COLOR = (0, 0, 255)  # blue
        AGENT_NEIGHBORHOOD_COLOR = (186, 238, 247)
        PREY_COLOR = (255, 0, 0)  # red
        
        # 그리드 생성 (기본 5x5, 환경 설정에 따라 조정)
        grid_shape = self.env._grid_shape
        self._base_img = draw_grid(grid_shape[0], grid_shape[1], cell_size=CELL_SIZE, fill='white')
        
        self._cell_size = CELL_SIZE
        self._agent_color = AGENT_COLOR
        self._agent_neighborhood_color = AGENT_NEIGHBORHOOD_COLOR
        self._prey_color = PREY_COLOR
    
    def _draw_predator(self, img, pos, agent_id):
        """Predator를 이미지에 그리기"""
        from src.gymenvs.utils.draw import draw_circle, write_cell_text, fill_cell
        
        # 이웃 영역 표시
        for neighbor in self._get_neighbor_coordinates(pos):
            fill_cell(img, neighbor, cell_size=self._cell_size, 
                     fill=self._agent_neighborhood_color, margin=0.1)
        
        # 에이전트 그리기
        draw_circle(img, pos, cell_size=self._cell_size, fill=self._agent_color)
        write_cell_text(img, text=str(agent_id + 1), pos=pos, cell_size=self._cell_size,
                       fill='white', margin=0.4)
    
    def _draw_prey(self, img, pos, prey_id):
        """Prey를 이미지에 그리기"""
        from src.gymenvs.utils.draw import draw_circle, write_circle, write_cell_text
        
        draw_circle(img, pos, cell_size=self._cell_size, fill=self._prey_color)
        write_cell_text(img, text=str(prey_id + 1), pos=pos, cell_size=self._cell_size,
                       fill='white', margin=0.4)
    
    def _get_neighbor_coordinates(self, pos):
        """주변 좌표 반환"""
        neighbors = []
        grid_shape = self.env._grid_shape
        
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            new_pos = [pos[0] + dx, pos[1] + dy]
            if (0 <= new_pos[0] < grid_shape[0] and 
                0 <= new_pos[1] < grid_shape[1]):
                neighbors.append(new_pos)
        return neighbors

    def close(self):
        """환경을 종료합니다."""
        # Close matplotlib window if exists
        if hasattr(self, '_fig') and self._fig is not None:
            import matplotlib.pyplot as plt
            plt.close(self._fig)
            self._fig = None
            self._ax = None
        return self.env.close()