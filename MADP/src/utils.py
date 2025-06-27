import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Dict, List, Any, Optional
import torch
import math 
import numpy as np

def create_progress_bar(total: int):
    """Return a tqdm progress bar for the given total steps."""
    return tqdm(total=total)

def init_history(keys: List[str]) -> Dict[str, List[float]]:
    """Initialize a history dict with empty lists for each metric key."""
    return {k: [] for k in keys}

def update_history(history: Dict[str, List[float]], metrics: Dict[str, Any]):
    """Append current metrics to history. Only keys present in history are recorded."""
    for key in history:
        if key in metrics:
            val = metrics[key]
            history[key].append(val.item() if torch.is_tensor(val) else float(val))

def plot_history(history: Dict[str, List[float]]):
    """Plot each metric in history as subplots in a grid layout."""
    # 비어있지 않은 키만 모음
    keys = [k for k, v in history.items() if v]
    n = len(keys)
    if n == 0:
        return

    # 격자 크기 계산: cols = ceil(sqrt(n)), rows = ceil(n/cols)
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
    axes = axes.flatten()  # 1D array로 변환

    for idx, key in enumerate(keys):
        ax = axes[idx]
        vals = history[key]
        steps = list(range(len(vals)))
        ax.plot(steps, vals)
        ax.set_title(f"{key}")
        ax.set_xlabel("epoch")
        ax.set_ylabel(key)
        ax.grid(True)

    # 남는 subplot 축은 숨깁니다.
    for j in range(idx + 1, len(axes)):
        axes[j].axis('off')

    fig.tight_layout()
    plt.show()

def plot_episode_returns(episode_returns):
    """
    episode_returns: List[np.ndarray]  (shape = (N_agents,))  또는
                     List[float]       (shape = ())
    """
    ep_arr = np.asarray(episode_returns)          # (E,)  또는 (E, N)
    episodes = np.arange(1, ep_arr.shape[0] + 1)

    plt.figure(figsize=(8,4))

    if ep_arr.ndim == 1:          # 스칼라 리턴
        plt.plot(episodes, ep_arr, label="total")
    else:                         # 에이전트별
        for i in range(ep_arr.shape[1]):
            plt.plot(episodes, ep_arr[:, i], label=f"agent {i}")
        plt.legend()

    # 선택적: 100-step 이동 평균 추가
    if ep_arr.ndim == 1:
        window = 100
        if len(ep_arr) >= window:
            ma = np.convolve(ep_arr, np.ones(window)/window, mode="valid")
            plt.plot(episodes[window-1:], ma, lw=2, label=f"{window}-step mean")
    plt.title("Episodic Return over Episodes")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_phase_success(episode_counts: List[int],
                       success_counts: List[int],
                       phase_names: Optional[List[str]] = None):
    """
    Plot bar chart of success rate per phase.
    """
    # Default phase names
    if phase_names is None:
        phase_names = [f"Phase {i+1}" for i in range(len(episode_counts))]
    # Compute success rates
    rates = []
    for eps, succ in zip(episode_counts, success_counts):
        if eps > 0:
            rates.append(succ / eps)
        else:
            rates.append(0.0)
    # Plot
    plt.figure()
    plt.bar(phase_names, rates)
    plt.title("Episode Success Rate by Phase")
    plt.xlabel("Phase")
    plt.ylabel("Success Rate")
    plt.ylim(0, 1)
    plt.grid(axis='y')
    plt.show()