import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Dict, List, Any, Optional
import torch
import math 
import numpy as np
import sys

def create_progress_bar(total: int, desc: str = "Training Progress"):
    """Return a tqdm progress bar for the given total steps."""
    return tqdm(
        total=total,
        desc=desc,
        unit="steps",
        bar_format="║{bar:30}║ {percentage:3.0f}% │ {n_fmt}/{total_fmt} │ ⏱️ {elapsed} │ ⏳ {remaining} │ 🚀 {rate_fmt}",
        ncols=100,
        leave=False,
        colour="cyan",
        dynamic_ncols=True,
        position=0,
        disable=not sys.stdout.isatty()
    )

def init_history(keys: List[str]) -> Dict[str, List[float]]:
    """Initialize a history dict with empty lists for each metric key."""
    return {k: [] for k in keys}

def update_history(history: Dict[str, List[float]], metrics: Dict[str, Any]):
    """Append current metrics to history. Only keys present in history are recorded."""
    for key in history:
        if key in metrics:
            val = metrics[key]
            history[key].append(val.item() if torch.is_tensor(val) else float(val))

def plot_history(history: Dict[str, List[float]], save_path: str = "training_history.png", task_name: str = ""):
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

    # 제목에 task_name 추가
    if task_name:
        fig.suptitle(f"Training History - {task_name}", fontsize=14, y=0.98)

    fig.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Training history saved to: {save_path}")
    plt.show()  # 그래프를 화면에 표시

def plot_episode_returns(episode_returns, save_path: str = "episode_returns.png", task_name: str = ""):
    """
    episode_returns: List[np.ndarray]  (shape = (N_agents,))  또는
                     List[float]       (shape = ())
    """
    ep_arr = np.asarray(episode_returns)          # (B,)  또는 (B, N) - B는 배치 수
    batches = np.arange(1, ep_arr.shape[0] + 1)

    plt.figure(figsize=(8,4))

    if ep_arr.ndim == 1:          # 스칼라 리턴
        plt.plot(batches, ep_arr, label="total", alpha=0.3)
        # 이동평균
        window = 10  # 배치 기준으로 window 크기 조정
        if len(ep_arr) >= window:
            ma = np.convolve(ep_arr, np.ones(window)/window, mode="valid")
            plt.plot(batches[window-1:], ma, lw=2, label=f"{window}-batch mean")
    else:                         # 에이전트별
        window = 10  # 배치 기준으로 window 크기 조정
        for i in range(ep_arr.shape[1]):
            # raw return (옅은 색)
            plt.plot(batches, ep_arr[:, i], label=f"agent {i} (raw)", alpha=0.3)
            # 이동평균 (진한 색)
            if len(ep_arr[:, i]) >= window:
                ma = np.convolve(ep_arr[:, i], np.ones(window)/window, mode="valid")
                plt.plot(batches[window-1:], ma, lw=2, label=f"agent {i} {window}-batch mean")
        plt.legend()

    # 제목에 task_name 추가
    title = "Average Return per Batch"
    if task_name:
        title = f"Average Return per Batch - {task_name}"
    plt.title(title)
    plt.xlabel("Batch")
    plt.ylabel("Average Return")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Batch returns saved to: {save_path}")
    plt.show()  # 그래프를 화면에 표시

def plot_phase_success(episode_counts: List[int],
                       success_counts: List[int],
                       phase_names: Optional[List[str]] = None,
                       save_path: str = "phase_success.png",
                       task_name: str = ""):
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
    
    # 제목에 task_name 추가
    title = "Episode Success Rate by Phase"
    if task_name:
        title = f"Episode Success Rate by Phase - {task_name}"
    plt.title(title)
    plt.xlabel("Phase")
    plt.ylabel("Success Rate")
    plt.ylim(0, 1)
    plt.grid(axis='y')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Phase success plot saved to: {save_path}")
    plt.show()  # 그래프를 화면에 표시