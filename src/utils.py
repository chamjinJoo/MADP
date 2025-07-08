import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Dict, List, Any, Optional
import torch
import math 
import numpy as np
import sys
import json
import os
from datetime import datetime

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
            if torch.is_tensor(val):
                history[key].append(val.item())
            elif isinstance(val, (np.integer, np.floating)):
                history[key].append(float(val))
            else:
                history[key].append(float(val))

def save_experiment_config(config: Dict[str, Any], save_dir: str, experiment_name: str = ""):
    """
    실험 설정과 하이퍼파라미터를 JSON 파일로 저장합니다.
    
    Args:
        config: 저장할 설정 딕셔너리
        save_dir: 저장할 디렉토리 경로
        experiment_name: 실험 이름 (선택사항)
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 현재 시간을 포함한 파일명 생성
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if experiment_name:
        filename = f"config_{experiment_name}_{timestamp}.json"
    else:
        filename = f"config_{timestamp}.json"
    
    config_path = os.path.join(save_dir, filename)
    
    # torch tensor를 리스트로 변환
    def convert_tensors(obj):
        """Convert tensors and numpy types to JSON serializable format"""
        if isinstance(obj, torch.Tensor):
            return obj.cpu().numpy().tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_tensors(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_tensors(item) for item in obj]
        else:
            return obj
    
    config_serializable = convert_tensors(config)
    
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config_serializable, f, indent=2, ensure_ascii=False)
    
    print(f"실험 설정이 저장되었습니다: {config_path}")
    return config_path

def load_experiment_config(config_path: str) -> Dict[str, Any]:
    """
    저장된 실험 설정을 JSON 파일에서 로드합니다.
    
    Args:
        config_path: 설정 파일 경로
        
    Returns:
        로드된 설정 딕셔너리
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    print(f"실험 설정을 로드했습니다: {config_path}")
    return config

def create_config_summary(config: Dict[str, Any]) -> str:
    """
    설정 딕셔너리에서 주요 하이퍼파라미터를 요약한 문자열을 생성합니다.
    
    Args:
        config: 설정 딕셔너리
        
    Returns:
        하이퍼파라미터 요약 문자열
    """
    summary_parts = []
    
    # 주요 하이퍼파라미터들 (필요에 따라 수정 가능)
    key_params = [
        'learning_rate', 'lr', 'batch_size', 'hidden_dim', 'gat_dim', 'z_dim',
        'num_agents', 'num_episodes', 'max_steps', 'gamma', 'tau',
        'use_gat', 'use_causal_gat', 'mixed_precision'
    ]
    
    for key in key_params:
        if key in config:
            value = config[key]
            if isinstance(value, (int, float, str, bool)):
                summary_parts.append(f"{key}: {value}")
            elif isinstance(value, list) and len(value) <= 5:
                summary_parts.append(f"{key}: {value}")
    
    return " | ".join(summary_parts)

def plot_history(history: Dict[str, List[float]], save_path: str = "training_history.png", 
                task_name: str = "", config: Optional[Dict[str, Any]] = None):
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

    # 제목에 task_name과 하이퍼파라미터 정보 추가
    title = "Training History"
    if task_name:
        title = f"Training History - {task_name}"
    
    if config:
        config_summary = create_config_summary(config)
        if config_summary:
            title += f"\n{config_summary}"
    
    fig.suptitle(title, fontsize=12, y=0.98)

    fig.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Training history saved to: {save_path}")
    plt.show()  # 그래프를 화면에 표시

def plot_episode_returns(episode_returns, save_path: str = "episode_returns.png", 
                        task_name: str = "", config: Optional[Dict[str, Any]] = None):
    """
    episode_returns: List[np.ndarray]  (shape = (N_agents,))  또는
                     List[float]       (shape = ())
    """
    ep_arr = np.asarray(episode_returns)          # (B,)  또는 (B, N) - B는 배치 수
    batches = np.arange(1, ep_arr.shape[0] + 1)

    plt.figure(figsize=(8,4))

    if ep_arr.ndim == 1:          # 스칼라 리턴
        plt.plot(batches, ep_arr, label="total", alpha=0.1)
        # 이동평균
        window = 20  # 배치 기준으로 window 크기 조정
        if len(ep_arr) >= window:
            ma = np.convolve(ep_arr, np.ones(window)/window, mode="valid")
            plt.plot(batches[window-1:], ma, lw=2, label=f"{window}-batch mean")
    else:                         # 에이전트별
        window = 20  # 배치 기준으로 window 크기 조정
        for i in range(ep_arr.shape[1]):
            # raw return (옅은 색)
            plt.plot(batches, ep_arr[:, i], label=f"agent {i} (raw)", alpha=0.1)
            # 이동평균 (진한 색)
            if len(ep_arr[:, i]) >= window:
                ma = np.convolve(ep_arr[:, i], np.ones(window)/window, mode="valid")
                plt.plot(batches[window-1:], ma, lw=2, label=f"agent {i} {window}-batch mean")
        plt.legend()

    # 제목에 task_name과 하이퍼파라미터 정보 추가
    title = "Average Return per Batch"
    if task_name:
        title = f"Average Return per Batch - {task_name}"
    
    if config:
        config_summary = create_config_summary(config)
        if config_summary:
            title += f"\n{config_summary}"
    
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
                       task_name: str = "",
                       config: Optional[Dict[str, Any]] = None):
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
    
    # 제목에 task_name과 하이퍼파라미터 정보 추가
    title = "Episode Success Rate by Phase"
    if task_name:
        title = f"Episode Success Rate by Phase - {task_name}"
    
    if config:
        config_summary = create_config_summary(config)
        if config_summary:
            title += f"\n{config_summary}"
    
    plt.title(title)
    plt.xlabel("Phase")
    plt.ylabel("Success Rate")
    plt.ylim(0, 1)
    plt.grid(axis='y')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Phase success plot saved to: {save_path}")
    plt.show()  # 그래프를 화면에 표시

def save_all_results(history: Dict[str, List[float]], 
                    episode_returns: List, 
                    save_dir: str, 
                    task_name: str = "",
                    config: Optional[Dict[str, Any]] = None,
                    episode_counts: Optional[List[int]] = None,
                    success_counts: Optional[List[int]] = None,
                    phase_names: Optional[List[str]] = None):
    """
    모든 실험 결과를 한 번에 저장합니다.
    
    Args:
        history: 훈련 히스토리
        episode_returns: 에피소드 리턴 리스트
        save_dir: 저장할 디렉토리
        task_name: 태스크 이름
        config: 실험 설정
        episode_counts: 페이즈별 에피소드 수 (선택사항)
        success_counts: 페이즈별 성공 수 (선택사항)
        phase_names: 페이즈 이름들 (선택사항)
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 설정 저장
    if config:
        save_experiment_config(config, save_dir, task_name)
    
    # 그래프들 저장
    plot_history(history, os.path.join(save_dir, "training_history.png"), task_name, config)
    plot_episode_returns(episode_returns, os.path.join(save_dir, "episode_returns.png"), task_name, config)
    
    if episode_counts and success_counts:
        plot_phase_success(episode_counts, success_counts, phase_names, 
                          os.path.join(save_dir, "phase_success.png"), task_name, config)
    
    # 히스토리 데이터를 JSON으로도 저장
    history_path = os.path.join(save_dir, "training_history.json")
    with open(history_path, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
    
    # 에피소드 리턴 데이터도 저장
    returns_path = os.path.join(save_dir, "episode_returns.json")
    returns_data = [r.tolist() if isinstance(r, np.ndarray) else r for r in episode_returns]
    with open(returns_path, 'w', encoding='utf-8') as f:
        json.dump(returns_data, f, indent=2, ensure_ascii=False)
    
    print(f"모든 결과가 {save_dir} 폴더에 저장되었습니다.")
    print(f"저장된 파일들:")
    print(f"  - training_history.png")
    print(f"  - episode_returns.png")
    if episode_counts and success_counts:
        print(f"  - phase_success.png")
    print(f"  - training_history.json")
    print(f"  - episode_returns.json")
    if config:
        print(f"  - config_{task_name}_*.json")