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
import wandb
import matplotlib.pyplot as plt

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

# ---------------------------------------------------------------------------
# Wandb Logging Functions
# ---------------------------------------------------------------------------
def ask_wandb_logging() -> bool:
    """
    학습 시작 전에 wandb 로깅 여부를 사용자에게 물어봅니다.
    Returns:
        True: wandb 로깅 사용
        False: wandb 로깅 사용 안함
    """
    try:
        answer = input("Wandb에 실험을 로깅하시겠습니까? (y/n): ").strip().lower()
        if answer in ["y", "yes", "1"]:
            print("Wandb 로깅을 활성화합니다.")
            return True
        else:
            print("Wandb 로깅을 비활성화합니다.")
            return False
    except Exception as e:
        print(f"입력 오류: {e}. 기본값(False)으로 진행합니다.")
        return False

def init_wandb(env_name: str, model, cfg, device: str) -> bool:
    """
    Wandb 초기화 및 설정
    
    Args:
        env_name: 환경 이름
        model: 모델 객체
        cfg: 설정 객체
        device: 디바이스
        
    Returns:
        wandb_enabled: wandb 초기화 성공 여부
    """
    try:
        # 실험 이름 생성
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"SCM_GAT_{env_name}_{timestamp}"
        
        # Wandb 설정
        wandb_config = {
            "env_name": env_name,
            "num_agents": model.num_agents,
            "model": {
                "gat_type": model.gat_type,
                "hidden_dim": model.hidden_dim,
                "gat_dim": model.gat_dim,
                "action_dim": model.action_dim,
                "obs_dim": model.obs_dim
            },
            "training": {
                "lr": cfg.lr,
                "total_steps": cfg.total_steps,
                "gamma": cfg.gamma,
                "gae_lambda": cfg.gae_lambda,
                "ent_coef": cfg.ent_coef,
                "value_coef": cfg.value_coef,
                "max_grad_norm": getattr(cfg, 'clip_grad', 1.0)
            },
            "device": str(device)
        }
        
        # Wandb 초기화
        wandb.init(
            project="SCM-GAT-MultiAgent",
            name=experiment_name,
            config=wandb_config,
            tags=[env_name, "SCM", "GAT", "MultiAgent"],
            notes=f"SCM-GAT Multi-Agent Training on {env_name}"
        )
        
        # 모델 구조를 wandb에 로그
        wandb.watch(model, log="all", log_freq=100)
        
        print(f"Wandb 초기화 완료: {experiment_name}")
        return True
        
    except Exception as e:
        print(f"Wandb 초기화 실패: {e}")
        return False

def log_gradients(model, wandb_enabled: bool = True) -> Optional[Dict[str, float]]:
    """
    Gradient 정보를 wandb에 로그
    
    Args:
        model: 모델 객체
        wandb_enabled: wandb 활성화 여부
        
    Returns:
        grad_norms: gradient norm 정보 딕셔너리
    """
    if not wandb_enabled:
        return None
        
    grad_norms = {}
    param_norms = {}
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            # Gradient norm
            grad_norm = param.grad.norm().item()
            grad_norms[f"grad_norm/{name}"] = grad_norm
            
            # Parameter norm
            param_norm = param.norm().item()
            param_norms[f"param_norm/{name}"] = param_norm
            
            # Gradient/Parameter ratio
            if param_norm > 0:
                grad_param_ratio = grad_norm / param_norm
                grad_norms[f"grad_param_ratio/{name}"] = grad_param_ratio
    
    # 전체 gradient norm
    total_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'))
    grad_norms["grad_norm/total"] = total_grad_norm
    
    # Wandb에 로그
    wandb.log(grad_norms)
    wandb.log(param_norms)
    
    return grad_norms

def log_causal_structure(causal_structure, wandb_enabled: bool = True, vmin=None, vmax=None):
    """
    Causal structure를 wandb에 로그
    Args:
        causal_structure: 인과구조 텐서
        wandb_enabled: wandb 활성화 여부
        vmin, vmax: heatmap의 컬러 범위 (None이면 자동으로 분위수 기반)
    """
    if not wandb_enabled:
        return
    arr = causal_structure.detach().cpu().numpy()
    # 자동 범위 설정: 5%~95% 분위수
    if vmin is None:
        vmin = np.percentile(arr, 5)
    if vmax is None:
        vmax = np.percentile(arr, 95)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(arr, cmap='viridis', vmin=vmin, vmax=vmax)
    ax.set_title('Causal Structure Matrix')
    ax.set_xlabel('Cause')
    ax.set_ylabel('Effect')
    plt.colorbar(im)
    wandb.log({"causal_structure": wandb.Image(fig)})
    plt.close(fig)
    # Causal structure 통계
    causal_stats = {
        "causal_structure/mean": causal_structure.mean().item(),
        "causal_structure/std": causal_structure.std().item(),
        "causal_structure/max": causal_structure.max().item(),
        "causal_structure/min": causal_structure.min().item(),
        "causal_structure/sparsity": (causal_structure < 0.1).float().mean().item()
    }
    wandb.log(causal_stats)

def log_episode_returns(batch_returns: List[np.ndarray], wandb_enabled: bool = True):
    """
    Episode returns를 wandb에 로그
    
    Args:
        batch_returns: 배치별 에피소드 리턴 리스트
        wandb_enabled: wandb 활성화 여부
    """
    if not wandb_enabled or len(batch_returns) == 0:
        return
        
    episode_return = batch_returns[-1]
    wandb.log({
        "episode_return/mean": np.mean(episode_return),
        "episode_return/std": np.std(episode_return),
        "episode_return/max": np.max(episode_return),
        "episode_return/min": np.min(episode_return)
    })

def log_metrics(metrics: Dict[str, float], wandb_enabled: bool = True):
    """
    메트릭을 wandb에 로그
    
    Args:
        metrics: 로그할 메트릭 딕셔너리
        wandb_enabled: wandb 활성화 여부
    """
    if wandb_enabled:
        wandb.log(metrics)

def finish_wandb(wandb_enabled: bool = True):
    """
    Wandb 종료
    
    Args:
        wandb_enabled: wandb 활성화 여부
    """
    if wandb_enabled:
        wandb.finish()
        print("Wandb 로깅 완료")

# ---------------------------------------------------------------------------
# Causal Structure Plotting Functions
# ---------------------------------------------------------------------------
def plot_causal_structure_evolution(causal_structure_list, output_dir, steps=None):
    import numpy as np
    import matplotlib.pyplot as plt
    arr = np.array(causal_structure_list)  # (steps, ...)
    n_phases = 4
    total_steps = arr.shape[0]
    phase_len = total_steps // n_phases
    fig, axes = plt.subplots(1, n_phases, figsize=(4*n_phases, 4))
    for i in range(n_phases):
        start = i * phase_len
        end = (i+1) * phase_len if i < n_phases-1 else total_steps
        phase_avg = arr[start:end].mean(axis=0)
        ax = axes[i]
        vmin = np.percentile(phase_avg, 5)
        vmax = np.percentile(phase_avg, 95)
        im = ax.imshow(phase_avg, cmap='viridis', vmin=vmin, vmax=vmax)
        ax.set_title(f'Phase {i+1}\nsteps {start}~{end}')
        ax.set_xlabel('Cause')
        ax.set_ylabel('Effect')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/causal_structure_phases.png")
    plt.close()

def plot_sparsity_analysis(arr: np.ndarray, output_dir: str):
    """
    Causal structure의 sparsity 변화 분석
    
    Args:
        arr: 인과구조 배열 (steps, N, N)
        output_dir: 저장할 디렉토리 경로
    """
    steps, N, _ = arr.shape
    
    # Sparsity 계산 (0에 가까운 entry들의 비율)
    sparsity_ratio = []
    mean_entries = []
    std_entries = []
    
    for step in range(steps):
        # 대각선 제외 (self-influence 제외)
        off_diagonal = arr[step].copy()
        np.fill_diagonal(off_diagonal, 0)
        
        # Sparsity ratio (0.1 이하의 값들의 비율)
        sparsity_ratio.append(np.mean(off_diagonal < 0.1))
        mean_entries.append(np.mean(off_diagonal))
        std_entries.append(np.std(off_diagonal))
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8))
    
    # Sparsity ratio 변화
    ax1.plot(sparsity_ratio, 'b-', linewidth=2)
    ax1.set_title('Sparsity Ratio Evolution (Ratio of entries < 0.1)')
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Sparsity Ratio')
    ax1.grid(True, alpha=0.3)
    
    # Mean entry 변화
    ax2.plot(mean_entries, 'r-', linewidth=2)
    ax2.set_title('Mean Entry Value Evolution')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Mean Entry Value')
    ax2.grid(True, alpha=0.3)
    
    # Standard deviation 변화
    ax3.plot(std_entries, 'g-', linewidth=2)
    ax3.set_title('Entry Standard Deviation Evolution')
    ax3.set_xlabel('Step')
    ax3.set_ylabel('Standard Deviation')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/sparsity_analysis.png")
    plt.show()
    
    # 결과 출력
    print(f"\n===== Sparsity Analysis =====")
    print(f"초기 Sparsity Ratio: {sparsity_ratio[0]:.3f}")
    print(f"최종 Sparsity Ratio: {sparsity_ratio[-1]:.3f}")
    print(f"Sparsity 변화: {sparsity_ratio[-1] - sparsity_ratio[0]:.3f}")
    print(f"초기 평균 Entry: {mean_entries[0]:.3f}")
    print(f"최종 평균 Entry: {mean_entries[-1]:.3f}")
    print(f"Entry 감소율: {(mean_entries[0] - mean_entries[-1]) / mean_entries[0] * 100:.1f}%")

def plot_entry_statistics(arr: np.ndarray, output_dir: str):
    """
    Entry 변화의 통계적 분석
    
    Args:
        arr: 인과구조 배열 (steps, N, N)
        output_dir: 저장할 디렉토리 경로
    """
    steps, N, _ = arr.shape
    
    # 각 entry의 변화량 계산
    initial_entries = arr[0]
    final_entries = arr[-1]
    change_entries = final_entries - initial_entries
    
    # 대각선 제거
    np.fill_diagonal(change_entries, 0)
    
    # 변화량 히스토그램
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    vmin = np.percentile(change_entries, 5)
    vmax = np.percentile(change_entries, 95)
    plt.hist(change_entries.flatten(), bins=20, alpha=0.7, color='blue')
    plt.title('Entry Change Distribution')
    plt.xlabel('Change in Entry Value')
    plt.ylabel('Frequency')
    
    # 감소한 entry들의 비율
    decreased_ratio = np.mean(change_entries < 0)
    increased_ratio = np.mean(change_entries > 0)
    unchanged_ratio = np.mean(change_entries == 0)
    
    plt.subplot(1, 3, 2)
    labels = ['Decreased', 'Increased', 'Unchanged']
    sizes = [decreased_ratio, increased_ratio, unchanged_ratio]
    colors = ['red', 'green', 'gray']
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%')
    plt.title('Entry Change Categories')
    
    # 가장 큰 변화를 보인 entry들
    plt.subplot(1, 3, 3)
    flat_changes = change_entries.flatten()
    top_decreases = np.argsort(flat_changes)[:5]  # 가장 큰 감소
    top_increases = np.argsort(flat_changes)[-5:]  # 가장 큰 증가
    
    plt.bar(range(5), flat_changes[top_decreases], color='red', alpha=0.7, label='Top Decreases')
    plt.bar(range(5, 10), flat_changes[top_increases], color='green', alpha=0.7, label='Top Increases')
    plt.title('Top 5 Entry Changes')
    plt.xlabel('Entry Index')
    plt.ylabel('Change Value')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/entry_statistics.png")
    plt.show()
    
    # 통계 출력
    print(f"\n===== Entry Change Statistics =====")
    print(f"감소한 Entry 비율: {decreased_ratio:.1%}")
    print(f"증가한 Entry 비율: {increased_ratio:.1%}")
    print(f"변화없는 Entry 비율: {unchanged_ratio:.1%}")
    print(f"평균 변화량: {np.mean(change_entries):.3f}")
    print(f"변화량 표준편차: {np.std(change_entries):.3f}")
    print(f"최대 감소량: {np.min(change_entries):.3f}")
    print(f"최대 증가량: {np.max(change_entries):.3f}")

# ---------------------------------------------------------------------------
# 기존 함수들
# ---------------------------------------------------------------------------

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
    def to_serializable(r):
        if isinstance(r, np.ndarray):
            return r.tolist()
        elif isinstance(r, np.generic):
            return float(r)
        else:
            return r
    returns_data = [to_serializable(r) for r in episode_returns]
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