# MADP (Multi-Agent Deep Policy)

Multi-Agent 환경에서 SCM(Structural Causal Model)과 GAT(Graph Attention Network)를 통합한 강화학습 프레임워크입니다.

## 주요 기능

- **SCM (Structural Causal Model)**: Agent 간 인과관계 모델링
- **GAT (Graph Attention Network)**: Agent 간 통신 및 협력
- **Causal Structure GAT**: SCM의 인과구조를 GAT에 통합
- **Wandb 모니터링**: 실시간 학습 과정 추적

## GAT 타입

### 사용 가능한 GAT 타입
- `"none"`: GAT 사용 안함
- `"basic"`: 기본 GAT (GraphAttentionLayer)
- `"causal"`: CausalGAT (인과 마스킹 적용)
- `"causal_structure"`: CausalStructureGAT (SCM 인과구조 활용)
- `"causal_enhanced"`: CausalStructureEnhancedGAT (향상된 인과구조 활용)

### 설정 방법
```yaml
# configs.yaml
model:
  gat_type: "causal_structure"  # 원하는 GAT 타입 선택
```

## 설치

```bash
pip install -r requirements.txt
```

## Wandb 설정

### 1. Wandb 계정 생성 및 로그인
```bash
pip install wandb
wandb login
```

### 2. 실험 실행
```bash
python main_scm.py --config configs.yaml
```

## 모니터링 기능

### 실시간 모니터링 지표

#### **Loss 지표**
- `scm_loss`: SCM 예측 loss
- `causal_consistency_loss`: 인과구조 일관성 loss
- `do_loss`: Do-intervention loss
- `cf_loss`: Counterfactual loss
- `causal_gat_consistency_loss`: SCM-GAT 일관성 loss
- `causal_attention_alignment_loss`: Attention 정렬 loss
- `causal_structure_regularization_loss`: 인과구조 정규화 loss
- `policy_loss`: 정책 loss
- `value_loss`: 가치 함수 loss
- `total_loss`: 전체 loss

#### **Gradient 모니터링**
- `grad_norm/total`: 전체 gradient norm
- `grad_norm/{layer_name}`: 각 레이어별 gradient norm
- `param_norm/{layer_name}`: 각 레이어별 parameter norm
- `grad_param_ratio/{layer_name}`: Gradient/Parameter 비율

#### **Causal Structure 분석**
- `causal_structure/mean`: 인과구조 평균값
- `causal_structure/std`: 인과구조 표준편차
- `causal_structure/sparsity`: Sparsity 비율
- `causal_structure`: 인과구조 히트맵 이미지

#### **성능 지표**
- `episode_return/mean`: 에피소드 리턴 평균
- `episode_return/std`: 에피소드 리턴 표준편차
- `episode_return/max`: 에피소드 리턴 최대값
- `episode_return/min`: 에피소드 리턴 최소값

### Wandb 대시보드 활용

1. **실험 비교**: 여러 실험을 동시에 비교하여 최적 하이퍼파라미터 탐색
2. **Gradient 분석**: Gradient explosion/vanishing 문제 조기 발견
3. **Causal Structure 진화**: 인과구조가 학습 과정에서 어떻게 변화하는지 시각화
4. **성능 추적**: 실시간 성능 모니터링으로 조기 종료 결정

### 설정 예시

```yaml
# configs.yaml
params:
  cuda: True
  lr: 3e-4
  total_steps: 3000
  gamma: 0.99
  gae_lambda: 0.95
  ent_coef: 0.01
  value_coef: 1.0
  max_grad_norm: 1.0

model:
  gat_type: "basic"  # "none", "basic", "causal", "causal_structure", "causal_enhanced"
  hidden_dim: 48
  gat_dim: 16
  num_heads: 4
  dropout: 0.1
```

## 사용법

### 기본 실행
```bash
python main_scm.py --config configs.yaml
```

### 특정 환경에서 실행
```bash
python main_scm.py --config configs.yaml --env mpe
```

### Wandb 프로젝트 설정
```bash
export WANDB_PROJECT="SCM-GAT-MultiAgent"
python main_scm.py --config configs.yaml
```

## 결과 분석

### 1. Causal Structure 진화
- 학습 초기: 모든 agent 간 연결이 동등하게 가정
- 학습 진행: 실제 중요한 연결만 강화
- 학습 완료: Sparse한 인과구조 형성

### 2. Gradient 안정성
- Gradient norm이 일정 범위 내에서 유지되는지 확인
- Gradient explosion/vanishing 문제 조기 발견

### 3. 성능 최적화
- Episode return이 지속적으로 향상되는지 모니터링
- Loss 함수들이 균형있게 감소하는지 확인

## 문제 해결

### Wandb 연결 실패
```bash
# Wandb 비활성화
export WANDB_MODE=disabled
python main_scm.py --config configs.yaml
```

### 메모리 부족
```yaml
# configs.yaml에서 배치 크기 조정
params:
  batch_size: 16  # 기본값: 32
```

### Gradient 문제
```yaml
# Gradient clipping 강도 조정
params:
  max_grad_norm: 0.5  # 기본값: 1.0
```
