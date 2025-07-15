# Multi-Agent Causal Discovery 기반 SCM 모델 구조 정리

## 1. 전체 구조 개요

본 모델은 Multi-Agent 환경에서 인과구조(Structural Causal Model, SCM)를 추론하고, 이를 RL(A2C)과 통합하는 구조로 설계되어 있다. 핵심 구성요소는 다음과 같다:

- **ACD_SCM**: 인과구조 추론(encoder/decoder)
- **MultiAgentActorCritic**: 각 에이전트별 정책/가치 함수 + 인과 SCM 통합

---

## 2. ACD_SCM 구조

### (1) 입력
- Trajectory:  $\mathbf{X} \in \mathbb{R}^{B \times N \times T \times (d+1)}$
  - $B$: batch size
  - $N$: 에이전트 수
  - $T$: 시계열 길이
  - $d$: 관찰 차원 (obs_dim), +1은 reward

### (2) Encoder (MLPEncoder)
- 입력: 각 에이전트의 시계열을 flatten하여 node feature로 사용
- 인접행렬(관계): rel_rec, rel_send (off-diagonal one-hot)
- 출력: $ \text{rel\_type\_logits} \in \mathbb{R}^{B \times E \times K} $
  - $E = N(N-1)$: 엣지(에이전트 쌍) 개수
  - $K = \text{edge\_types} \times \text{관계종류}$ (예: 2×4=8)

#### 수식
- 각 엣지(e)에 대해:
  $A_{ij}^{(r)} = P(\text{edge } i \to j \text{ of type } r \text{ exists} \mid \mathbf{X}) = \text{softmax}(\text{logit}_{ij}^{(r)})[1]$
  - $A_{ij}^{(r)}$: 인과관계 존재 확률
  - $\text{logit}_{ij}^{(r)}$: encoder의 출력값

### (3) Decoder (MLPDecoder)
- 입력: 인과구조(softmax된 rel_type), trajectory
- 출력: 다음 시계열 예측(재구성)

---

## 3. MultiAgentActorCritic 구조

- 각 에이전트별로 actor/critic 네트워크를 가짐
- SCM(ACD_SCM)에서 추론한 인과구조를 활용

### (1) Actor
- 입력: 마지막 timestep의 obs
- 출력: 각 에이전트의 action logits

### (2) Critic
- 입력: 마지막 timestep의 obs
- 출력: 각 에이전트의 value

### (3) Forward 전체 흐름
1. trajectory 입력 → SCM(encoder) → 인과구조(logits, softmax)
2. SCM(decoder) → 시계열 예측
3. 마지막 obs → actor/critic → 정책/가치

---

## 4. 도식 (텍스트)

```
Trajectory (obs+reward)
      │
      ▼
  [MLPEncoder] ──▶ rel_type_logits ──softmax──▶ 인과구조 확률 (A_{ij}^{(r)})
      │                                        │
      ▼                                        │
  [MLPDecoder] ◀───────────────────────────────┘
      │
      ▼
  시계열 예측

(병렬)
Trajectory ──▶ 마지막 obs ──▶ [Actor/Critic] ──▶ 정책/가치
```

---

## 5. 인과관계 추론의 의미
- 각 엣지(에이전트 쌍, 관계별)에 대해 softmax(logits)로 인과관계 존재/비존재 확률을 추론
- 확률적 인과 그래프($A_{ij}^{(r)}$)로 해석 가능

---

## 6. 요약
- 본 구조는 trajectory 기반 인과구조 학습(ACD)과 multi-agent RL을 통합
- 인과관계의 유무는 확률적 adjacency matrix($A_{ij}^{(r)}$)로 표현
- 각 에이전트의 정책/가치 추론과 인과구조 추론이 end-to-end로 결합됨 