# 1단계 준비 작업 구현 계획 (PPO 빈칸 코드와 채점기)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 사용자가 PPO 핵심 빈칸 5개를 직접 채울 수 있도록, 빈칸이 뚫린 `ppo.py`와 채점기, 나머지 코드(신경망, 저장소, 학습 반복문, 비교 그래프)를 만들고 정답 코드로 미리 검증한다.

**Architecture:** `ppo/` 폴더에 파일을 역할별로 나눈다. 사용자가 쓰는 코드는 `ppo.py` 한 파일에만 두고, 나머지 파일은 Claude가 작성한다. 정답 코드는 프로젝트 밖 임시 폴더에만 두고, 프로젝트 폴더를 복사한 곳에서 검증한다.

**Tech Stack:** Python 3.12, PyTorch 2.11, Isaac Lab 3.0.0rc1 (Isaac Sim 6.1), rsl-rl-lib 5.4.1 (비교 기준), pytest, tensorboard, matplotlib

**Spec:** [docs/specs/2026-10-09-stage1-ppo-design.md](../specs/2026-10-09-stage1-ppo-design.md)

## 이 계획을 읽는 사용자를 위한 안내

이 문서는 **Claude가 수행할 준비 작업**의 순서를 적은 것이다. 사용자가 직접 할 일은 이 계획이 모두 끝난 뒤에 시작된다(빈칸 채우기). 작업은 6개이며, 각 작업이 끝날 때마다 무엇을 확인했는지 보고한다.

| 작업 | 만드는 것 | 확인 방법 |
|---|---|---|
| 1 | 신경망 2개와 경험 저장소 (`model.py`) | 검사 코드 통과 |
| 2 | 빈칸 코드(`ppo.py`)와 채점기(`test_ppo.py`) | 빈칸 상태에서는 전부 실패하고, 정답 코드로는 전부 통과 |
| 3 | 학습 반복문 (`train.py`) | 정답 코드로 2번 반복하는 짧은 시험 실행 |
| 4 | 비교 그래프 (`plot_compare.py`)와 `rsl_rl` 기준 실험 3개 | 검사 코드 통과, 기준 실험 3개 완료 |
| 5 | 정답 코드로 전체 사전 확인 | 씨앗값 3개 모두 통과 기준 충족 |
| 6 | 기록 정리와 사용자에게 넘기기 | 프로젝트 폴더가 빈칸 상태인지 확인 |

## Global Constraints

- 모든 명령은 conda 환경 `env_isaaclab`에서 실행한다: `source ~/miniconda3/etc/profile.d/conda.sh && conda activate env_isaaclab`
- 시뮬레이터를 켜는 명령 앞에는 `export OMNI_KIT_ACCEPT_EULA=yes`를 둔다.
- 물리 엔진은 `rsl_rl` 기준 실험과 같은 `physics=isaacsim_physx`를 쓴다.
- 사용자가 쓰는 코드는 `ppo/ppo.py`에만 둔다. **정답 코드는 프로젝트 폴더에 절대 두지 않고**, 임시 폴더 `$S/answer/ppo.py`에만 둔다. 이 계획 문서에도 정답 코드를 적지 않는다(사용자가 이 문서를 읽기 때문이다).
- 임시 폴더: `S=/tmp/claude-1000/-home-tmdwodl------ISAACSIM-code/ee6c8af5-ce31-4af3-8d62-13daaa53396a/scratchpad`
- 설정값 (설계 문서 7장): 환경 4096개, 한 번에 16걸음, 150회 반복, 은닉층 `[32, 32]`, ELU, 관찰값 정규화 없음, 처음 표준편차 1.0, 잘림 범위 0.2, 가치 손실 계수 1.0, 엔트로피 계수 0.005, 고치기 반복 5회, 미니배치 4개, 할인율 0.99, λ 0.95, 기울기 최대 크기 1.0, 학습 속도 0.001 고정, 가치 손실 자르기 없음
- 통과 기준 (설계 문서 1장): 씨앗값 1, 2, 3 모두에서 150회째 반복의 평균 총점 4.5 이상, 평균 판 길이 290 이상
- 기록 이름은 `rsl_rl`과 같게 한다: `Train/mean_reward`, `Train/mean_episode_length`, `Loss/value`, `Loss/surrogate`, `Loss/entropy`, `Loss/learning_rate`, `Policy/mean_std`
- 시뮬레이터 실행 구간(`with launch_simulation(...)`)을 벗어나면 프로세스가 바로 종료된다. 저장과 기록 닫기는 반드시 그 구간 안에서 한다.
- 한국어로 쓰는 문서와 설명은 fluent-korean 지침을 따른다. 코드를 작성할 때는 ponytail 스킬을 사용한다 (사용자 전역 지침).
- git 커밋은 사용자가 허락한 경우에만 한다. 각 작업의 커밋 단계는 허락을 받았을 때만 수행한다.

## Review Focus

설계 문서에는 적혀 있지 않지만 실제로 쓰다가 문제가 되기 쉬운 경우 5가지이다. 각 항목을 검증하는 검사는 해당 작업에 넣었다.

1. **모든 이득이 같은 값일 때**: 이득을 정규화하다가 0으로 나누어 NaN이 생기면 학습이 조용히 망가진다. `ppo_update`는 이 경우에도 유한한 값을 유지해야 한다. → 작업 2 `test_update_constant_advantages_stay_finite`
2. **표준편차의 모양이 `(A,)`일 때와 `(N, A)`일 때**: 두 모양 모두에서 `log_prob`과 `entropy`가 `(N,)` 모양으로 나와야 한다. → 작업 2 `test_evaluate_action_accepts_per_env_std`, `test_sample_action_accepts_per_env_std`
3. **학습이 끝난 뒤 결과 파일이 실제로 남는지**: 시뮬레이터 실행 구간 밖에서 저장하면 파일이 생기지 않는다. → 작업 3의 시험 실행에서 신경망 파일과 기록 파일이 있는지 확인
4. **학습 기록이 없거나 일부 씨앗값만 있을 때**: `plot_compare.py`가 오류 추적 메시지를 쏟아내지 않고 무엇이 없는지 알려줘야 한다. → 작업 4 `test_find_runs_missing_root`, 그리고 기록이 없는 상태에서 직접 실행해 안내 문구 확인
5. **다른 폴더에서 명령을 실행할 때**: `ppo/` 폴더 안에서 `python train.py`를 실행해도 기록이 항상 `code/logs/`에 저장되어야 한다. → 작업 3에서 `ppo/` 폴더 안에서 실행해 기록 위치 확인

---

## 파일 구조

| 파일 | 역할 | 작성자 |
|---|---|---|
| `ppo/model.py` | 정책 신경망 `Actor`, 가치 신경망 `Critic`, 경험 저장소 `RolloutStorage` | Claude |
| `ppo/test_model.py` | `model.py` 검사 | Claude |
| `ppo/ppo.py` | 설정값 `PPOConfig`(완성)와 빈칸 함수 5개 | 사용자 (빈칸), Claude (틀) |
| `ppo/test_ppo.py` | 빈칸 채점기 | Claude |
| `ppo/train.py` | 학습 반복문, Isaac Lab 연결, 기록 | Claude |
| `ppo/plot_compare.py` | 비교 그래프와 통과 판정 | Claude |
| `ppo/test_plot_compare.py` | `plot_compare.py` 검사 | Claude |
| `$S/answer/ppo.py` | 정답 코드 (프로젝트 밖, 비공개) | Claude |

`ppo/` 폴더에는 `__init__.py`를 만들지 않는다. `python ppo/train.py`와 `pytest ppo/...`는 모두 `ppo/` 폴더를 모듈 검색 경로 맨 앞에 넣으므로, `from ppo import ...`와 `from model import ...`가 그대로 동작한다.

---

### Task 1: 신경망과 경험 저장소 (`model.py`)

**Files:**
- Create: `ppo/model.py`
- Test: `ppo/test_model.py`

**Interfaces:**
- Consumes: 없음
- Produces:
  - `Actor(obs_dim: int, act_dim: int, hidden=(32, 32), init_std=1.0)`: `forward(obs: (N, obs_dim)) -> mean: (N, act_dim)`, 속성 `std -> (act_dim,)`, 학습 가능한 값 `log_std`
  - `Critic(obs_dim: int, hidden=(32, 32))`: `forward(obs: (N, obs_dim)) -> value: (N,)`
  - `RolloutStorage(num_steps, num_envs, obs_dim, act_dim, device)`: 속성 `obs (T, N, obs_dim)`, `actions (T, N, act_dim)`, `log_probs`, `rewards`, `dones`, `values` (각각 `(T, N)`, 모두 float), 메서드 `add(obs, actions, log_probs, rewards, dones, values)`, `flatten(advantages, returns) -> dict`, `clear()`
  - `flatten`이 돌려주는 dict의 키: `"obs" (B, obs_dim)`, `"actions" (B, act_dim)`, `"log_probs" (B,)`, `"advantages" (B,)`, `"returns" (B,)`, 여기서 `B = T * N`

- [ ] **Step 1: pytest 설치**

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate env_isaaclab
pip install pytest
python -c "import pytest; print(pytest.__version__)"
```

Expected: 버전 번호가 출력된다 (예: `9.1.1`). 설치 후 `python -c "import torch, isaaclab; print(torch.__version__)"`가 `2.11.0+cu128`을 출력하는지도 확인한다.

- [ ] **Step 2: 실패하는 검사 작성**

`ppo/test_model.py`:

```python
"""model.py 검사. (시뮬레이터 없이 CPU에서 실행)

실행: pytest ppo/test_model.py
"""
import math

import pytest
import torch

from model import Actor, Critic, RolloutStorage


def test_actor_output_shapes():
    actor = Actor(obs_dim=4, act_dim=2)
    mean = actor(torch.zeros(7, 4))
    assert mean.shape == (7, 2)
    assert actor.std.shape == (2,)


def test_actor_initial_std_and_learnable():
    actor = Actor(obs_dim=4, act_dim=1, init_std=1.0)
    assert torch.allclose(actor.std, torch.ones(1))
    assert actor.log_std.requires_grad
    assert any(p is actor.log_std for p in actor.parameters())


def test_actor_matches_rsl_rl_network_size():
    # 4 -> 32 -> 32 -> 1 신경망의 값 개수 1249개 + 표준편차 1개
    actor = Actor(obs_dim=4, act_dim=1)
    assert sum(p.numel() for p in actor.parameters()) == 1250


def test_critic_output_shape():
    critic = Critic(obs_dim=4)
    assert critic(torch.zeros(7, 4)).shape == (7,)
    assert sum(p.numel() for p in critic.parameters()) == 1249


def _filled_storage(T=3, N=2):
    storage = RolloutStorage(num_steps=T, num_envs=N, obs_dim=1, act_dim=1, device="cpu")
    for t in range(T):
        mark = torch.tensor([t * 10.0 + n for n in range(N)])  # (걸음, 환경)마다 다른 번호표
        storage.add(
            obs=mark.unsqueeze(1),
            actions=mark.unsqueeze(1) + 0.1,
            log_probs=mark + 0.2,
            rewards=mark + 0.3,
            dones=torch.tensor([t == 1, False][:N]),
            values=mark + 0.4,
        )
    return storage


def test_storage_keeps_step_and_env_order():
    storage = _filled_storage()
    assert storage.obs.shape == (3, 2, 1)
    assert storage.rewards[2, 1].item() == pytest.approx(21.3)
    assert storage.dones.dtype == torch.float32
    assert storage.dones[1, 0].item() == 1.0 and storage.dones[0, 0].item() == 0.0


def test_storage_flatten_keeps_rows_aligned():
    storage = _filled_storage()
    advantages = storage.obs.squeeze(-1) + 0.5  # 같은 번호표로 표시
    returns = storage.obs.squeeze(-1) + 0.6
    batch = storage.flatten(advantages, returns)
    assert set(batch) == {"obs", "actions", "log_probs", "advantages", "returns"}
    assert batch["obs"].shape == (6, 1) and batch["advantages"].shape == (6,)
    mark = batch["obs"].squeeze(-1)
    assert torch.allclose(batch["actions"].squeeze(-1), mark + 0.1)
    assert torch.allclose(batch["log_probs"], mark + 0.2)
    assert torch.allclose(batch["advantages"], mark + 0.5)
    assert torch.allclose(batch["returns"], mark + 0.6)


def test_storage_rejects_overflow_and_early_flatten():
    storage = RolloutStorage(num_steps=2, num_envs=1, obs_dim=1, act_dim=1, device="cpu")
    row = dict(
        obs=torch.zeros(1, 1), actions=torch.zeros(1, 1), log_probs=torch.zeros(1),
        rewards=torch.zeros(1), dones=torch.zeros(1), values=torch.zeros(1),
    )
    storage.add(**row)
    with pytest.raises(RuntimeError):
        storage.flatten(torch.zeros(2, 1), torch.zeros(2, 1))  # 아직 다 안 찼음
    storage.add(**row)
    with pytest.raises(RuntimeError):
        storage.add(**row)  # 가득 찼음
    storage.clear()
    storage.add(**row)  # 비운 뒤에는 다시 담을 수 있음


def test_storage_copies_values():
    storage = RolloutStorage(num_steps=1, num_envs=1, obs_dim=1, act_dim=1, device="cpu")
    obs = torch.ones(1, 1)
    storage.add(obs, torch.zeros(1, 1), torch.zeros(1), torch.zeros(1), torch.zeros(1), torch.zeros(1))
    obs.fill_(99.0)  # 원본을 바꿔도 저장된 값은 그대로여야 함
    assert storage.obs[0, 0, 0].item() == 1.0
```

- [ ] **Step 3: 검사가 실패하는지 확인**

Run: `cd /home/tmdwodl/바탕화면/ISAACSIM/code && pytest ppo/test_model.py -q`
Expected: `ModuleNotFoundError: No module named 'model'`로 실패한다.

- [ ] **Step 4: `model.py` 작성**

`ppo/model.py`:

```python
"""신경망 2개와 경험 저장소. (준비된 코드: 읽고 이해하면 된다)

- Actor          : 관찰값을 받아 행동의 평균을 내놓는 정책 신경망
- Critic         : 관찰값을 받아 "지금 상황이 얼마나 좋은지"를 숫자 하나로 예측하는 가치 신경망
- RolloutStorage : 환경에서 모은 경험을 (걸음 수 T, 환경 수 N) 모양으로 담아두는 저장소
"""
import math

import torch
import torch.nn as nn


def mlp(in_dim, hidden, out_dim):
    """선형층 사이에 ELU를 끼운 단순한 신경망을 만든다."""
    layers, d = [], in_dim
    for h in hidden:
        layers += [nn.Linear(d, h), nn.ELU()]
        d = h
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)


class Actor(nn.Module):
    """정책 신경망: 관찰값 (N, obs_dim) -> 행동 평균 (N, act_dim)"""

    def __init__(self, obs_dim, act_dim, hidden=(32, 32), init_std=1.0):
        super().__init__()
        self.net = mlp(obs_dim, hidden, act_dim)
        # 행동의 퍼짐(표준편차). 관찰값과 무관하게 행동마다 하나씩 두고 신경망과 함께 학습한다.
        # 표준편차는 항상 양수여야 하므로 로그 값으로 저장하고, 쓸 때 exp를 취한다.
        self.log_std = nn.Parameter(torch.full((act_dim,), math.log(init_std)))

    def forward(self, obs):
        return self.net(obs)

    @property
    def std(self):
        return self.log_std.exp()  # (act_dim,)


class Critic(nn.Module):
    """가치 신경망: 관찰값 (N, obs_dim) -> 가치 (N,)"""

    def __init__(self, obs_dim, hidden=(32, 32)):
        super().__init__()
        self.net = mlp(obs_dim, hidden, 1)

    def forward(self, obs):
        return self.net(obs).squeeze(-1)


class RolloutStorage:
    """경험을 걸음 순서대로 담는다. 모든 값의 앞 두 축은 (걸음 T, 환경 N)이다."""

    def __init__(self, num_steps, num_envs, obs_dim, act_dim, device):
        self.num_steps = num_steps
        self.obs = torch.zeros(num_steps, num_envs, obs_dim, device=device)
        self.actions = torch.zeros(num_steps, num_envs, act_dim, device=device)
        self.log_probs = torch.zeros(num_steps, num_envs, device=device)
        self.rewards = torch.zeros(num_steps, num_envs, device=device)
        self.dones = torch.zeros(num_steps, num_envs, device=device)  # 판이 끝났으면 1.0
        self.values = torch.zeros(num_steps, num_envs, device=device)
        self.step = 0

    def add(self, obs, actions, log_probs, rewards, dones, values):
        """한 걸음의 경험을 담는다. 값을 복사해서 담으므로 원본이 나중에 바뀌어도 안전하다."""
        if self.step >= self.num_steps:
            raise RuntimeError("저장소가 가득 찼습니다. clear()를 먼저 호출하세요.")
        t = self.step
        self.obs[t].copy_(obs)
        self.actions[t].copy_(actions)
        self.log_probs[t].copy_(log_probs)
        self.rewards[t].copy_(rewards)
        self.dones[t].copy_(dones)  # bool이 들어와도 0.0 / 1.0으로 바뀐다
        self.values[t].copy_(values)
        self.step += 1

    def flatten(self, advantages, returns):
        """(T, N, ...) 모양을 (T*N, ...)로 펴서 ppo_update에 넘길 묶음(dict)을 만든다."""
        if self.step != self.num_steps:
            raise RuntimeError(f"저장소가 아직 다 차지 않았습니다 ({self.step}/{self.num_steps}).")
        return {
            "obs": self.obs.flatten(0, 1),
            "actions": self.actions.flatten(0, 1),
            "log_probs": self.log_probs.flatten(0, 1),
            "advantages": advantages.flatten(0, 1),
            "returns": returns.flatten(0, 1),
        }

    def clear(self):
        self.step = 0
```

- [ ] **Step 5: 검사가 통과하는지 확인**

Run: `cd /home/tmdwodl/바탕화면/ISAACSIM/code && pytest ppo/test_model.py -q`
Expected: `8 passed`

- [ ] **Step 6: 커밋 (사용자가 허락한 경우에만)**

```bash
cd /home/tmdwodl/바탕화면/ISAACSIM && git add code/ppo/model.py code/ppo/test_model.py
git commit -m "feat(ppo): add actor/critic networks and rollout storage"
```

---

### Task 2: 빈칸 코드(`ppo.py`)와 채점기(`test_ppo.py`)

**Files:**
- Modify: `ppo/ppo.py` (현재 빈 파일)
- Create: `ppo/test_ppo.py`
- Create (프로젝트 밖): `$S/answer/ppo.py`

**Interfaces:**
- Consumes: `model.Actor`, `model.Critic` (Task 1)
- Produces (`ppo/ppo.py`):
  - `PPOConfig` (dataclass): `gamma=0.99`, `lam=0.95`, `clip_param=0.2`, `value_coef=1.0`, `entropy_coef=0.005`, `num_epochs=5`, `num_mini_batches=4`, `max_grad_norm=1.0`, `learning_rate=1e-3`
  - `evaluate_action(mean: (N, A), std: (A,) | (N, A), action: (N, A)) -> (log_prob: (N,), entropy: (N,))`
  - `sample_action(mean: (N, A), std: (A,) | (N, A)) -> (action: (N, A), log_prob: (N,))`
  - `compute_gae(rewards: (T, N), values: (T, N), dones: (T, N) float, last_values: (N,), gamma: float, lam: float) -> (advantages: (T, N), returns: (T, N))`
  - `ppo_loss(new_log_prob, old_log_prob, advantages, new_values, returns, entropy, clip_param, value_coef, entropy_coef) -> (loss: 0차원 텐서, info: dict[str, float])`, `info`의 키: `"policy_loss"`, `"value_loss"`, `"entropy"`, `"approx_kl"`, `"clip_fraction"`
  - `ppo_update(actor, critic, optimizer, batch: dict, cfg: PPOConfig) -> info: dict[str, float]` (같은 키 5개, 모든 미니배치의 평균)

- [ ] **Step 1: 채점기 작성**

`ppo/test_ppo.py`:

```python
"""ppo.py 빈칸 채점기. (시뮬레이터 없이 CPU에서 몇 초 안에 끝난다)

빈칸 하나만 채점
    pytest ppo/test_ppo.py -k evaluate   (다-1 evaluate_action)
    pytest ppo/test_ppo.py -k sample     (다-2 sample_action)
    pytest ppo/test_ppo.py -k gae        (라   compute_gae)
    pytest ppo/test_ppo.py -k loss       (마   ppo_loss)
    pytest ppo/test_ppo.py -k update     (바   ppo_update)
전체 채점
    pytest ppo/test_ppo.py
"""
import copy
import math

import torch
from torch.distributions import Normal

from model import Actor, Critic
from ppo import PPOConfig, compute_gae, evaluate_action, ppo_loss, ppo_update, sample_action

INFO_KEYS = {"policy_loss", "value_loss", "entropy", "approx_kl", "clip_fraction"}


# ======================= 다-1. evaluate_action =======================

def test_evaluate_action_matches_torch():
    torch.manual_seed(1)
    mean, std, action = torch.randn(6, 3), torch.rand(3) + 0.1, torch.randn(6, 3)
    log_prob, entropy = evaluate_action(mean, std, action)
    dist = Normal(mean, std)
    assert log_prob.shape == (6,), "log_prob은 행동 개수(A) 방향으로 더해서 (N,) 모양이어야 합니다."
    assert entropy.shape == (6,), "entropy도 A 방향으로 더해서 (N,) 모양이어야 합니다."
    assert torch.allclose(log_prob, dist.log_prob(action).sum(-1), atol=1e-5), "로그 확률 수식을 확인하세요."
    assert torch.allclose(entropy, dist.entropy().sum(-1), atol=1e-5), "엔트로피 수식을 확인하세요."


def test_evaluate_action_accepts_per_env_std():
    torch.manual_seed(2)
    mean, std, action = torch.randn(6, 3), torch.rand(6, 3) + 0.1, torch.randn(6, 3)
    log_prob, entropy = evaluate_action(mean, std, action)
    dist = Normal(mean, std)
    assert torch.allclose(log_prob, dist.log_prob(action).sum(-1), atol=1e-5)
    assert torch.allclose(entropy, dist.entropy().sum(-1), atol=1e-5)


def test_evaluate_action_keeps_gradient():
    mean = torch.zeros(4, 2, requires_grad=True)
    log_std = torch.zeros(2, requires_grad=True)
    log_prob, entropy = evaluate_action(mean, log_std.exp(), torch.full((4, 2), 2.0))
    (log_prob.sum() + entropy.sum()).backward()
    msg = "기울기가 끊겼습니다. .item()이나 .detach()를 쓰지 않았는지 확인하세요."
    assert mean.grad is not None and mean.grad.abs().sum() > 0, msg
    assert log_std.grad is not None and log_std.grad.abs().sum() > 0, msg


# ======================= 다-2. sample_action =======================

def test_sample_action_shapes():
    action, log_prob = sample_action(torch.zeros(5, 3), torch.ones(3))
    assert action.shape == (5, 3), "action 모양은 (N, A)여야 합니다."
    assert log_prob.shape == (5,), "log_prob 모양은 (N,)이어야 합니다."


def test_sample_action_follows_mean_and_std():
    torch.manual_seed(0)
    mean = torch.tensor([1.0, -2.0]).repeat(200_000, 1)
    std = torch.tensor([0.5, 2.0])
    action, _ = sample_action(mean, std)
    assert torch.allclose(action.mean(0), mean[0], atol=0.02), "뽑은 행동들의 평균이 mean과 다릅니다."
    assert torch.allclose(action.std(0), std, atol=0.02), "뽑은 행동들의 퍼짐이 std와 다릅니다."


def test_sample_action_is_random():
    torch.manual_seed(0)
    mean = torch.zeros(4, 2)
    first, _ = sample_action(mean, torch.ones(2))
    second, _ = sample_action(mean, torch.ones(2))
    assert not torch.equal(first, second), "부를 때마다 다른 행동이 뽑혀야 합니다."


def test_sample_action_log_prob_matches_torch():
    torch.manual_seed(0)
    mean, std = torch.randn(6, 3), torch.rand(3) + 0.1
    action, log_prob = sample_action(mean, std)
    assert torch.allclose(log_prob, Normal(mean, std).log_prob(action).sum(-1), atol=1e-5)


def test_sample_action_accepts_per_env_std():
    torch.manual_seed(0)
    mean, std = torch.randn(6, 3), torch.rand(6, 3) + 0.1
    action, log_prob = sample_action(mean, std)
    assert action.shape == (6, 3) and log_prob.shape == (6,)
    assert torch.allclose(log_prob, Normal(mean, std).log_prob(action).sum(-1), atol=1e-5)


# ======================= 라. compute_gae =======================

def test_gae_hand_example():
    # 손으로 계산: 오차 = 1 + 0.9*0.5 - 0.5 = 0.95 (세 걸음 모두 같음)
    #   이득[2] = 0.95
    #   이득[1] = 0.95 + 0.9*0.8*0.95    = 1.634
    #   이득[0] = 0.95 + 0.9*0.8*1.634   = 2.12648
    rewards = torch.tensor([[1.0], [1.0], [1.0]])
    values = torch.tensor([[0.5], [0.5], [0.5]])
    adv, ret = compute_gae(rewards, values, torch.zeros(3, 1), torch.tensor([0.5]), gamma=0.9, lam=0.8)
    expected = torch.tensor([[2.12648], [1.634], [0.95]])
    assert adv.shape == (3, 1) and ret.shape == (3, 1)
    assert torch.allclose(adv, expected, atol=1e-5), f"기대값 {expected.flatten().tolist()}, 결과 {adv.flatten().tolist()}"
    assert torch.allclose(ret, expected + values, atol=1e-5), "returns는 advantages + values여야 합니다."


def test_gae_done_blocks_future_reward():
    # 걸음 1에서 판이 끝난다. 걸음 2의 큰 점수(100)는 다음 판의 것이므로 걸음 0, 1에 섞이면 안 된다.
    rewards = torch.tensor([[1.0], [2.0], [100.0]])
    dones = torch.tensor([[0.0], [1.0], [0.0]])
    adv, _ = compute_gae(rewards, torch.zeros(3, 1), dones, torch.tensor([50.0]), gamma=1.0, lam=1.0)
    assert torch.allclose(adv, torch.tensor([[3.0], [2.0], [150.0]])), "판이 끝난 걸음 너머의 이득이 섞였습니다."


def test_gae_done_blocks_next_value():
    # 걸음 0에서 판이 끝난다. 걸음 0의 오차를 계산할 때 다음 가치(5)를 쓰면 안 된다.
    rewards = torch.zeros(2, 1)
    values = torch.tensor([[1.0], [5.0]])
    dones = torch.tensor([[1.0], [0.0]])
    adv, _ = compute_gae(rewards, values, dones, torch.tensor([7.0]), gamma=1.0, lam=0.0)
    assert torch.allclose(adv, torch.tensor([[-1.0], [2.0]])), "판이 끝난 걸음에서는 다음 가치를 0으로 봐야 합니다."


def test_gae_lambda_zero_is_one_step_error():
    torch.manual_seed(3)
    rewards, values, last = torch.randn(5, 2), torch.randn(5, 2), torch.randn(2)
    adv, _ = compute_gae(rewards, values, torch.zeros(5, 2), last, gamma=0.9, lam=0.0)
    next_values = torch.cat([values[1:], last.unsqueeze(0)])
    assert torch.allclose(adv, rewards + 0.9 * next_values - values, atol=1e-5)


def test_gae_lambda_one_is_full_return():
    torch.manual_seed(4)
    rewards, values, last = torch.randn(5, 2), torch.randn(5, 2), torch.randn(2)
    adv, _ = compute_gae(rewards, values, torch.zeros(5, 2), last, gamma=1.0, lam=1.0)
    future_sum = rewards.flip(0).cumsum(0).flip(0) + last  # 그 걸음부터 끝까지 받은 점수 + 마지막 가치
    assert torch.allclose(adv, future_sum - values, atol=1e-5)


def test_gae_envs_are_independent():
    torch.manual_seed(5)
    T, N = 16, 7
    rewards, values, last = torch.randn(T, N), torch.randn(T, N), torch.randn(N)
    dones = (torch.rand(T, N) < 0.1).float()
    adv, ret = compute_gae(rewards, values, dones, last, gamma=0.99, lam=0.95)
    assert adv.shape == (T, N) and ret.shape == (T, N)
    for n in range(N):  # 환경 하나만 따로 계산한 결과와 같아야 한다
        one, _ = compute_gae(rewards[:, n:n + 1], values[:, n:n + 1], dones[:, n:n + 1], last[n:n + 1], 0.99, 0.95)
        assert torch.allclose(adv[:, n:n + 1], one, atol=1e-5), "환경끼리 값이 섞였습니다."


def test_gae_does_not_modify_inputs():
    rewards, values, dones, last = torch.ones(3, 2), torch.ones(3, 2), torch.zeros(3, 2), torch.ones(2)
    copies = [x.clone() for x in (rewards, values, dones, last)]
    compute_gae(rewards, values, dones, last, gamma=0.9, lam=0.8)
    for original, saved in zip((rewards, values, dones, last), copies):
        assert torch.equal(original, saved), "입력으로 받은 텐서를 바꾸면 안 됩니다."


# ======================= 마. ppo_loss =======================

def _loss(new_lp, old_lp, adv, new_values=None, returns=None, entropy=None, clip=0.2, vc=1.0, ec=0.0):
    n = new_lp.shape[0]
    new_values = torch.zeros(n) if new_values is None else new_values
    returns = torch.zeros(n) if returns is None else returns
    entropy = torch.zeros(n) if entropy is None else entropy
    return ppo_loss(new_lp, old_lp, adv, new_values, returns, entropy, clip, vc, ec)


def _policy_grad(ratio, adv, clip=0.2):
    """확률 비율이 ratio이고 이득이 adv일 때, 손실을 new_log_prob으로 미분한 값."""
    new_lp = torch.tensor([math.log(ratio)], requires_grad=True)
    loss, _ = _loss(new_lp, torch.zeros(1), torch.tensor([adv]), clip=clip)
    loss.backward()
    return new_lp.grad.item()


def test_loss_unchanged_policy():
    lp = torch.tensor([-1.0, -0.5, -2.0])
    loss, info = _loss(lp.clone(), lp, torch.tensor([1.0, -2.0, 4.0]))
    assert loss.dim() == 0, "loss는 숫자 하나(0차원 텐서)여야 합니다."
    assert math.isclose(loss.item(), -1.0, abs_tol=1e-6), "비율이 1이면 정책 손실은 -평균(이득)입니다."
    assert math.isclose(info["approx_kl"], 0.0, abs_tol=1e-6)
    assert info["clip_fraction"] == 0.0


def test_loss_value_term():
    lp = torch.zeros(2)
    loss, info = _loss(lp, lp, torch.zeros(2), new_values=torch.tensor([1.0, 2.0]), returns=torch.tensor([0.0, 4.0]), vc=0.5)
    assert math.isclose(info["value_loss"], 2.5, abs_tol=1e-6), "가치 손실은 (new_values - returns)^2의 평균입니다."
    assert math.isclose(loss.item(), 1.25, abs_tol=1e-6), "가치 손실에 value_coef를 곱해서 더해야 합니다."


def test_loss_entropy_term_lowers_loss():
    lp = torch.zeros(2)
    loss, info = _loss(lp, lp, torch.zeros(2), entropy=torch.tensor([2.0, 4.0]), ec=0.1)
    assert math.isclose(info["entropy"], 3.0, abs_tol=1e-6)
    assert math.isclose(loss.item(), -0.3, abs_tol=1e-6), "엔트로피가 클수록 손실이 작아져야 합니다 (빼기)."


def test_loss_total_is_sum_of_parts():
    torch.manual_seed(6)
    new_lp, old_lp, adv = torch.randn(8) * 0.1, torch.randn(8) * 0.1, torch.randn(8)
    loss, info = ppo_loss(new_lp, old_lp, adv, torch.randn(8), torch.randn(8), torch.rand(8), 0.2, 0.7, 0.03)
    expected = info["policy_loss"] + 0.7 * info["value_loss"] - 0.03 * info["entropy"]
    assert math.isclose(loss.item(), expected, abs_tol=1e-5)


def test_loss_clip_stops_gradient_when_already_improved():
    # 좋은 행동(이득 +1)의 확률을 이미 1.5배로 올렸다 -> 더 올리지 않도록 기울기가 0이어야 한다
    assert _policy_grad(1.5, +1.0) == 0.0, "비율이 1+ε를 넘으면 잘려서 기울기가 0이어야 합니다."


def test_loss_clip_stops_gradient_when_already_reduced():
    # 나쁜 행동(이득 -1)의 확률을 이미 0.5배로 내렸다 -> 더 내리지 않도록 기울기가 0이어야 한다
    assert _policy_grad(0.5, -1.0) == 0.0, "비율이 1-ε보다 작으면 잘려서 기울기가 0이어야 합니다."


def test_loss_no_clip_inside_range():
    assert math.isclose(_policy_grad(1.1, +1.0), -1.1, rel_tol=1e-5), "범위 안에서는 잘리지 않아야 합니다."


def test_loss_no_clip_when_change_was_harmful():
    # 나쁜 행동(이득 -1)의 확률을 1.5배로 올려버렸다 -> 되돌릴 수 있도록 기울기가 살아 있어야 한다
    assert math.isclose(_policy_grad(1.5, -1.0), 1.5, rel_tol=1e-5), "min을 써서 더 나쁜 쪽을 골라야 합니다."


def test_loss_info_values_and_types():
    new_lp = torch.tensor([0.0, math.log(1.5)])  # 비율 1.0과 1.5
    _, info = _loss(new_lp, torch.zeros(2), torch.ones(2))
    assert set(info) >= INFO_KEYS, f"info에 {sorted(INFO_KEYS)} 키가 모두 있어야 합니다."
    for key in INFO_KEYS:
        assert isinstance(info[key], float), f"info['{key}']는 .item()으로 바꾼 파이썬 숫자여야 합니다."
    assert math.isclose(info["clip_fraction"], 0.5, abs_tol=1e-6)
    assert math.isclose(info["approx_kl"], (0.5 - math.log(1.5)) / 2, abs_tol=1e-6)


# ======================= 바. ppo_update =======================

def _models(obs_dim=4, lr=1e-3, sgd=False):
    actor, critic = Actor(obs_dim, 1), Critic(obs_dim)
    params = list(actor.parameters()) + list(critic.parameters())
    optimizer = torch.optim.SGD(params, lr=lr) if sgd else torch.optim.Adam(params, lr=lr)
    return actor, critic, optimizer


def _batch(actor, obs):
    """지금 정책으로 행동을 뽑아서 ppo_update에 넣을 묶음을 만든다."""
    with torch.no_grad():
        dist = Normal(actor(obs), actor.std)
        actions = dist.sample()
        log_probs = dist.log_prob(actions).sum(-1)
    n = obs.shape[0]
    return {"obs": obs, "actions": actions, "log_probs": log_probs,
            "advantages": torch.randn(n), "returns": torch.randn(n)}


def _flat_params(*modules):
    return torch.cat([p.detach().flatten() for m in modules for p in m.parameters()])


def test_update_uses_all_data_once_per_epoch():
    torch.manual_seed(0)
    actor, critic, optimizer = _models(obs_dim=1)
    B = 64
    batch = _batch(actor, torch.arange(B, dtype=torch.float32).unsqueeze(1))  # 관찰값을 0..63 번호표로 쓴다
    seen = []
    actor.register_forward_hook(lambda module, inputs, output: seen.append(inputs[0].detach().flatten().clone()))
    ppo_update(actor, critic, optimizer, batch, PPOConfig(num_epochs=3, num_mini_batches=4))
    assert len(seen) == 12, "actor는 미니배치마다 한 번씩, 총 num_epochs × num_mini_batches번 불러야 합니다."
    assert all(len(s) == 16 for s in seen), "미니배치 크기는 B / num_mini_batches여야 합니다."
    for epoch in range(3):
        ids = torch.cat(seen[epoch * 4:(epoch + 1) * 4]).sort().values
        assert torch.equal(ids, torch.arange(B, dtype=torch.float32)), "한 번의 반복에서 모든 경험을 빠짐없이 한 번씩 써야 합니다."
    assert not torch.equal(seen[0], torch.arange(16, dtype=torch.float32)), "순서를 섞어야 합니다 (torch.randperm)."
    assert not torch.equal(seen[0], seen[4]), "반복마다 새로 섞어야 합니다."


def test_update_changes_both_networks():
    torch.manual_seed(0)
    actor, critic, optimizer = _models()
    before_actor, before_critic = _flat_params(actor), _flat_params(critic)
    ppo_update(actor, critic, optimizer, _batch(actor, torch.randn(64, 4)), PPOConfig())
    assert not torch.equal(before_actor, _flat_params(actor)), "정책 신경망이 바뀌지 않았습니다."
    assert not torch.equal(before_critic, _flat_params(critic)), "가치 신경망이 바뀌지 않았습니다."


def test_update_returns_averaged_info():
    torch.manual_seed(0)
    actor, critic, optimizer = _models()
    info = ppo_update(actor, critic, optimizer, _batch(actor, torch.randn(64, 4)), PPOConfig())
    assert set(info) >= INFO_KEYS
    for key in INFO_KEYS:
        assert isinstance(info[key], float) and math.isfinite(info[key]), f"info['{key}']가 유한한 파이썬 숫자가 아닙니다."


def test_update_moves_policy_toward_better_actions():
    # 가짜 문제: 행동 값이 클수록 좋다(이득 = 행동 값). 여러 번 고치면 행동 평균이 커져야 한다.
    torch.manual_seed(0)
    actor, critic, optimizer = _models(obs_dim=1, lr=1e-2)
    obs = torch.zeros(256, 1)
    before = actor(obs[:1]).item()
    for _ in range(20):
        batch = _batch(actor, obs)
        batch["advantages"] = batch["actions"].squeeze(-1).clone()
        batch["returns"] = torch.zeros(256)
        ppo_update(actor, critic, optimizer, batch, PPOConfig())
    assert actor(obs[:1]).item() > before + 0.3, "좋은 행동 쪽으로 정책이 움직이지 않았습니다. 손실의 부호를 확인하세요."


def test_update_normalizes_advantages():
    # 이득에 1000을 곱하고 5를 더해도, 정규화를 하면 결과가 같아야 한다
    torch.manual_seed(0)
    actor, critic, _ = _models()
    batch = _batch(actor, torch.randn(64, 4))
    results = []
    for scale, shift in ((1.0, 0.0), (1000.0, 5.0)):
        a, c = copy.deepcopy(actor), copy.deepcopy(critic)
        optimizer = torch.optim.Adam(list(a.parameters()) + list(c.parameters()), lr=1e-3)
        scaled = dict(batch, advantages=batch["advantages"] * scale + shift)
        torch.manual_seed(123)
        ppo_update(a, c, optimizer, scaled, PPOConfig())
        results.append(_flat_params(a))
    assert torch.allclose(results[0], results[1], atol=1e-3), "이득을 배치 전체 기준으로 정규화(평균 0, 표준편차 1)해야 합니다."


def test_update_constant_advantages_stay_finite():
    torch.manual_seed(0)
    actor, critic, optimizer = _models()
    batch = _batch(actor, torch.randn(64, 4))
    batch["advantages"] = torch.ones(64)  # 표준편차가 0
    ppo_update(actor, critic, optimizer, batch, PPOConfig())
    assert torch.isfinite(_flat_params(actor, critic)).all(), "0으로 나누어 NaN이 생겼습니다. 나눌 때 1e-8을 더하세요."


def test_update_clips_gradient_norm():
    # 가치 목표를 아주 크게 줘서 기울기를 폭발시킨다. 기울기 크기를 자르면 한 걸음에 바뀌는 양이 제한된다.
    torch.manual_seed(0)
    actor, critic, optimizer = _models(lr=1.0, sgd=True)
    batch = _batch(actor, torch.randn(64, 4))
    batch["returns"] = torch.full((64,), 1e6)
    before = _flat_params(actor, critic)
    ppo_update(actor, critic, optimizer, batch, PPOConfig(num_epochs=1, num_mini_batches=1, max_grad_norm=1e-3))
    moved = (_flat_params(actor, critic) - before).norm().item()
    assert 0 < moved <= 1.5e-3, f"기울기 크기를 max_grad_norm으로 잘라야 합니다 (바뀐 양 {moved:.3g})."
```

- [ ] **Step 2: 빈칸 코드 작성**

`ppo/ppo.py`:

```python
"""PPO의 핵심 (직접 작성하는 파일).

아래 빈칸 함수 5개를 위에서부터 순서대로 채운다.

    다-1. evaluate_action   행동의 로그 확률과 엔트로피 계산
    다-2. sample_action     행동 뽑기
    라.   compute_gae       이득 계산
    마.   ppo_loss          PPO 손실
    바.   ppo_update        고치기 반복

빈칸 하나를 채울 때마다 그 함수 설명 끝에 적힌 채점 명령을 실행한다.

표기
    N = 동시에 도는 환경 수 (수레-막대는 4096)
    T = 한 번에 모으는 걸음 수 (16)
    A = 행동 개수 (수레-막대는 1)
    B = T * N (모은 경험 전체 개수)
"""
import math
from dataclasses import dataclass

import torch


@dataclass
class PPOConfig:
    """PPO 설정값. (완성된 코드: 고칠 필요 없음)"""

    gamma: float = 0.99          # 할인율: 먼 미래의 점수를 얼마나 깎아서 볼지
    lam: float = 0.95            # 이득 계산에서 여러 걸음의 오차를 섞는 정도 (GAE의 λ)
    clip_param: float = 0.2      # 확률 비율을 자르는 범위 ε
    value_coef: float = 1.0      # 가치 손실에 곱하는 계수
    entropy_coef: float = 0.005  # 엔트로피에 곱하는 계수
    num_epochs: int = 5          # 모은 경험을 몇 번 반복해서 쓸지
    num_mini_batches: int = 4    # 한 번 반복할 때 경험을 몇 묶음으로 나눌지
    max_grad_norm: float = 1.0   # 기울기 크기의 최대값
    learning_rate: float = 1e-3  # 학습 속도


def evaluate_action(mean, std, action):
    """[빈칸 다-1] 이미 정해진 행동이 정규분포 N(mean, std)에서 나올 로그 확률과, 분포의 엔트로피를 구한다.

    학습할 때는 "예전에 했던 행동이 지금 신경망에서는 얼마나 자주 나올까"를 알아야 한다.
    그 값을 구하는 함수이다. 엔트로피는 분포가 얼마나 넓게 퍼져 있는지를 나타낸다.

    입력
        mean   : (N, A)            정책 신경망이 내놓은 행동 평균
        std    : (A,) 또는 (N, A)  행동의 표준편차
        action : (N, A)            평가할 행동
    출력
        log_prob : (N,)  행동의 로그 확률 (행동 A개의 값을 더한 것)
        entropy  : (N,)  분포의 엔트로피 (행동 A개의 값을 더한 것)

    수식 (행동 하나마다 계산한 뒤 마지막 축으로 더한다)
        로그 확률 = -(action - mean)^2 / (2 * std^2) - log(std) - 0.5 * log(2π)
        엔트로피  = 0.5 + 0.5 * log(2π) + log(std)

    힌트
        - std가 (A,) 모양이면 std.expand_as(mean)으로 (N, A) 모양으로 넓힌 뒤 계산한다.
        - log(2π)는 math.log(2 * math.pi)로 구한다.
        - 마지막 축으로 더하기: 텐서.sum(-1)
        - 기울기가 흘러야 하므로 .item()이나 .detach()를 쓰지 않는다.
        - torch.distributions.Normal을 쓰지 않고 수식을 직접 쓰는 것을 권장한다.

    채점: pytest ppo/test_ppo.py -k evaluate
    """
    raise NotImplementedError("빈칸 다-1: ppo.py의 evaluate_action을 채워주세요.")


def sample_action(mean, std):
    """[빈칸 다-2] 정규분포 N(mean, std)에서 행동을 뽑고, 그 행동의 로그 확률을 함께 돌려준다.

    신경망이 내놓은 평균 그대로 행동하면 항상 같은 행동만 하게 된다.
    그래서 평균 주변에서 무작위로 행동을 뽑아, 새로운 행동도 시도해 보게 한다.

    입력
        mean : (N, A)            정책 신경망이 내놓은 행동 평균
        std  : (A,) 또는 (N, A)  행동의 표준편차
    출력
        action   : (N, A)  뽑은 행동
        log_prob : (N,)    뽑은 행동의 로그 확률

    힌트
        - 표준정규분포 난수는 torch.randn_like(mean)으로 만든다.
        - 행동 = 평균 + 표준편차 × 난수
        - 로그 확률은 위에서 만든 evaluate_action을 불러서 구한다.

    채점: pytest ppo/test_ppo.py -k sample
    """
    raise NotImplementedError("빈칸 다-2: ppo.py의 sample_action을 채워주세요.")


def compute_gae(rewards, values, dones, last_values, gamma, lam):
    """[빈칸 라] 걸음마다 "그때 한 행동이 평소보다 얼마나 좋았나"(이득)를 계산한다. (GAE)

    가치 신경망은 "이 상황이면 앞으로 이만큼 점수를 받겠다"(가치)를 예측한다.
    실제로 받은 점수가 그 예측보다 높았다면 그때 한 행동이 평소보다 좋았던 것이다.
    이 차이를 여러 걸음에 걸쳐 섞어서 계산한 값이 이득이다.

    입력
        rewards     : (T, N)  걸음마다 받은 점수
        values      : (T, N)  걸음마다 가치 신경망이 예측한 가치
        dones       : (T, N)  그 걸음에서 판이 끝났으면 1.0, 아니면 0.0
        last_values : (N,)    마지막 걸음 다음 상태의 가치
        gamma       : 할인율
        lam         : GAE의 λ
    출력
        advantages : (T, N)  이득
        returns    : (T, N)  가치 신경망이 맞혀야 할 목표값 (= advantages + values)

    수식 (마지막 걸음 t = T-1부터 0까지 거꾸로 계산한다)
        다음 가치 = last_values (t가 마지막 걸음일 때) 또는 values[t + 1]
        안 끝남   = 1 - dones[t]
        오차[t]   = rewards[t] + gamma * 다음 가치 * 안 끝남 - values[t]
        이득[t]   = 오차[t] + gamma * lam * 안 끝남 * 이득[t + 1]     (이득[T]는 0)

    힌트
        - 결과를 담을 텐서는 torch.zeros_like(rewards)로 만든다.
        - 거꾸로 도는 반복문: for t in reversed(range(T))
        - 판이 끝난 걸음에서는 "안 끝남"이 0이 되어, 다음 판의 값이 섞이지 않는다.
        - 입력으로 받은 텐서의 값을 바꾸지 않는다.

    채점: pytest ppo/test_ppo.py -k gae
    """
    raise NotImplementedError("빈칸 라: ppo.py의 compute_gae를 채워주세요.")


def ppo_loss(new_log_prob, old_log_prob, advantages, new_values, returns, entropy,
             clip_param, value_coef, entropy_coef):
    """[빈칸 마] 신경망을 어느 방향으로 얼마나 수정할지 정하는 손실을 계산한다.

    이득이 높았던 행동은 확률을 올리고, 낮았던 행동은 확률을 내리는 것이 목표이다.
    다만 한 번에 너무 많이 바꾸면 학습이 망가지므로, 확률이 바뀐 비율을 1±ε 범위로 자른다.
    이 "자르기"가 PPO의 핵심이다.

    입력 (모두 미니배치 크기 M의 1차원 텐서)
        new_log_prob : (M,)  지금 신경망 기준의 로그 확률 (기울기가 흐른다)
        old_log_prob : (M,)  경험을 모을 때의 로그 확률
        advantages   : (M,)  이득 (이미 정규화된 값)
        new_values   : (M,)  지금 가치 신경망의 예측 (기울기가 흐른다)
        returns      : (M,)  가치 신경망이 맞혀야 할 목표값
        entropy      : (M,)  지금 분포의 엔트로피 (기울기가 흐른다)
        clip_param, value_coef, entropy_coef : 숫자
    출력
        loss : 숫자 하나 (0차원 텐서, 기울기가 흐른다)
        info : dict. 값은 모두 .item()으로 바꾼 파이썬 숫자
            "policy_loss", "value_loss", "entropy", "approx_kl", "clip_fraction"

    수식
        비율      = exp(new_log_prob - old_log_prob)
        정책 손실 = -평균( min(비율 * 이득, clamp(비율, 1-ε, 1+ε) * 이득) )
        가치 손실 = 평균( (new_values - returns)^2 )
        loss      = 정책 손실 + value_coef * 가치 손실 - entropy_coef * 평균(entropy)

        학습 상태를 확인하는 지표 (기울기가 필요 없으므로 with torch.no_grad(): 안에서 계산)
        approx_kl     = 평균( (비율 - 1) - log(비율) )     신경망이 바뀐 정도
        clip_fraction = 평균( |비율 - 1| > ε )            잘린 경험의 비율

    힌트
        - 범위 자르기: torch.clamp(비율, 1 - clip_param, 1 + clip_param)
        - 두 텐서에서 원소마다 작은 쪽 고르기: torch.min(a, b)
        - 참/거짓 텐서의 평균: (조건).float().mean()

    채점: pytest ppo/test_ppo.py -k loss
    """
    raise NotImplementedError("빈칸 마: ppo.py의 ppo_loss를 채워주세요.")


def ppo_update(actor, critic, optimizer, batch, cfg):
    """[빈칸 바] 모은 경험을 작은 묶음으로 나누어, 신경망을 여러 번 수정한다.

    경험 B개를 한꺼번에 쓰지 않고, 순서를 섞어 미니배치로 나눈 뒤 미니배치마다 한 번씩 수정한다.
    이 과정을 num_epochs번 반복한다.

    입력
        actor     : 정책 신경망.  actor(obs) -> 행동 평균 (M, A),  actor.std -> 표준편차 (A,)
        critic    : 가치 신경망.  critic(obs) -> 가치 (M,)
        optimizer : actor와 critic의 값을 함께 수정하는 최적화 도구
        batch     : dict. 모두 앞 축이 B인 텐서
            "obs" (B, 관찰값 개수), "actions" (B, A), "log_probs" (B,), "advantages" (B,), "returns" (B,)
        cfg       : PPOConfig
    출력
        info : dict. ppo_loss가 돌려준 info 5개 값을 모든 미니배치에 대해 평균 낸 것 (파이썬 숫자)

    할 일
        1. 이득을 배치 전체 기준으로 정규화한다: (이득 - 평균) / (표준편차 + 1e-8)
        2. num_epochs번 반복한다.
            2-1. torch.randperm(B, device=obs.device)로 순서를 섞는다.
            2-2. 섞은 순서를 num_mini_batches개로 나누고, 미니배치마다 아래를 수행한다.
                - actor와 critic을 미니배치의 obs로 한 번씩 부른다.
                - evaluate_action으로 새 로그 확률과 엔트로피를 구한다.
                - ppo_loss로 손실을 구한다.
                - optimizer.zero_grad() -> loss.backward()
                - torch.nn.utils.clip_grad_norm_(값 목록, cfg.max_grad_norm)으로 기울기 크기를 자른다.
                  (값 목록 = actor와 critic의 parameters()를 합친 것)
                - optimizer.step()
        3. 미니배치마다 받은 info를 평균 내서 돌려준다.

    힌트
        - B는 항상 num_mini_batches로 나누어떨어진다. 미니배치 크기 = B // cfg.num_mini_batches
        - 미니배치 고르기: 순서[i * 크기 : (i + 1) * 크기]를 인덱스로 써서 batch["obs"][인덱스]
        - 1e-8을 더하는 이유: 모든 이득이 같으면 표준편차가 0이 되어 0으로 나누게 된다.

    채점: pytest ppo/test_ppo.py -k update
    """
    raise NotImplementedError("빈칸 바: ppo.py의 ppo_update를 채워주세요.")
```

- [ ] **Step 3: 빈칸 상태에서 채점기가 깔끔하게 실패하는지 확인**

Run: `cd /home/tmdwodl/바탕화면/ISAACSIM/code && pytest ppo/test_ppo.py -q 2>&1 | tail -5`
Expected: 검사 31개가 모두 실패하고(`31 failed`), 실패 이유가 전부 `NotImplementedError: 빈칸 ...`이다. `ImportError`나 `SyntaxError`가 하나라도 있으면 채점기나 빈칸 코드에 문제가 있는 것이므로 고친다.

확인 명령: `pytest ppo/test_ppo.py -q 2>&1 | grep -c "NotImplementedError: 빈칸"` (31 이상이 나와야 한다)

- [ ] **Step 4: 정답 코드 작성 (프로젝트 밖)**

`$S/answer/ppo.py`에 `ppo/ppo.py`와 같은 틀(`PPOConfig`와 함수 5개의 이름·입력·출력)로 정답을 작성한다. 정답은 빈칸 설명에 적힌 수식과 "할 일"을 그대로 구현한 것이다. 정답 내용은 이 문서에 적지 않는다.

```bash
S=/tmp/claude-1000/-home-tmdwodl------ISAACSIM-code/ee6c8af5-ce31-4af3-8d62-13daaa53396a/scratchpad
mkdir -p $S/answer
# $S/answer/ppo.py 를 작성한다
```

- [ ] **Step 5: 정답 코드로 채점기가 전부 통과하는지 확인**

프로젝트 폴더를 임시 폴더로 복사하고, 복사본의 `ppo.py`만 정답으로 바꿔서 채점한다. 프로젝트 폴더의 파일은 건드리지 않는다.

```bash
S=/tmp/claude-1000/-home-tmdwodl------ISAACSIM-code/ee6c8af5-ce31-4af3-8d62-13daaa53396a/scratchpad
rm -rf $S/verify && mkdir -p $S/verify/ppo
cp /home/tmdwodl/바탕화면/ISAACSIM/code/ppo/*.py $S/verify/ppo/
cp $S/answer/ppo.py $S/verify/ppo/ppo.py
cd $S/verify && pytest ppo/test_ppo.py ppo/test_model.py -q
```

Expected: `39 passed` (채점기 31개 + `model.py` 검사 8개). 실패하는 검사가 있으면 정답과 채점기 중 어느 쪽이 틀렸는지 수식으로 따져서 고친다. `test_update_moves_policy_toward_better_actions`의 기준값(0.3)이 정답 코드로도 아슬아슬하면, 정답 코드가 실제로 움직인 양의 절반 정도로 기준값을 낮춘다.

- [ ] **Step 6: 일부러 틀린 구현을 채점기가 잡아내는지 확인**

채점기가 흔한 실수를 놓치지 않는지 확인한다. `$S/verify/ppo/ppo.py`를 한 군데씩 바꿔서 해당 검사가 실패하는지 보고, 확인한 뒤 정답으로 되돌린다.

| 일부러 넣는 실수 | 실패해야 하는 검사 |
|---|---|
| `compute_gae`에서 "안 끝남"을 곱하지 않음 | `test_gae_done_blocks_future_reward`, `test_gae_done_blocks_next_value` |
| `ppo_loss`에서 `torch.min` 대신 자른 쪽만 사용 | `test_loss_no_clip_when_change_was_harmful` |
| `ppo_loss`에서 엔트로피를 더함 (빼야 함) | `test_loss_entropy_term_lowers_loss` |
| `ppo_update`에서 순서를 섞지 않음 | `test_update_uses_all_data_once_per_epoch` |
| `ppo_update`에서 이득을 정규화하지 않음 | `test_update_normalizes_advantages` |
| `ppo_update`에서 기울기 크기를 자르지 않음 | `test_update_clips_gradient_norm` |

Expected: 실수 6가지 모두에서 표에 적힌 검사가 실패한다. 잡아내지 못하는 실수가 있으면 채점기를 보강한다.

마지막에 `cp $S/answer/ppo.py $S/verify/ppo/ppo.py`로 되돌리고 `pytest ppo/test_ppo.py -q`가 다시 전부 통과하는지 확인한다.

- [ ] **Step 7: 커밋 (사용자가 허락한 경우에만)**

```bash
cd /home/tmdwodl/바탕화면/ISAACSIM && git add code/ppo/ppo.py code/ppo/test_ppo.py
git commit -m "feat(ppo): add PPO skeleton with blanks and grading tests"
```

---

### Task 3: 학습 반복문 (`train.py`)

**Files:**
- Create: `ppo/train.py`

**Interfaces:**
- Consumes: `model.Actor`, `model.Critic`, `model.RolloutStorage` (Task 1), `ppo.PPOConfig`, `ppo.sample_action`, `ppo.compute_gae`, `ppo.ppo_update` (Task 2)
- Produces:
  - 실행 명령: `python ppo/train.py --seed <정수> [--max_iterations <정수>]`
  - 기록 폴더: `<code>/logs/my_ppo/cartpole/<YYYY-MM-DD_HH-MM-SS>_seed<번호>/` (tensorboard 기록 파일, `model_<반복 번호>.pt`)
  - tensorboard 기록 이름: `Train/mean_reward`, `Train/mean_episode_length`, `Loss/value`, `Loss/surrogate`, `Loss/entropy`, `Loss/approx_kl`, `Loss/clip_fraction`, `Loss/learning_rate`, `Policy/mean_std`, `Perf/total_fps` (가로축은 반복 번호 0부터)

- [ ] **Step 1: `train.py` 작성**

`ppo/train.py`:

```python
"""직접 구현한 PPO로 Isaac Lab 수레-막대(Cartpole)를 학습시킨다. (준비된 코드: 읽고 이해하면 된다)

실행
    python ppo/train.py --seed 1

전체 흐름 (150번 반복)
    1) 경험 모으기 : 환경 4096개를 16걸음씩 진행하며 관찰값·행동·점수를 저장소에 담는다.
    2) 이득 계산   : compute_gae  (빈칸 라)
    3) 신경망 수정 : ppo_update   (빈칸 바)
    4) 기록        : 평균 총점 등을 화면과 tensorboard에 남긴다.
"""
import argparse
import statistics
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import gymnasium as gym
import torch
from torch.utils.tensorboard import SummaryWriter

from isaaclab.app import add_launcher_args, launch_simulation

import isaaclab_tasks  # noqa: F401  (환경 이름을 등록하려면 불러와야 한다)
from isaaclab_tasks.utils import resolve_task_config, setup_preset_cli

from model import Actor, Critic, RolloutStorage
from ppo import PPOConfig, compute_gae, ppo_update, sample_action

TASK = "Isaac-Cartpole-Direct"
NUM_STEPS = 16       # 한 번에 모으는 걸음 수 T
SAVE_INTERVAL = 50   # 신경망을 저장하는 간격 (반복 횟수)
# 어느 폴더에서 실행하든 기록은 항상 code/logs/my_ppo/cartpole/ 아래에 남긴다.
LOG_ROOT = Path(__file__).resolve().parent.parent / "logs" / "my_ppo" / "cartpole"


def parse_args():
    parser = argparse.ArgumentParser(description="직접 구현한 PPO로 Cartpole을 학습시킨다.")
    parser.add_argument("--seed", type=int, default=1, help="씨앗값 (무작위 숫자의 출발 번호)")
    parser.add_argument("--max_iterations", type=int, default=150, help="반복 횟수")
    add_launcher_args(parser)  # Isaac Lab이 쓰는 옵션(--device 등)을 추가한다
    argv = sys.argv[1:]
    # 비교 기준(rsl_rl) 실험과 같은 물리 엔진을 쓴다.
    if not any(arg.startswith("physics=") for arg in argv):
        argv.append("physics=isaacsim_physx")
    args, hydra_args = setup_preset_cli(parser, argv)
    sys.argv = [sys.argv[0]] + hydra_args  # 남은 옵션은 Isaac Lab의 설정 도구(Hydra)가 읽는다
    return args


def main():
    args = parse_args()
    cfg = PPOConfig()
    env_cfg, _ = resolve_task_config(TASK, "")
    env_cfg.seed = args.seed

    # 이 구간을 벗어나면 프로그램이 바로 끝난다. 저장과 기록은 모두 이 안에서 한다.
    with launch_simulation(env_cfg, args):
        env = gym.make(TASK, cfg=env_cfg)
        base = env.unwrapped
        num_envs, device = base.num_envs, base.device
        obs_dim = base.single_observation_space["policy"].shape[0]
        act_dim = base.single_action_space.shape[0]

        torch.manual_seed(args.seed)
        actor = Actor(obs_dim, act_dim).to(device)
        critic = Critic(obs_dim).to(device)
        optimizer = torch.optim.Adam(
            list(actor.parameters()) + list(critic.parameters()), lr=cfg.learning_rate
        )
        storage = RolloutStorage(NUM_STEPS, num_envs, obs_dim, act_dim, device)

        log_dir = LOG_ROOT / f"{datetime.now():%Y-%m-%d_%H-%M-%S}_seed{args.seed}"
        writer = SummaryWriter(str(log_dir))
        print(f"[기록 폴더] {log_dir}")

        # 판마다 받은 총점과 길이를 세는 도구. rsl_rl과 똑같이 "최근 끝난 100판"의 평균을 본다.
        episode_return = torch.zeros(num_envs, device=device)
        episode_length = torch.zeros(num_envs, device=device)
        recent_returns, recent_lengths = deque(maxlen=100), deque(maxlen=100)

        obs = env.reset()[0]["policy"]  # (N, obs_dim)
        for it in range(args.max_iterations):
            start = time.time()

            # ---------- 1) 경험 모으기 (기울기 계산 없음) ----------
            with torch.no_grad():
                for _ in range(NUM_STEPS):
                    action, log_prob = sample_action(actor(obs), actor.std)  # 빈칸 다
                    value = critic(obs)
                    next_obs, reward, terminated, truncated, _ = env.step(action)

                    # [시간 초과 보정] 수레-막대는 300걸음이 지나면 막대가 서 있어도 판을 끝낸다(truncated).
                    # 이것은 실패가 아니라 시간이 다 된 것이므로, 미래 점수를 0으로 보면 안 된다.
                    # rsl_rl과 같은 방식으로, 시간 초과 걸음의 점수에 gamma * V(지금 상태)를 더해 보정한다.
                    learn_reward = reward + cfg.gamma * value * truncated.float()
                    done = terminated | truncated  # 실패했거나 시간이 다 됐으면 판이 끝난 것
                    storage.add(obs, action, log_prob, learn_reward, done, value)

                    # 통계는 보정하지 않은 원래 점수로 센다.
                    episode_return += reward
                    episode_length += 1
                    finished = done.nonzero().squeeze(-1)
                    recent_returns.extend(episode_return[finished].tolist())
                    recent_lengths.extend(episode_length[finished].tolist())
                    episode_return[finished] = 0
                    episode_length[finished] = 0

                    obs = next_obs["policy"]  # 판이 끝난 환경은 Isaac Lab이 알아서 새 판을 시작해 준다

                # ---------- 2) 이득 계산 ----------
                advantages, returns = compute_gae(  # 빈칸 라
                    storage.rewards, storage.values, storage.dones, critic(obs), cfg.gamma, cfg.lam
                )

            # ---------- 3) 신경망 수정 ----------
            info = ppo_update(actor, critic, optimizer, storage.flatten(advantages, returns), cfg)  # 빈칸 바
            storage.clear()

            # ---------- 4) 기록 ----------
            fps = NUM_STEPS * num_envs / (time.time() - start)
            mean_std = actor.std.mean().item()
            writer.add_scalar("Loss/value", info["value_loss"], it)
            writer.add_scalar("Loss/surrogate", info["policy_loss"], it)
            writer.add_scalar("Loss/entropy", info["entropy"], it)
            writer.add_scalar("Loss/approx_kl", info["approx_kl"], it)
            writer.add_scalar("Loss/clip_fraction", info["clip_fraction"], it)
            writer.add_scalar("Loss/learning_rate", cfg.learning_rate, it)
            writer.add_scalar("Policy/mean_std", mean_std, it)
            writer.add_scalar("Perf/total_fps", fps, it)
            line = f"[{it + 1:3d}/{args.max_iterations}]"
            if recent_returns:
                mean_return = statistics.mean(recent_returns)
                mean_length = statistics.mean(recent_lengths)
                writer.add_scalar("Train/mean_reward", mean_return, it)
                writer.add_scalar("Train/mean_episode_length", mean_length, it)
                line += f" 평균 총점 {mean_return:6.2f} | 평균 판 길이 {mean_length:6.1f} |"
            line += (
                f" 표준편차 {mean_std:.3f} | KL {info['approx_kl']:.4f}"
                f" | 잘린 비율 {info['clip_fraction']:.2f} | 초당 {fps:,.0f}걸음"
            )
            print(line)

            if (it + 1) % SAVE_INTERVAL == 0 or it + 1 == args.max_iterations:
                torch.save(
                    {"actor": actor.state_dict(), "critic": critic.state_dict()},
                    log_dir / f"model_{it + 1}.pt",
                )

        if recent_returns:
            mean_return = statistics.mean(recent_returns)
            mean_length = statistics.mean(recent_lengths)
            verdict = "통과" if mean_return >= 4.5 and mean_length >= 290 else "미달"
            print(f"[최종] 평균 총점 {mean_return:.2f} (기준 4.5) | 평균 판 길이 {mean_length:.1f} (기준 290) -> {verdict}")
        writer.close()
        env.close()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 정답 코드로 짧은 시험 실행 (2번 반복)**

복사본을 새로 만들고(방금 만든 `train.py` 포함), 일부러 `ppo/` 폴더 안에서 실행한다. 이렇게 하면 다른 폴더에서 실행해도 기록 위치가 올바른지(Review Focus 5) 함께 확인된다.

```bash
S=/tmp/claude-1000/-home-tmdwodl------ISAACSIM-code/ee6c8af5-ce31-4af3-8d62-13daaa53396a/scratchpad
cp /home/tmdwodl/바탕화면/ISAACSIM/code/ppo/*.py $S/verify/ppo/ && cp $S/answer/ppo.py $S/verify/ppo/ppo.py
source ~/miniconda3/etc/profile.d/conda.sh && conda activate env_isaaclab && export OMNI_KIT_ACCEPT_EULA=yes
cd $S/verify/ppo && python train.py --seed 1 --max_iterations 2 > $S/verify/smoke.txt 2>&1
grep -E "기록 폴더|^\[ *[0-9]+/2\]|최종|Traceback|Error" $S/verify/smoke.txt
find $S/verify/logs -type f | sort
```

Expected:
- `[기록 폴더] .../verify/logs/my_ppo/cartpole/<시간>_seed1`이 출력된다. 기록 폴더가 `verify/ppo/logs`가 아니라 `verify/logs` 아래에 있어야 한다.
- `[  1/2]`, `[  2/2]` 두 줄이 출력되고 `Traceback`이 없다.
- 기록 폴더 안에 `events.out.tfevents...` 파일과 `model_2.pt`가 있다 (Review Focus 3).

- [ ] **Step 3: 빈칸 상태에서 실행하면 어떤 안내가 나오는지 확인**

사용자가 빈칸을 채우기 전에 `train.py`를 실행했을 때, 무엇을 해야 하는지 알 수 있는 문구가 나오는지 확인한다.

```bash
cd /home/tmdwodl/바탕화면/ISAACSIM/code && python ppo/train.py --seed 1 --max_iterations 1 2>&1 | grep -E "NotImplementedError" | tail -1
```

Expected: `NotImplementedError: 빈칸 다-2: ppo.py의 sample_action을 채워주세요.`

확인 후 이 실행으로 생긴 빈 기록 폴더를 지운다: `rm -rf /home/tmdwodl/바탕화면/ISAACSIM/code/logs/my_ppo`

- [ ] **Step 4: 커밋 (사용자가 허락한 경우에만)**

```bash
cd /home/tmdwodl/바탕화면/ISAACSIM && git add code/ppo/train.py
git commit -m "feat(ppo): add Cartpole training loop for the hand-written PPO"
```

---

### Task 4: 비교 그래프 (`plot_compare.py`)와 `rsl_rl` 기준 실험

**Files:**
- Create: `ppo/plot_compare.py`
- Test: `ppo/test_plot_compare.py`
- 생성되는 기록: `logs/rsl_rl/cartpole_direct/<시간>_seed1`, `_seed2`, `_seed3`

**Interfaces:**
- Consumes: Task 3이 남기는 기록 폴더 이름 규칙(`<시간>_seed<번호>`)과 기록 이름(`Train/mean_reward`, `Train/mean_episode_length`)
- Produces:
  - `find_runs(root) -> dict[int, Path]`: 씨앗값마다 가장 최근 기록 폴더
  - `load_curve(run_dir, tag) -> (steps: list[int], values: list[float])`
  - `passed(final_reward: float, final_length: float) -> bool`
  - 실행 명령: `python ppo/plot_compare.py` → `docs/images/stage1_cartpole_compare.png` 저장, 씨앗값별 최종값과 통과 여부를 표로 출력. 직접 구현한 PPO의 씨앗값 1, 2, 3이 모두 통과하면 종료 코드 0, 아니면 1

- [ ] **Step 1: 실패하는 검사 작성**

`ppo/test_plot_compare.py`:

```python
"""plot_compare.py 검사.

실행: pytest ppo/test_plot_compare.py
"""
from plot_compare import find_runs, passed


def test_find_runs_picks_latest_per_seed(tmp_path):
    for name in ("2026-10-10_09-00-00_seed1", "2026-10-10_10-00-00_seed1", "2026-10-10_09-30-00_seed2",
                 "2026-10-09_19-45-58", "notes_seedling"):
        (tmp_path / name).mkdir()
    (tmp_path / "2026-10-10_11-00-00_seed3").write_text("폴더가 아닌 파일")
    runs = find_runs(tmp_path)
    assert set(runs) == {1, 2}, "이름이 '_seed<번호>'로 끝나는 폴더만 골라야 합니다."
    assert runs[1].name == "2026-10-10_10-00-00_seed1", "같은 씨앗값이 여러 개면 가장 최근 것을 골라야 합니다."


def test_find_runs_missing_root(tmp_path):
    assert find_runs(tmp_path / "없는_폴더") == {}


def test_passed_thresholds():
    assert passed(4.5, 290.0)
    assert passed(4.92, 300.0)
    assert not passed(4.49, 300.0)
    assert not passed(4.92, 289.9)
```

- [ ] **Step 2: 검사가 실패하는지 확인**

Run: `cd /home/tmdwodl/바탕화면/ISAACSIM/code && pytest ppo/test_plot_compare.py -q`
Expected: `ModuleNotFoundError: No module named 'plot_compare'`로 실패한다.

- [ ] **Step 3: `plot_compare.py` 작성**

그래프 설계 (dataviz 스킬 기준)
- 평균 총점과 평균 판 길이는 눈금이 다르므로 세로축을 둘로 나누지 않고 **그래프 2개를 위아래로** 놓는다.
- 색은 구현 방식(직접 구현한 PPO, `rsl_rl`)에 고정한다. 기본 팔레트의 1번(파랑 `#2a78d6`)과 2번(주황 `#eb6834`)을 순서대로 쓴다. 씨앗값이 달라도 같은 구현이면 같은 색이다.
- 씨앗값별 곡선은 가늘고 연하게(1px), 씨앗값 평균은 2px로 그린다.
- 두 곡선이 끝에서 거의 겹치므로 선 끝에 숫자를 붙이지 않고, 범례에 최종값을 함께 적는다. 글자는 선 색이 아니라 글자색을 쓴다.
- 눈금선은 가는 실선의 연한 회색, 통과 기준은 가는 회색 선과 글자로 표시한다.
- 이 PC에는 `node`가 없어서 dataviz 스킬의 색 검증 스크립트를 실행할 수 없다. 대신 스킬이 이미 검증해 둔 기본 팔레트의 1번, 2번 색을 그대로 쓴다.

`ppo/plot_compare.py`:

```python
"""직접 구현한 PPO와 rsl_rl의 학습 곡선을 겹쳐 그리고, 통과 여부를 판정한다.

실행
    python ppo/plot_compare.py

읽는 기록
    logs/my_ppo/cartpole/<시간>_seed<번호>/          (직접 구현한 PPO)
    logs/rsl_rl/cartpole_direct/<시간>_seed<번호>/   (비교 기준)
결과
    docs/images/stage1_cartpole_compare.png
"""
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 창을 띄우지 않고 파일로만 저장한다
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "images" / "stage1_cartpole_compare.png"
MINE = "직접 구현한 PPO"
SOURCES = {  # 이름: (기록 폴더, 선 색)
    MINE: (ROOT / "logs" / "my_ppo" / "cartpole", "#2a78d6"),
    "rsl_rl": (ROOT / "logs" / "rsl_rl" / "cartpole_direct", "#eb6834"),
}
PANELS = [  # (기록 이름, 세로축 제목, 통과 기준)
    ("Train/mean_reward", "평균 총점", 4.5),
    ("Train/mean_episode_length", "평균 판 길이", 290.0),
]
SEEDS = (1, 2, 3)
TEXT, TEXT_SUB, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"


def find_runs(root):
    """root 아래에서 이름이 '_seed<번호>'로 끝나는 폴더를 찾아 {씨앗값: 가장 최근 폴더}로 돌려준다."""
    runs = {}
    for path in sorted(Path(root).glob("*_seed*")):  # 이름이 시간으로 시작하므로 정렬하면 최근 것이 뒤에 온다
        match = re.search(r"_seed(\d+)$", path.name)
        if match and path.is_dir():
            runs[int(match.group(1))] = path
    return runs


def load_curve(run_dir, tag):
    """tensorboard 기록에서 tag의 (반복 번호 목록, 값 목록)을 읽는다. 없으면 빈 목록을 돌려준다."""
    events = EventAccumulator(str(run_dir))
    events.Reload()
    if tag not in events.Tags()["scalars"]:
        return [], []
    scalars = events.Scalars(tag)
    return [s.step for s in scalars], [s.value for s in scalars]


def passed(final_reward, final_length):
    """통과 기준: 평균 총점 4.5 이상, 평균 판 길이 290 이상."""
    return final_reward >= 4.5 and final_length >= 290.0


def main():
    # curves[이름][씨앗값][기록 이름] = (반복 번호 목록, 값 목록)
    curves = {
        name: {seed: {tag: load_curve(run, tag) for tag, _, _ in PANELS} for seed, run in find_runs(root).items()}
        for name, (root, _) in SOURCES.items()
    }
    if not curves[MINE]:
        print(f"직접 구현한 PPO의 학습 기록이 없습니다: {SOURCES[MINE][0]}")
        print("먼저 python ppo/train.py --seed 1 을 실행하세요.")
        sys.exit(1)

    plt.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]  # 한글이 들어 있는 글꼴
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True, facecolor=SURFACE)
    for ax, (tag, ylabel, threshold) in zip(axes, PANELS):
        ax.set_facecolor(SURFACE)
        ax.axhline(threshold, color="#b5b4ad", linewidth=1)
        ax.text(0.5, threshold, f" 통과 기준 {threshold:g}", color=TEXT_SUB, fontsize=9, va="bottom")
        for name, (_, color) in SOURCES.items():
            seed_curves = [c[tag] for c in curves[name].values() if c[tag][1]]
            if not seed_curves:
                continue
            for steps, values in seed_curves:
                ax.plot(steps, values, color=color, linewidth=1, alpha=0.4)
            length = min(len(values) for _, values in seed_curves)  # 길이가 다르면 짧은 쪽에 맞춘다
            mean = [sum(values[i] for _, values in seed_curves) / len(seed_curves) for i in range(length)]
            ax.plot(seed_curves[0][0][:length], mean, color=color, linewidth=2,
                    solid_capstyle="round", solid_joinstyle="round",
                    label=f"{name} (씨앗값 {len(seed_curves)}개 평균, 최종 {mean[-1]:.2f})")
        ax.set_ylabel(ylabel, color=TEXT)
        ax.grid(True, color=GRID, linewidth=1)
        ax.set_axisbelow(True)
        ax.tick_params(colors=TEXT_SUB)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.legend(frameon=False, labelcolor=TEXT, loc="lower right")
    axes[0].set_title("수레-막대 학습 곡선 비교 (연한 선은 씨앗값별, 굵은 선은 평균)", color=TEXT, loc="left")
    axes[1].set_xlabel("반복 횟수", color=TEXT)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT, dpi=150, facecolor=SURFACE)
    print(f"[그래프 저장] {OUT}")

    # 씨앗값별 최종값과 통과 여부
    reward_tag, length_tag = PANELS[0][0], PANELS[1][0]
    print(f"\n{'구현':<16}{'씨앗값':>6}{'평균 총점':>10}{'평균 판 길이':>12}  판정")
    all_passed = True
    for name in SOURCES:
        for seed in SEEDS:
            run = curves[name].get(seed)
            if run is None or not run[reward_tag][1] or not run[length_tag][1]:
                print(f"{name:<16}{seed:>6}{'기록 없음':>10}")
                all_passed = all_passed and name != MINE
                continue
            reward, length = run[reward_tag][1][-1], run[length_tag][1][-1]
            ok = passed(reward, length)
            print(f"{name:<16}{seed:>6}{reward:>10.2f}{length:>12.1f}  {'통과' if ok else '미달'}")
            if name == MINE:
                all_passed = all_passed and ok
    print(f"\n직접 구현한 PPO: {'씨앗값 3개 모두 통과' if all_passed else '아직 통과하지 못함'}")
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 검사가 통과하는지 확인**

Run: `cd /home/tmdwodl/바탕화면/ISAACSIM/code && pytest ppo/test_plot_compare.py -q`
Expected: `3 passed`

- [ ] **Step 5: 기록이 없을 때 안내 문구가 나오는지 확인 (Review Focus 4)**

Run: `cd /home/tmdwodl/바탕화면/ISAACSIM/code && python ppo/plot_compare.py; echo "종료 코드 $?"`
Expected: `직접 구현한 PPO의 학습 기록이 없습니다: ...`와 `먼저 python ppo/train.py --seed 1 을 실행하세요.`가 출력되고, `Traceback` 없이 `종료 코드 1`이 나온다.

- [ ] **Step 6: `rsl_rl` 기준 실험을 씨앗값 3개로 실행**

```bash
cd /home/tmdwodl/바탕화면/ISAACSIM/code
source ~/miniconda3/etc/profile.d/conda.sh && conda activate env_isaaclab && export OMNI_KIT_ACCEPT_EULA=yes
for seed in 1 2 3; do
  isaaclab train --rl_library rsl_rl --task Isaac-Cartpole-Direct --seed $seed --run_name seed$seed physics=isaacsim_physx \
    > logs/cartpole_rsl_rl_seed$seed.txt 2>&1
  echo "seed $seed 종료 코드 $?"
done
ls logs/rsl_rl/cartpole_direct/
```

Expected: 종료 코드가 3번 모두 0이고, `logs/rsl_rl/cartpole_direct/` 아래에 이름이 `_seed1`, `_seed2`, `_seed3`으로 끝나는 폴더가 하나씩 생긴다. 이름 끝에 `_seed<번호>`가 붙지 않았다면 `--run_name` 옵션이 적용되지 않은 것이므로, `isaaclab train --rl_library rsl_rl --help`에서 옵션 이름을 다시 확인한다.

- [ ] **Step 7: 기준 실험 3개의 최종값 확인**

```bash
cd /home/tmdwodl/바탕화면/ISAACSIM/code && python - <<'EOF'
from pathlib import Path
import sys
sys.path.insert(0, "ppo")
from plot_compare import find_runs, load_curve, passed
for seed, run in sorted(find_runs(Path("logs/rsl_rl/cartpole_direct")).items()):
    reward = load_curve(run, "Train/mean_reward")[1][-1]
    length = load_curve(run, "Train/mean_episode_length")[1][-1]
    print(seed, run.name, round(reward, 3), round(length, 1), "통과" if passed(reward, length) else "미달")
EOF
```

Expected: 씨앗값 3개 모두 평균 총점이 4.9 근처, 평균 판 길이가 300 근처이고 "통과"로 나온다. 기준 실험이 통과 기준을 넘지 못하는 씨앗값이 있으면 통과 기준 자체가 너무 높은 것이므로, 작업을 멈추고 사용자와 기준을 다시 상의한다.

- [ ] **Step 8: 커밋 (사용자가 허락한 경우에만)**

```bash
cd /home/tmdwodl/바탕화면/ISAACSIM && git add code/ppo/plot_compare.py code/ppo/test_plot_compare.py
git commit -m "feat(ppo): add learning-curve comparison plot against rsl_rl"
```

---

### Task 5: 정답 코드로 전체 사전 확인

**Files:**
- 프로젝트 파일은 만들거나 고치지 않는다. 모든 실행은 `$S/verify/`에서 한다.
- Modify: `DEVLOG.md` (7장 "학습 실험 기록"에 결과 추가)

**Interfaces:**
- Consumes: Task 1~4에서 만든 모든 파일, `$S/answer/ppo.py`
- Produces: "준비된 코드와 설정값으로 통과 기준에 도달할 수 있다"는 확인 결과 (DEVLOG에 기록)

- [ ] **Step 1: 복사본을 최신 상태로 만들고 전체 검사 실행**

```bash
S=/tmp/claude-1000/-home-tmdwodl------ISAACSIM-code/ee6c8af5-ce31-4af3-8d62-13daaa53396a/scratchpad
rm -rf $S/verify && mkdir -p $S/verify/ppo $S/verify/logs
cp /home/tmdwodl/바탕화면/ISAACSIM/code/ppo/*.py $S/verify/ppo/ && cp $S/answer/ppo.py $S/verify/ppo/ppo.py
ln -s /home/tmdwodl/바탕화면/ISAACSIM/code/logs/rsl_rl $S/verify/logs/rsl_rl   # 기준 실험 기록을 함께 읽기 위한 연결
cd $S/verify && pytest ppo -q
```

Expected: `42 passed` (채점기 31개 + `model.py` 8개 + `plot_compare.py` 3개)

- [ ] **Step 2: 정답 코드로 씨앗값 3개 학습**

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate env_isaaclab && export OMNI_KIT_ACCEPT_EULA=yes
cd $S/verify
for seed in 1 2 3; do
  python ppo/train.py --seed $seed > train_seed$seed.txt 2>&1
  grep -E "^\[최종\]|Traceback" train_seed$seed.txt
done
```

Expected: 씨앗값마다 `[최종] 평균 총점 4.xx (기준 4.5) | 평균 판 길이 3xx.x (기준 290) -> 통과`가 출력된다.

- [ ] **Step 3: 비교 그래프와 판정 확인**

```bash
cd $S/verify && python ppo/plot_compare.py; echo "종료 코드 $?"
```

Expected: 표에 직접 구현한 PPO와 `rsl_rl`의 씨앗값 1, 2, 3이 모두 "통과"로 나오고, `직접 구현한 PPO: 씨앗값 3개 모두 통과`, `종료 코드 0`이 출력된다. 저장된 그림 `$S/verify/docs/images/stage1_cartpole_compare.png`를 열어서 아래를 눈으로 확인한다.
- 한글이 깨지지 않고 보인다.
- 범례와 통과 기준 글자가 곡선과 겹쳐서 읽기 어렵지 않다.
- 두 구현의 굵은 선이 비슷한 모양으로 올라가서 비슷한 높이에서 평평해진다.

글자가 겹치면 범례 위치(`loc`)나 통과 기준 글자의 위치를 조정하고, 프로젝트의 `ppo/plot_compare.py`에도 같은 수정을 반영한다.

- [ ] **Step 4: 통과 기준에 못 미친 경우의 처리**

Step 2나 Step 3에서 "미달"인 씨앗값이 있으면 아래 순서로 원인을 좁힌다.
1. `train_seed<번호>.txt`에서 KL과 잘린 비율이 정상 범위(KL 0.01 안팎, 잘린 비율 0.1~0.3)인지 본다.
2. `train.py`의 경험 모으기 부분(시간 초과 보정, 판 끝남 처리)을 `rsl_rl`의 동작과 다시 비교한다.
3. 코드에 잘못이 없는데도 기준에 못 미치면, 설계 문서 7장에 따라 "학습 속도 자동 조절"을 추가하는 실험이 필요하다. 이것은 설계를 바꾸는 일이므로, **작업을 멈추고 사용자에게 결과를 보고한 뒤 상의한다.**

- [ ] **Step 5: 시간 초과 보정의 효과 확인 (설계 문서 10장)**

복사본의 `train.py`에서 보정을 끄고 씨앗값 1로 한 번 학습시켜, 보정이 있을 때와 비교한다. 프로젝트 파일은 바꾸지 않는다.

```bash
cd $S/verify && cp ppo/train.py ppo/train_backup.py
sed -i 's/learn_reward = reward + cfg.gamma \* value \* truncated.float()/learn_reward = reward/' ppo/train.py
grep -n "learn_reward = reward$" ppo/train.py   # 한 줄이 나와야 한다 (바뀌었는지 확인)
python ppo/train.py --seed 1 > train_seed1_no_bootstrap.txt 2>&1
grep -E "^\[최종\]" train_seed1_no_bootstrap.txt
mv ppo/train_backup.py ppo/train.py
# 보정을 끈 기록이 비교 그래프에 섞이지 않도록 다른 폴더로 옮긴다 (가장 최근에 생긴 seed1 폴더)
mkdir -p no_bootstrap && mv "$(ls -d logs/my_ppo/cartpole/*_seed1 | tail -1)" no_bootstrap/
ls logs/my_ppo/cartpole/   # seed1, seed2, seed3 폴더가 하나씩 남아 있어야 한다
```

Expected: 보정을 끈 실행의 `[최종]` 줄이 출력된다. 보정이 있을 때의 결과와 숫자를 나란히 적어 둔다. 어느 쪽이 높게 나오든 결과를 그대로 기록한다.

- [ ] **Step 6: 결과를 DEVLOG 7장에 기록**

`DEVLOG.md`의 "7. 학습 실험 기록" 표에 아래 형식으로 행을 추가한다. 숫자는 Step 2, 3, 5와 Task 4 Step 7에서 실제로 나온 값을 적는다.

```markdown
| 1 | 2026-10-10 | 1. 비교 기준 | `rsl_rl`을 씨앗값 1, 2, 3으로 다시 실행 (공정한 비교를 위해) | 씨앗값별 평균 총점 / 평균 판 길이: (실제 값) | (실제 시간) | 3개 모두 통과 기준을 넘는지 |
| 2 | 2026-10-10 | 1. 사전 확인 | 준비된 코드에 정답 구현을 넣어 씨앗값 1, 2, 3으로 실행 (정답 구현은 프로젝트에 넣지 않음) | 씨앗값별 평균 총점 / 평균 판 길이: (실제 값) | (실제 시간) | 준비된 코드와 설정값으로 통과 기준에 도달할 수 있음을 확인 |
| 3 | 2026-10-10 | 1. 사전 확인 | 실험 2에서 시간 초과 보정만 끔 (씨앗값 1) | (실제 값) | (실제 시간) | 보정이 있을 때와의 차이 |
```

"(실제 값)"과 "(실제 시간)"은 실행 결과로 채운다. 문서에 괄호 표시를 그대로 남기지 않는다.

- [ ] **Step 7: 커밋 (사용자가 허락한 경우에만)**

```bash
cd /home/tmdwodl/바탕화면/ISAACSIM && git add code/DEVLOG.md
git commit -m "docs: record stage 1 baseline and pre-check results"
```

---

### Task 6: 기록 정리와 사용자에게 넘기기

**Files:**
- Modify: `DEVLOG.md` (용어와 지표 설명 장 추가, 진행 상태와 일지 갱신)
- Modify: `docs/specs/2026-10-09-stage1-ppo-design.md` (상태 표시)

**Interfaces:**
- Consumes: Task 1~5의 결과
- Produces: 사용자가 빈칸 채우기를 바로 시작할 수 있는 상태의 프로젝트 폴더

- [ ] **Step 1: 프로젝트 폴더가 빈칸 상태인지 확인**

```bash
cd /home/tmdwodl/바탕화면/ISAACSIM/code
grep -c "raise NotImplementedError" ppo/ppo.py                                    # 5가 나와야 한다
pytest ppo/test_model.py ppo/test_plot_compare.py -q                              # 11 passed
pytest ppo/test_ppo.py -q 2>&1 | tail -1                                          # 31 failed
python - <<'CHECK'
import ast
tree = ast.parse(open("ppo/ppo.py", encoding="utf-8").read())
for node in tree.body:
    if isinstance(node, ast.FunctionDef):
        # 빈칸 함수의 본문은 "설명 + raise NotImplementedError" 두 줄뿐이어야 한다
        assert len(node.body) == 2 and isinstance(node.body[1], ast.Raise), f"{node.name}에 코드가 들어 있습니다"
        print(node.name, "-> 빈칸 상태")
CHECK
ls ppo/
```

Expected: 함수 5개가 모두 `빈칸 상태`로 출력된다. 즉 `ppo/ppo.py`에 정답 코드가 섞여 있지 않다. `ppo/` 폴더에는 `model.py`, `plot_compare.py`, `ppo.py`, `test_model.py`, `test_plot_compare.py`, `test_ppo.py`, `train.py`만 있다 (`__pycache__`, `.pytest_cache`는 무시).

- [ ] **Step 2: `.gitignore`에 캐시 폴더가 들어 있는지 확인**

Run: `cd /home/tmdwodl/바탕화면/ISAACSIM && git status --short`
Expected: `__pycache__`나 `.pytest_cache`가 목록에 보이지 않는다. 보이면 `.gitignore`에 해당 줄을 추가한다. `code/logs/`도 목록에 없어야 한다.

- [ ] **Step 3: DEVLOG에 "용어와 지표 설명" 장 추가**

`DEVLOG.md`의 8장 "최종 정리" 바로 앞에 새 장을 넣고, 기존 8장의 번호를 9로 바꾼다. 목차도 함께 고친다.

```markdown
## 8. 용어와 지표 설명

> 처음 보는 용어와, 학습이 잘 되고 있는지 볼 때 쓰는 지표를 정리한다. 새 용어가 나올 때마다 추가한다.

### 용어

| 용어 | 뜻 |
|---|---|
| PPO | 강화학습의 학습 방법 중 하나이다. "해본다 → 점수를 받는다 → 점수를 잘 받은 행동을 더 하도록 신경망을 조금 수정한다"를 반복하되, 한 번에 수정하는 양을 제한해서 학습이 망가지지 않게 한다. |
| 환경 | 로봇(또는 수레-막대)이 움직이는 시뮬레이션 세계이다. 행동을 넣으면 다음 관찰값과 점수를 돌려준다. |
| 관찰값 | 환경이 신경망에 알려주는 현재 상태이다. 수레-막대는 수레 위치, 막대 각도, 수레 속도, 막대 각속도 4개이다. |
| 판(에피소드) | 시작부터 실패하거나 시간이 다 될 때까지의 한 번의 시도이다. 수레-막대는 최대 300걸음이다. |
| 정책 신경망 | 관찰값을 보고 어떤 행동을 할지 정하는 신경망이다. |
| 가치 신경망 | 관찰값을 보고 "앞으로 점수를 얼마나 받을지"를 예측하는 신경망이다. |
| 이득 | 그때 한 행동이 평소보다 얼마나 좋았는지를 나타내는 숫자이다. 실제로 받은 점수가 가치 신경망의 예측보다 높으면 양수가 된다. |
| GAE | 이득을 계산하는 방법의 이름이다. 여러 걸음에 걸친 오차를 섞어서 계산한다. |
| 씨앗값 | 컴퓨터가 무작위 숫자를 만들 때 쓰는 출발 번호이다. 씨앗값이 같으면 결과가 똑같이 재현된다. 결과가 운이 아니라는 것을 보이려고 씨앗값 3개로 실험한다. |
| rsl_rl | 스위스 취리히 공대 로봇 연구실이 만든, 검증된 PPO 프로그램이다. Isaac Lab에 함께 들어 있다. 이 프로젝트에서는 직접 구현한 PPO와 비교하는 기준으로만 쓴다. |
| 시간 초과 보정 | 판이 실패가 아니라 시간이 다 되어 끝났을 때, 미래 점수를 0으로 보지 않도록 점수에 가치 예측을 더해 주는 처리이다. |

### 학습 상태 지표

학습을 실행하면 반복마다 아래 값이 화면과 tensorboard에 기록된다.

| 지표 | 뜻 | 정상일 때 | 이상 신호 |
|---|---|---|---|
| 평균 총점 | 최근 끝난 100판에서 받은 총점의 평균 | 오르다가 평평해진다 | 오르지 않거나, 오르다가 무너진다 |
| 평균 판 길이 | 최근 끝난 100판의 걸음 수 평균 (수레-막대는 최대 300) | 300에 가까워진다 | 낮은 값에 머문다 |
| 표준편차 | 행동을 뽑을 때의 퍼짐 정도. 클수록 다양한 행동을 시도한다 | 천천히 줄어든다 | 갑자기 0 근처로 떨어지거나(새 행동을 시도하지 않음) 계속 커진다 |
| KL | 한 번 수정할 때 신경망이 바뀐 정도 | 0.01 안팎 | 0.05를 넘는 값이 자주 나온다 (한 번에 너무 많이 수정함) |
| 잘린 비율 | 확률이 바뀐 비율이 1±0.2 범위를 벗어나서 잘린 경험의 비율 | 0.1~0.3 | 0에 가깝다(수정하는 양이 너무 적음) 또는 0.5 이상이다 |
| 가치 손실 | 가치 신경망의 예측이 실제와 얼마나 다른지 | 처음에 크다가 줄어든다 | 계속 커진다 |
```

- [ ] **Step 4: DEVLOG의 진행 상태와 일지 갱신**

- 문서 위쪽의 "현재 단계"를 `**1단계 진행 중 (빈칸 채우기)**`로 바꾼다.
- 3장 진행 계획 표에서 1단계의 상태를 `⬜`에서 `🔄`로 바꾼다.
- 3장 "설계 진행 상황" 표에서 ①, ②의 "(검토 대기)"를 "(확정)"으로 바꾼다.
- 5장 진행 일지에 `### 2026-10-10` 항목을 추가한다. 한 일에는 구현 계획 작성, 준비 코드 7개 파일 작성, 기준 실험 3개, 정답 구현으로 한 사전 확인 결과(7장 실험 번호)를 적고, 다음 할 일에는 "빈칸 다-1(evaluate_action)부터 채우기"를 적는다.
- 작업 중 오류가 있었다면 6장 양식에 맞춰 기록한다.

- [ ] **Step 5: 설계 문서의 상태 표시 변경**

`docs/specs/2026-10-09-stage1-ppo-design.md`의 `- 상태: **검토 대기**`를 `- 상태: **확정 (2026-10-10)**`으로 바꾸고, 그 아래에 `- 구현 계획: [docs/plans/2026-10-10-stage1-ppo-prep.md](../plans/2026-10-10-stage1-ppo-prep.md)` 줄을 추가한다.

- [ ] **Step 6: 커밋 (사용자가 허락한 경우에만)**

```bash
cd /home/tmdwodl/바탕화면/ISAACSIM && git add code/DEVLOG.md code/docs
git commit -m "docs: add glossary and hand off stage 1 blanks"
```

- [ ] **Step 7: 사용자에게 넘기기**

사용자에게 아래 내용을 전달한다.
1. 준비가 끝났다는 사실과, 사전 확인에서 나온 결과(정답 구현의 씨앗값별 점수, `rsl_rl`의 점수).
2. 지금 `pytest ppo/test_ppo.py`를 실행하면 31개가 모두 실패하는 것이 정상이라는 점.
3. 첫 번째 빈칸(다-1 `evaluate_action`)의 원리 설명: 정규분포가 무엇인지, 로그 확률을 왜 쓰는지, 수식의 각 항이 무엇을 뜻하는지.
4. 빈칸을 채운 뒤 실행할 채점 명령: `pytest ppo/test_ppo.py -k evaluate`

---

## 변경 기록

### 2026-10-10: 학습 속도 자동 조절 추가 (작업 5에서 발견)

작업 5의 사전 확인에서, 정답 구현이 씨앗값 3에서 통과 기준에 못 미쳤다(평균 총점 4.61, 평균 판 길이 282.0). 원인은 학습 속도를 0.001로 고정한 설정이었다. 사용자와 상의해서 아래와 같이 바꿨다. 자세한 과정은 DEVLOG 6장의 [오류-1]에 있다.

- 새 빈칸 `adapt_learning_rate(optimizer, approx_kl, desired_kl) -> 학습 속도`를 "바-1"로 추가했다. 기존 `ppo_update`는 "바-2"가 되었고, 미니배치마다 `ppo_loss` 뒤에 `adapt_learning_rate`를 부른다.
- `PPOConfig`에 `desired_kl = 0.01`을 추가했다. `ppo.py`에 상수 `LR_MIN = 1e-5`, `LR_MAX = 1e-2`, `LR_FACTOR = 1.5`를 추가했다.
- `train.py`는 `cfg.learning_rate` 대신 실제 학습 속도(`optimizer.param_groups[0]["lr"]`)를 기록하고 화면에도 보여준다.
- 빈칸은 5개에서 6개로, 채점기 검사는 31개에서 38개로 늘었다(학습 속도 조절 검사 6개, 고치기 반복 검사 1개 추가). 전체 검사는 49개이다(채점기 38개, `model.py` 8개, `plot_compare.py` 3개).
- Global Constraints의 설정값 중 "학습 속도 0.001 고정"은 "처음 0.001, 이후 KL을 보고 자동 조절"로 바뀌었다.

이 문서 본문에 적힌 `ppo.py`, `test_ppo.py`, `train.py`의 코드는 바뀌기 전의 것이다. 지금의 정확한 내용은 저장소의 파일을 기준으로 한다.
