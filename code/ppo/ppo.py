"""PPO의 핵심 (직접 작성하는 파일).

아래 빈칸 함수 6개를 위에서부터 순서대로 채운다.

    다-1. evaluate_action       행동의 로그 확률과 엔트로피 계산
    다-2. sample_action         행동 뽑기
    라.   compute_gae           이득 계산
    마.   ppo_loss              PPO 손실
    바-1. adapt_learning_rate   학습 속도 조절
    바-2. ppo_update            고치기 반복

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
    learning_rate: float = 1e-3  # 처음 학습 속도 (학습하는 동안 바-1이 자동으로 조절한다)
    desired_kl: float | None = 0.01  # 목표 KL: 신경망이 한 번에 이 정도만 바뀌게 한다. None이면 학습 속도를 조절하지 않는다


# 학습 속도 조절(빈칸 바-1)에서 쓰는 값. (완성된 코드: 고칠 필요 없음)
LR_MIN = 1e-5    # 학습 속도의 최소값
LR_MAX = 1e-2    # 학습 속도의 최대값
LR_FACTOR = 1.5  # 한 번에 올리거나 내리는 배율


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


def adapt_learning_rate(optimizer, approx_kl, desired_kl):
    """[빈칸 바-1] 신경망이 바뀐 정도(KL)를 보고 학습 속도를 조절한다.

    학습 속도가 너무 크면 신경망이 한 번에 많이 바뀌어서 학습이 무너지고,
    너무 작으면 거의 바뀌지 않아서 학습이 느리다. 그래서 방금 잰 KL이 목표(desired_kl)보다
    많이 크면 속도를 내리고, 많이 작으면 속도를 올린다. rsl_rl도 같은 방법을 쓴다.

    입력
        optimizer  : 최적화 도구. 지금 학습 속도는 optimizer.param_groups[0]["lr"]에 들어 있다.
        approx_kl  : 방금 ppo_loss가 계산한 KL 근사값 (파이썬 숫자)
        desired_kl : 목표 KL (예: 0.01). None이면 아무것도 바꾸지 않는다.
    출력
        조절한 뒤의 학습 속도 (파이썬 숫자). 바꾸지 않았으면 원래 학습 속도.

    규칙
        approx_kl > desired_kl * 2      -> 학습 속도를 LR_FACTOR로 나눈다. 단, LR_MIN보다 작아지지 않게 한다.
        0 < approx_kl < desired_kl / 2  -> 학습 속도에 LR_FACTOR를 곱한다. 단, LR_MAX보다 커지지 않게 한다.
        그 밖의 경우                    -> 그대로 둔다.

    힌트
        - KL이 정확히 0인 경우는 신경망이 아직 한 번도 바뀌지 않은 것이므로, 속도를 올리지 않는다.
        - 둘 중 큰 값과 작은 값: max(a, b), min(a, b)
        - 바꾼 학습 속도는 모든 묶음에 넣어야 실제로 적용된다.
              for group in optimizer.param_groups:
                  group["lr"] = 새 학습 속도

    채점: pytest ppo/test_ppo.py -k adapt
    """
    raise NotImplementedError("빈칸 바-1: ppo.py의 adapt_learning_rate를 채워주세요.")


def ppo_update(actor, critic, optimizer, batch, cfg):
    """[빈칸 바-2] 모은 경험을 작은 묶음으로 나누어, 신경망을 여러 번 수정한다.

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
                - adapt_learning_rate(optimizer, info["approx_kl"], cfg.desired_kl)로 학습 속도를 조절한다.
                  (info는 방금 ppo_loss가 돌려준 것이다)
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
    raise NotImplementedError("빈칸 바-2: ppo.py의 ppo_update를 채워주세요.")
