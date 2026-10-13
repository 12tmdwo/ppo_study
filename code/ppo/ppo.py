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

    std = std.expand_as(mean)  # (N, A)
    log_prob = -((action - mean) ** 2) / (2 * std ** 2) - torch.log(std) - 0.5 * math.log(2 *math.pi)
    log_prob = log_prob.sum(-1)  # (N,)

    entropy = 0.5 + 0.5 * math.log(2 * math.pi) + torch.log(std)
    entropy = entropy.sum(-1)  # (N,)

    return log_prob, entropy


def sample_action(mean, std):
    action = torch.randn_like(mean)  # (N, A)
    action = mean + std * action  # (N, A)
    log_prob, _ = evaluate_action(mean, std, action)

    return action, log_prob



def compute_gae(rewards, values, dones, last_values, gamma, lam):

    T = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    for a in reversed(range(T)):
        next_values = last_values if a == T - 1 else values[a + 1]
        not_done = 1.0 - dones[a]
        delta = rewards[a] + (gamma * next_values * not_done) - values[a]
        advantages[a] = delta + gamma * lam * not_done * (advantages[a + 1] if a < T - 1 else 0.0)
    returns = advantages + values

    return advantages, returns


def ppo_loss(new_log_prob, old_log_prob, advantages, new_values, returns, entropy,
             clip_param, value_coef, entropy_coef):
    ratio = torch.exp(new_log_prob - old_log_prob)

    policy_loss = -torch.mean(torch.min(ratio * advantages, torch.clamp(ratio, 1 - clip_param, 1 + clip_param) * advantages))
    value_loss = torch.mean((new_values - returns) ** 2)

    loss = policy_loss + value_coef * value_loss - entropy_coef * torch.mean(entropy)

    return loss, {
        "policy_loss": policy_loss.item(),
        "value_loss": value_loss.item(),
        "entropy": torch.mean(entropy).item(),
        "approx_kl": torch.mean((ratio - 1) - torch.log(ratio)).item(),
        "clip_fraction": torch.mean((torch.abs(ratio - 1) > clip_param).float()).item()
    }                                                                                                                                                                                                   



def adapt_learning_rate(optimizer, approx_kl, desired_kl):
    
    if desired_kl is None:
        return optimizer.param_groups[0]["lr"]

    lr = optimizer.param_groups[0]["lr"]
    if approx_kl > desired_kl * 2:
        lr = max(lr / LR_FACTOR, LR_MIN)
    elif 0 < approx_kl < desired_kl / 2:
        lr = min(lr * LR_FACTOR, LR_MAX)

    for group in optimizer.param_groups:
        group["lr"] = lr

    return lr



def ppo_update(actor, critic, optimizer, batch, cfg):

    advantages = batch["advantages"]
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    infos = []
    for epoch in range(cfg.num_epochs):
        perm = torch.randperm(advantages.size(0), device=advantages.device)
        mini_batch_size = advantages.size(0) // cfg.num_mini_batches

        for i in range(cfg.num_mini_batches):
            idx = perm[i * mini_batch_size:(i + 1) * mini_batch_size]
            mini_batch = {k: v[idx] for k, v in batch.items()}

            new_mean = actor(mini_batch["obs"])
            new_std = actor.std
            new_log_prob, entropy = evaluate_action(new_mean, new_std, mini_batch["actions"])
            new_values = critic(mini_batch["obs"])

            loss, info = ppo_loss(
                new_log_prob, mini_batch["log_probs"], advantages[idx],
                new_values, mini_batch["returns"], entropy,
                cfg.clip_param, cfg.value_coef, cfg.entropy_coef
            )
            infos.append(info)
            adapt_learning_rate(optimizer, info["approx_kl"], cfg.desired_kl)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(actor.parameters()) + list(critic.parameters()), cfg.max_grad_norm)
            optimizer.step()

    return {k: sum(info[k] for info in infos) / len(infos) for k in info}
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
