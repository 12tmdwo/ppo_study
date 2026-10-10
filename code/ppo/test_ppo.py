"""ppo.py 빈칸 채점기. (시뮬레이터 없이 CPU에서 몇 초 안에 끝난다)

빈칸 하나만 채점
    pytest ppo/test_ppo.py -k evaluate   (다-1 evaluate_action)
    pytest ppo/test_ppo.py -k sample     (다-2 sample_action)
    pytest ppo/test_ppo.py -k gae        (라   compute_gae)
    pytest ppo/test_ppo.py -k loss       (마   ppo_loss)
    pytest ppo/test_ppo.py -k adapt      (바-1 adapt_learning_rate)
    pytest ppo/test_ppo.py -k update     (바-2 ppo_update)
전체 채점
    pytest ppo/test_ppo.py
"""
import copy
import math

import torch
from torch.distributions import Normal

from model import Actor, Critic
from ppo import (
    LR_FACTOR, LR_MAX, LR_MIN, PPOConfig,
    adapt_learning_rate, compute_gae, evaluate_action, ppo_loss, ppo_update, sample_action,
)

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


# ======================= 바-1. adapt_learning_rate =======================

def _optimizer(lr):
    return torch.optim.Adam([torch.nn.Parameter(torch.zeros(1))], lr=lr)


def _lr(optimizer):
    return optimizer.param_groups[0]["lr"]


def test_adapt_learning_rate_lowers_when_change_too_big():
    optimizer = _optimizer(1e-3)
    new_lr = adapt_learning_rate(optimizer, approx_kl=0.03, desired_kl=0.01)  # 0.03 > 0.01 * 2
    assert math.isclose(_lr(optimizer), 1e-3 / LR_FACTOR), "KL이 목표의 2배보다 크면 학습 속도를 LR_FACTOR로 나눠야 합니다."
    assert math.isclose(new_lr, _lr(optimizer)), "조절한 뒤의 학습 속도를 돌려줘야 합니다."


def test_adapt_learning_rate_raises_when_change_too_small():
    optimizer = _optimizer(1e-3)
    adapt_learning_rate(optimizer, approx_kl=0.001, desired_kl=0.01)  # 0 < 0.001 < 0.01 / 2
    assert math.isclose(_lr(optimizer), 1e-3 * LR_FACTOR), "KL이 목표의 절반보다 작으면 학습 속도에 LR_FACTOR를 곱해야 합니다."


def test_adapt_learning_rate_keeps_when_change_is_fine():
    for kl in (0.01, 0.02, 0.005, 0.0):  # 목표 근처, 경계값 두 개, 아직 신경망이 안 바뀐 경우(0)
        optimizer = _optimizer(1e-3)
        adapt_learning_rate(optimizer, approx_kl=kl, desired_kl=0.01)
        assert _lr(optimizer) == 1e-3, f"KL이 {kl}일 때는 학습 속도를 바꾸지 않아야 합니다."


def test_adapt_learning_rate_respects_limits():
    optimizer = _optimizer(LR_MAX)
    adapt_learning_rate(optimizer, approx_kl=0.001, desired_kl=0.01)
    assert _lr(optimizer) == LR_MAX, "학습 속도는 LR_MAX보다 커지면 안 됩니다."
    optimizer = _optimizer(LR_MIN)
    adapt_learning_rate(optimizer, approx_kl=0.5, desired_kl=0.01)
    assert _lr(optimizer) == LR_MIN, "학습 속도는 LR_MIN보다 작아지면 안 됩니다."


def test_adapt_learning_rate_off_without_target():
    optimizer = _optimizer(1e-3)
    new_lr = adapt_learning_rate(optimizer, approx_kl=0.5, desired_kl=None)
    assert _lr(optimizer) == 1e-3 and new_lr == 1e-3, "desired_kl이 None이면 아무것도 바꾸지 않아야 합니다."


def test_adapt_learning_rate_applies_to_every_group():
    first, second = torch.nn.Parameter(torch.zeros(1)), torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.Adam([{"params": [first]}, {"params": [second]}], lr=1e-3)
    adapt_learning_rate(optimizer, approx_kl=0.03, desired_kl=0.01)
    assert all(math.isclose(group["lr"], 1e-3 / LR_FACTOR) for group in optimizer.param_groups), \
        "optimizer.param_groups의 모든 묶음에 새 학습 속도를 넣어야 합니다."


# ======================= 바-2. ppo_update =======================

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


def test_update_changes_learning_rate_by_kl():
    def run(log_prob_shift, desired_kl):
        torch.manual_seed(0)
        actor, critic, optimizer = _models()
        batch = _batch(actor, torch.randn(64, 4))
        batch["log_probs"] = batch["log_probs"] + log_prob_shift  # 예전 로그 확률을 일부러 어긋나게 해서 KL의 크기를 정한다
        ppo_update(actor, critic, optimizer, batch, PPOConfig(num_epochs=1, desired_kl=desired_kl))
        return optimizer.param_groups[0]["lr"]

    msg = "ppo_update 안에서 미니배치마다 adapt_learning_rate를 불러야 합니다."
    assert run(-1.0, 0.01) < 1e-3, "KL이 클 때 학습 속도가 내려가야 합니다. " + msg
    assert run(-0.01, 0.01) > 1e-3, "KL이 아주 작을 때 학습 속도가 올라가야 합니다. " + msg
    assert run(-1.0, None) == 1e-3, "cfg.desired_kl이 None이면 학습 속도가 그대로여야 합니다."
