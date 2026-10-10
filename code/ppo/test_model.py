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
