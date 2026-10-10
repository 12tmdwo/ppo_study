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
