"""직접 구현한 PPO로 Isaac Lab 수레-막대(Cartpole)를 학습시킨다. (준비된 코드: 읽고 이해하면 된다)

실행
    python ppo/train.py --seed 1

전체 흐름 (150번 반복)
    1) 경험 모으기 : 환경 4096개를 16걸음씩 진행하며 관찰값·행동·점수를 저장소에 담는다.
    2) 이득 계산   : compute_gae  (빈칸 라)
    3) 신경망 수정 : ppo_update   (빈칸 바-2. 그 안에서 바-1이 학습 속도를 조절한다)
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
            info = ppo_update(actor, critic, optimizer, storage.flatten(advantages, returns), cfg)  # 빈칸 바-2
            storage.clear()

            # ---------- 4) 기록 ----------
            fps = NUM_STEPS * num_envs / (time.time() - start)
            mean_std = actor.std.mean().item()
            learning_rate = optimizer.param_groups[0]["lr"]  # 빈칸 바-1이 조절한 지금의 학습 속도
            writer.add_scalar("Loss/value", info["value_loss"], it)
            writer.add_scalar("Loss/surrogate", info["policy_loss"], it)
            writer.add_scalar("Loss/entropy", info["entropy"], it)
            writer.add_scalar("Loss/approx_kl", info["approx_kl"], it)
            writer.add_scalar("Loss/clip_fraction", info["clip_fraction"], it)
            writer.add_scalar("Loss/learning_rate", learning_rate, it)
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
                f" | 잘린 비율 {info['clip_fraction']:.2f} | 학습 속도 {learning_rate:.5f} | 초당 {fps:,.0f}걸음"
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
