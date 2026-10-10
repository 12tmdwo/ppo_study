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


def mean_curve(seed_curves):
    """씨앗값별 곡선 [(반복 번호 목록, 값 목록), ...]에서, 모든 씨앗값에 있는 반복 번호만 골라 평균 곡선을 만든다."""
    tables = [dict(zip(steps, values)) for steps, values in seed_curves]
    common = sorted(set.intersection(*(set(table) for table in tables)))
    return common, [sum(table[step] for table in tables) / len(tables) for step in common]


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
            steps, mean = mean_curve(seed_curves)  # 같은 반복 번호끼리 평균을 낸다
            if steps:
                ax.plot(steps, mean, color=color, linewidth=2,
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
