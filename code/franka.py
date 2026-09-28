# Isaac Sim 6.1 standalone 연습 예제
# 1) 바닥 깔기  2) Franka 불러오기  3) 매 스텝 관절 각도 읽기  4) 관절 하나에 목표각 주기
#
# 실행: conda activate env_isaaclab && python franka_hello.py

# --- 1. Isaac Sim 실행 (다른 isaacsim import보다 반드시 먼저) ---
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})  # True로 바꾸면 화면 없이 실행

# --- 2. Isaac Sim 모듈 import (SimulationApp 생성 이후에만 가능) ---
import numpy as np
import omni.timeline
import isaacsim.core.experimental.utils.stage as stage_utils
from isaacsim.core.experimental.prims import Articulation
from isaacsim.core.simulation_manager import SimulationManager
from isaacsim.storage.native import get_assets_root_path


def to_np(x):
    """experimental API는 warp 배열을 돌려주므로 numpy로 변환"""
    return x.numpy() if hasattr(x, "numpy") else np.asarray(x)


assets_root = get_assets_root_path()
if assets_root is None:
    raise RuntimeError("Isaac Sim 에셋 경로를 찾지 못했습니다.")

# --- 3. 씬 구성: 바닥 + Franka ---
stage_utils.add_reference_to_stage(
    usd_path=assets_root + "/Isaac/Environments/Grid/default_environment.usd",
    path="/World/ground",
)

franka_usd = assets_root + "/Isaac/Robots_Multiphysics/FrankaRobotics/FrankaPanda/franka/franka.usda"
stage_utils.add_reference_to_stage(
    usd_path=franka_usd,
    path="/World/Franka",
    variants=[("Gripper", "alternatefinger"), ("Mesh", "quality")],
)

# 로봇을 Articulation 객체로 감싸기 (관절 읽기/쓰기용 핸들)
robot = Articulation("/World/Franka", positions=[[0.0, 0.0, 0.0]], reset_xform_op_properties=True)

# --- 4. 물리 시뮬레이션 시작 ---
omni.timeline.get_timeline_interface().play()
simulation_app.update()

TOTAL_STEPS = 600
PRINT_EVERY = 50
COMMAND_STEP = 100      # 이 스텝에서 관절 명령을 준다
TARGET_JOINT = "panda_joint4"
TARGET_ANGLE = -1.0     # rad

printed_info = False

for step in range(TOTAL_STEPS):
    if SimulationManager.is_simulating():
        # 처음 한 번만 로봇 정보 출력
        if not printed_info:
            print("관절 개수:", robot.num_dofs)
            print("관절 이름:", robot.dof_names)
            printed_info = True

        # [관찰] 관절 각도·속도 읽기
        if step % PRINT_EVERY == 0:
            pos = to_np(robot.get_dof_positions())[0]
            vel = to_np(robot.get_dof_velocities())[0]
            print(f"[step {step:4d}] 각도: {np.round(pos, 3)}")
            print(f"             속도: {np.round(vel, 3)}")

        # [행동] 특정 스텝에서 관절 하나에 목표각 주기
        if step == COMMAND_STEP:
            idx = robot.get_dof_indices(TARGET_JOINT)
            robot.set_dof_position_targets(TARGET_ANGLE, dof_indices=idx)
            print(f">>> {TARGET_JOINT} 목표각 {TARGET_ANGLE} rad 명령")

    # [스텝 진행] 물리 + 렌더링 한 번
    simulation_app.update()

simulation_app.close()