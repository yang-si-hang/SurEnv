"""
记录仿真场景的数据
Date: 2025-12-9
"""
from pathlib import Path
import h5py
import shutil
import numpy as np
import gymnasium as gym

from sim_envs.const import DATA_DIR_PATH
from sim_envs.utils.pybullet_utils import compute_delta_rotations
import sim_envs.gymnasium_utils

EPISODE_NUM = 100
TASK_NAME = "NeedlePick-v0"
DATA_DIR_PATH = Path(DATA_DIR_PATH)
DEMO_DIR = DATA_DIR_PATH / f"demo/{TASK_NAME}-{EPISODE_NUM}"

# shutil.rmtree(DEMO_DIR)
DEMO_DIR.mkdir(parents=True, exist_ok=True)

success = []
saved = 0

for i in range(EPISODE_NUM):
    print(f"{'='*5} Try to record episode {i}...")

    env = gym.make(f'{TASK_NAME}', render_mode="rgb_array", obs_type="rgb", max_episode_steps=800)

    try:
        obs, info = env.reset()
        terminated = False
        truncated = False
        step:int = 0
        total_reward:float = 0
        data_dict = {   # TODO 需要根据任务修改
            '/observations/robot_state': [],
            '/action': [],
        }
        if env.unwrapped.obs_type in ['rgb', 'rgbd']:
            data_dict['/observations/images/wrist_cam'] = []
            data_dict['/observations/images/env_cam'] = []

        while not (terminated or truncated):
            rob_pose = env.unwrapped._get_robot_state(idx=0)  # 7-dim
            action = env.unwrapped.get_oracle_action(rob_pose)

            rob_state = obs["robot_state"].copy()
            data_dict['/observations/robot_state'].append(rob_state)

            # image data
            wrist_img = obs["images"]["wrist_cam"].copy()
            env_img = obs["images"]["env_cam"].copy()
            data_dict['/observations/images/wrist_cam'].append(wrist_img)
            data_dict['/observations/images/env_cam'].append(env_img)

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            step += 1

        final_rob_state = obs["robot_state"].copy()
        data_dict['/observations/robot_state'].append(final_rob_state)

        if info.get('is_success', False):
            print(f"Total Reward: {total_reward}, Steps: {step}")
            print("Result: ✅ Success!")
            success.append(True)
        else:
            success.append(False)
            if truncated:
                print("Result: ⌛️ Truncated (Time Limit Reached).")
            else:
                print("Result: ❌ Failure.")
            continue  # 只保存成功的轨迹

        # --- 4. 离线计算动作与对齐 (Data Alignment) ---
        
        # 转换状态列表为 numpy 数组: shape (N+1, D)
        all_states = np.array(data_dict['/observations/robot_state'])
        
        # 计算差分动作: Action_t = State_{t+1} - State_t
        delta_pos = np.diff(all_states[:, :3], axis=0)  # 3-dim
        delta_orn_mat = compute_delta_rotations(all_states[:, 3:7])  # 9-dim
        delta_orn_mat = delta_orn_mat[:, [0, 3, 6, 1, 4, 7]]    # 取旋转矩阵的第一、二列，shape (N, 6)
        jaw_angle = all_states[1:, 7]  # 夹爪用绝对编码(SRT中没具体说怎么做) 1-dim        
        robot_actions = np.concatenate([delta_pos, delta_orn_mat, jaw_angle[:, None]], axis=1)  # shape (N, 10)

        data_dict['/action'] = robot_actions

        # 统一截断到 N，丢弃最后那一个多余的 State
        final_len = np.shape(robot_actions)[0]
        for key, value in data_dict.items():
            data_dict[key] = value[:final_len]

        dataset_index = saved
        dataset_path = DEMO_DIR / f'{dataset_index}'
        with h5py.File(dataset_path.with_suffix('.hdf5'), 'w', rdcc_nbytes=1024 ** 2 * 2) as root:
            root.attrs['sim'] = True
            root.attrs['episode_length'] = final_len

            root.create_dataset('action', (final_len, 10), dtype='float32',) # 3+6+1

            obs = root.create_group('observations')
            obs.create_dataset('robot_state', (final_len, 8), dtype='float32',) # 3+4+1

            image = obs.create_group('images') 

            _ = image.create_dataset('wrist_cam', (final_len, 480, 640, 3), dtype='uint8',
                                    chunks=(1, 480, 640, 3), )
            _ = image.create_dataset('env_cam', (final_len, 480, 640, 3), dtype='uint8',
                                    chunks=(1, 480, 640, 3), )

            for name, array in data_dict.items():
                root[name][...] = array

    finally:
        env.close()

        saved += 1

print(f'Saved to {DEMO_DIR}')
print(f'Success: {np.sum(success)} / {len(success)}')