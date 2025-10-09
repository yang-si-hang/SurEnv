"""
Data generation for the case of Psm Envs and demonstrations.
Refer to
https://github.com/openai/baselines/blob/master/baselines/her/experiment/data_generation/fetch_data_generation.py
"""
import os
from pathlib import Path
import argparse
import gymnasium as gym
from gymnasium.envs.registration import registry
import time
import numpy as np
import imageio
from sim_envs.const import ROOT_DIR_PATH
import sim_envs.gymnasium_utils
from sim_envs.utils.data_utils import flatten_dict

DATA_DIR_PATH = os.path.join(ROOT_DIR_PATH, 'data')

parser = argparse.ArgumentParser(description='generate demonstrations for imitation')
parser.add_argument('--env', type=str, required=True,
                    help='the environment to generate demonstrations')
parser.add_argument('--output_dir', type=str, default=None)
parser.add_argument('--video', action='store_true',
                    help='whether or not to record video')      # 只需要在命令行加上 --video 即可
parser.add_argument('--steps', type=int,
                    help='how many steps allowed to run')       # 每个epsisode的最大步数
parser.add_argument("--episodes", type=int, default=200,
                    help="number of episodes to generate")

# 定义一个字符串列表，模拟终端输入
test_args = [
    '--env', 'GoalNeedlePick-v0',
    '--output_dir', f'{DATA_DIR_PATH}/demo/Needlepick_random_1000',
    '--video',
    '--episodes', '1000'
]
args = parser.parse_args(test_args) # 将参数列表传递给 parse_args()
# args = parser.parse_args()    # 从命令行获取参数

def main():
    env = gym.make(args.env, render_mode='rgb_array')  # 'human'
    num_episodes = args.episodes
    num_itr = num_episodes  # if not args.video else 1      # 生成演示数据的次数
    cnt = 0

    if args.output_dir is not None:
        SAVE_DIR_PATH = args.output_dir
    else:
        SAVE_DIR_PATH = os.path.join(DATA_DIR_PATH, 'demo', f'{args.env}_{num_itr}')
    Path.mkdir(Path(SAVE_DIR_PATH), exist_ok=True)
    print("Saving data to:", SAVE_DIR_PATH)

    if args.steps is None:
        args.steps = env._max_episode_steps

    env.reset()
    print("Reset!")
    init_time = time.time()

    while cnt < num_itr:
        print("\n", "="*5, f"ITERATION NUMBER {cnt:d}", "="*5)
        obs, info = env.reset()
        data_to_save, images = goToGoal(env, obs)

        if data_to_save is not None:
            step_dir_path = os.path.join(SAVE_DIR_PATH, f'{cnt:04d}')
            if not os.path.exists(step_dir_path):
                os.makedirs(step_dir_path)
            np.savez_compressed(os.path.join(step_dir_path, 'data.npz'), **data_to_save)
        
            # 保存所有演示的视频
            if args.video:
                writer = imageio.get_writer(os.path.join(step_dir_path, 'video.mp4'), fps=env.metadata['render_fps'])
                for img in images:
                    writer.append_data(img)
                writer.close()
        # else:
        #     continue

        cnt += 1

    used_time = time.time() - init_time
    print("Saved data at:", SAVE_DIR_PATH)
    print("Time used: {:.1f}m, {:.1f}s\n".format(used_time // 60, used_time % 60))
    print(f"Trials: {num_itr}/{cnt}")
    env.close()

def goToGoal(env, last_obs):
    episode_data_buffer = {}
    images = []  # record video
    masks = []

    time_step = 0  # count the total number of time steps
    episode_init_time = time.time()

    obs, success = last_obs, False

    while time_step < min(env.spec.max_episode_steps, args.steps) and not success:
        rob_pose = obs["observation"][0:7]
        action = env.unwrapped.get_oracle_action(rob_pose)     # 环境内部定义的oracle策略
        if isinstance(action, tuple):
            action = action[0]
        if args.video:
            # img, mask = env.render('img_array')
            img = env.render()
            images.append(img)
            # masks.append(mask)

        obs, reward, terminated, truncated, info = env.step(action)
        # print(f" -> obs: {obs}, reward: {reward}, done: {done}, info: {info}.")
        current_step_data = {
            'observation': obs,
            'info': info,
            'action': action
        }
        flat_step_data = flatten_dict(current_step_data)

        for key, value in flat_step_data.items():
            if key not in episode_data_buffer:
                episode_data_buffer[key] = []
            episode_data_buffer[key].append(value)
        
        time_step += 1

        if isinstance(obs, dict) and info['is_success'] > 0 and not success:
            print("Timesteps to finish:", time_step)
            success = True

        # TODO ！！！
        # time.sleep(0.01)

    print("Episode time used: {:.2f}s\n".format(time.time() - episode_init_time))

    # 此时，episode_data_buffer 中的值都是列表
    aggregated_data_to_save = {}
    for key, list_of_values in episode_data_buffer.items():
        # 使用 np.array() 可以智能地将列表转换为一个大的数组
        # 例如，一个包含 50 个 shape 为 (7,) 的 action 数组的列表
        # 会被转换为一个 shape 为 (50, 7) 的数组
        aggregated_data_to_save[key] = np.array(list_of_values)

    if success:
        return aggregated_data_to_save, images
    else:
        print("❌ Failed to reach the goal in this episode.")
        return None, None


if __name__ == "__main__":
    main()
