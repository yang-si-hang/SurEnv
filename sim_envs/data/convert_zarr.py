import os
import numpy as np
import zarr
from tqdm import tqdm

from sim_envs.const import DATA_DIR_PATH, ROOT_DIR_PATH

# ==================== 配置区 ====================
RAW_DATA_DIR = os.path.join(DATA_DIR_PATH, f'demo/Needlepick_random_1000')

# 2. 设置你希望保存 Zarr 数据集的路径
#    (这个文件夹会自动创建，通常以 .zarr 结尾)
PROJECT_DIR = os.path.dirname(ROOT_DIR_PATH)
ZARR_PATH = os.path.join(PROJECT_DIR, f'diffusion_policy/data/demo/needlepick/needlepick_random_1000.zarr')
# ===============================================


def main():
    print(f"正在从 {RAW_DATA_DIR} 转换数据到 {ZARR_PATH}")

    # 获取所有 episode 文件夹，并进行排序
    episode_dirs = sorted([d for d in os.listdir(RAW_DATA_DIR) if os.path.isdir(os.path.join(RAW_DATA_DIR, d))])
    
    if not episode_dirs:
        print("错误：在指定目录下没有找到 episode 文件夹。")
        return

    # --- 步骤 1: 检查第一个 npz 文件，获取数据键和形状 ---
    print("步骤 1: 检查数据结构...")
    first_episode_dir = episode_dirs[0]
    first_episode_path = os.path.join(RAW_DATA_DIR, first_episode_dir)
    
    # 找到第一个 episode 里的第一个 npz 文件
    first_npz_file = sorted(os.listdir(first_episode_path))[0]
    first_npz_path = os.path.join(first_episode_path, first_npz_file)
    
    with np.load(first_npz_path) as data:
        data_keys = list(data.keys())
        data_shapes = {key: data[key].shape for key in data_keys}
        data_dtypes = {key: data[key].dtype for key in data_keys}

    print("检测到以下数据键、形状和类型:")
    for key in data_keys:
        print(f"- {key}: shape={data_shapes[key]}, dtype={data_dtypes[key]}")

    # --- 步骤 2: 初始化 Zarr 数据集 ---
    print("\n步骤 2: 初始化 Zarr 存储...")
    # 'w' 模式会覆盖已存在的文件
    root = zarr.open(ZARR_PATH, 'w')
    data_group = root.create_group('data')
    meta_group = root.create_group('meta')

    zarr_arrays = {}
    for key in data_keys:
        # 初始化可追加的 Zarr 数组，初始长度为0
        zarr_arrays[key] = data_group.create_dataset(
            key,
            shape=(0,) + data_shapes[key][1:],  # 例如 (0, obs_dim)
            chunks=(1024,) + data_shapes[key][1:], # chunks 的设置会影响性能
            dtype=data_dtypes[key]
        )
    print("Zarr 存储初始化完成。")

    # --- 步骤 3: 遍历所有 episode 并写入数据 ---
    print("\n步骤 3: 开始处理所有 episode...")
    episode_ends = []
    total_steps = 0

    for episode_dir in tqdm(episode_dirs, desc="处理 Episodes"):
        episode_path = os.path.join(RAW_DATA_DIR, episode_dir)
        # 假设每个文件夹下只有一个 data.npz 文件
        npz_path = os.path.join(episode_path, 'data.npz')

        with np.load(npz_path) as data:
            # 直接追加整个 episode 的数据
            for key in data_keys:
                zarr_arrays[key].append(data[key])
            
            # 更新 episode 结束点
            # 我们需要从数据中获取轨迹长度，以第一个key为例
            first_key = data_keys[0]
            episode_len = data[first_key].shape[0]
            total_steps += episode_len
            episode_ends.append(total_steps)

    # --- 步骤 4: 写入元数据 ---
    print("\n步骤 4: 写入元数据...")
    meta_group.create_dataset('episode_ends', data=np.array(episode_ends, dtype=np.int64))
    print("元数据写入完成。")

    # --- 总结 ---
    print("\n转换完成！")
    print(f"总共处理了 {len(episode_dirs)} 个 episodes。")
    print(f"总共处理了 {total_steps} 个时间步。")
    print("Zarr 数据集结构:")
    print(root.tree())


if __name__ == '__main__':
    main()