"""
将保存为hdf5格式的记录可视化
Date: 2025-12-9
"""
import os
from pathlib import Path
import numpy as np
import cv2
import h5py
import argparse

# Force headless backend before importing pyplot
import matplotlib
if os.environ.get("DISPLAY", "") == "" or os.environ.get("MPLBACKEND", "").lower() != "" or True:
    # Always use Agg to ensure headless safety (no GUI required)
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Make IPython optional in headless/minimal envs
try:
    import IPython
    e = IPython.embed
except ImportError:
    e = lambda *args, **kwargs: None

def load_hdf5(dataset_dir, dataset_name):
    dataset_path = Path(dataset_dir) / f"{dataset_name}.hdf5"
    if not dataset_path.is_file():
        print(f'Dataset does not exist at \n{dataset_path}\n')
        exit()

    with h5py.File(dataset_path, 'r') as root:
        is_sim = root.attrs['sim']
        robot_state = root['/observations/robot_state'][()]
        action = root['/action'][()]
        image_dict = dict()
        for cam_name in root[f'/observations/images/'].keys():
            image_dict[cam_name] = root[f'/observations/images/{cam_name}'][()]

    return robot_state, action, image_dict

def save_videos(video, dt, video_path=None):
    if isinstance(video, list):
        cam_names = list(video[0].keys())
        h, w, _ = video[0][cam_names[0]].shape
        w = w * len(cam_names)
        fps = int(1/dt)
        out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        for ts, image_dict in enumerate(video):
            images = []
            for cam_name in cam_names:
                image = image_dict[cam_name]
                image = image[:, :, [2, 1, 0]] # swap B and R channel
                images.append(image)
            images = np.concatenate(images, axis=1)
            out.write(images)
        out.release()
        print(f'Saved video to: {video_path}')
    elif isinstance(video, dict):
        cam_names = list(video.keys())
        all_cam_videos = []
        for cam_name in cam_names:
            all_cam_videos.append(video[cam_name])
        all_cam_videos = np.concatenate(all_cam_videos, axis=2) # width dimension

        n_frames, h, w, _ = all_cam_videos.shape
        fps = int(1 / dt)
        out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        for t in range(n_frames):
            image = all_cam_videos[t]
            image = image[:, :, [2, 1, 0]]  # swap B and R channel
            out.write(image)
        out.release()
        print(f'Saved video to: {video_path}')

def visualize_traj(state_list, action_list, plot_path=None,  ylim=None, label_overwrite=None):
    """
    Visualize the robot joint positions over time.
    robot_state: (N, 8) numpy array, where first 3 are position, next 4 are orientation (quaternion), last is jaw angle
    """
    if label_overwrite:
        label1, label2 = label_overwrite
    else:
        label1, label2 = 'State', 'Action'

    state = np.array(state_list)  # (N, 8)
    action = np.array(action_list)  # (N, 10)

    labels = ['X Position (m)', 'Y Position (m)', 'Z Position (m)', 'Qx', 'Qy', 'Qz', 'Qw', 'Jaw Angle (rad)']

    # plot robot state: tip position, orientation (quaternion), jaw angle
    time_steps = state.shape[0]
    time_array = np.arange(time_steps)

    fig, axs = plt.subplots(8, 1, figsize=(10, 20))
    for i in range(8):
        axs[i].plot(time_array, state[:, i], label=labels[i])
        axs[i].set_title(labels[i])
        axs[i].set_xlabel('Time Step')
        axs[i].set_ylabel('Value')
        axs[i].legend()
        axs[i].grid()

    plt.tight_layout()
    state_plot_path = plot_path.replace('.png', '_state.png')
    plt.savefig(state_plot_path)
    print(f'Saved plot to: {state_plot_path}')

    # plot robot action: delta position, delta orientation (first two columns of rotation matrix), jaw angle
    time_steps = action.shape[0]
    time_array = np.arange(time_steps)

    fig, axs = plt.subplots(10, 1, figsize=(10, 25))
    action_labels = [
        'Delta X Position (m)', 'Delta Y Position (m)', 'Delta Z Position (m)',
        'R11', 'R21', 'R31', 'R12', 'R22', 'R32',
        'Jaw Angle (rad)'
    ]
    for i in range(10):
        axs[i].plot(time_array, action[:, i], label=action_labels[i])
        axs[i].set_title(action_labels[i])
        axs[i].set_xlabel('Time Step')
        axs[i].set_ylabel('Value')
        axs[i].legend()
        axs[i].grid()

    plt.tight_layout()
    action_plot_path = plot_path.replace('.png', '_action.png')
    plt.savefig(action_plot_path)
    print(f'Saved action plot to: {action_plot_path}')

def main():
    from sim_envs.const import DATA_DIR_PATH
    FPS = 100
    dataset_dir = Path(DATA_DIR_PATH) / "demo/NeedlePick-v0-3"
    dataset_name = 1
    robot_state, action, image_dict = load_hdf5(dataset_dir, dataset_name)

    save_videos(image_dict, 1./FPS, video_path=str(Path(dataset_dir) / f"{dataset_name}_video.mp4"))
    visualize_traj(robot_state, action, plot_path=str(Path(dataset_dir) / f"{dataset_name}_traj.png"))

if __name__ == "__main__":
    main()