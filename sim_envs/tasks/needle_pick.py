""" Needle pick task for RL

action: [dx, dy, dz, dyaw, jaw_control]
state: 
"""
import os
import time
from typing import Tuple
import numpy as np
from PIL import Image
from pathlib import Path
import pybullet as p
import gymnasium as gym
from gymnasium import spaces
# np.set_printoptions(linewidth=120)

from sim_envs.gymnasium_utils.goal_env import GoalEnvWrapper
from sim_envs.utils.pybullet_utils import (
    p_step,
    get_link_pose, get_joint_positions,
    wrap_angle,
    get_pose_in_camera_frame,
)
from sim_envs.tasks.psm_env import PsmEnv
from sim_envs.const import ASSET_DIR_PATH, DEBUG_DIR_PATH

class NeedlePickEnv(PsmEnv):
    """ 需要补充
    Action space:

    Observation space:

    """
    # MAX_EPISODE_STEPS = 800 # max steps per episode
    POSE_TRAY = ((0.55, 0, 0.6751), (0, 0, 0))
    WORKSPACE_LIMITS = ((0.50, 0.60), (-0.05, 0.05), (0.685, 0.745))  # reduce tip pad contact
    NEEDLE_SIZE = 0.03
    SCALING = 5.    # 决定针和tray的大小, 并且缩放工作空间; 由于覆盖了父类, 导致父类几何对象全部随之缩放
    NEEDLE_WORKSPACE = SCALING * np.array([[WORKSPACE_LIMITS[0][0] + NEEDLE_SIZE, WORKSPACE_LIMITS[0][1] - NEEDLE_SIZE],
                                           [WORKSPACE_LIMITS[1][0] + NEEDLE_SIZE, WORKSPACE_LIMITS[1][1] - NEEDLE_SIZE],
                                           [WORKSPACE_LIMITS[2][0]+0.01, WORKSPACE_LIMITS[2][0]+0.01]])

    # TODO: grasp is sometimes not stable; check how to fix it

    def __init__(self, render_mode=None, cid=-1, obs_type='state'):
        if obs_type not in self.metadata["obs_modes"]:
            raise ValueError(f"Unsupported obs_type '{obs_type}'. Supported types: {self.metadata['obs_modes']}")

        super().__init__(render_mode, cid, obs_type)
        self.current_step = 0
        self.needle_id = None

        self._env_setup()
        p_step(0.3)  # wait for stable

        # 初始化任务目标, 如果想要读取goal, 调用 self.goal
        self.goal = self._sample_goal()
        self._sample_goal_callback()

        obs = self._get_obs()
        if self.obs_type == "state":
            self.observation_space = spaces.Dict({
                'robot_state': spaces.Box(-np.inf, np.inf, shape=obs['robot_state'].shape, dtype=np.float32),
                'object_pos': spaces.Box(-np.inf, np.inf, shape=obs['object_pos'].shape, dtype=np.float32),
                'object_rel_pos': spaces.Box(-np.inf, np.inf, shape=obs['object_rel_pos'].shape, dtype=np.float32),
                'grasping_pose': spaces.Box(-np.inf, np.inf, shape=obs['grasping_pose'].shape, dtype=np.float32),
            })
        elif self.obs_type == "rgb":
            self.observation_space = spaces.Dict({
                'robot_state': spaces.Box(-np.inf, np.inf, shape=obs['robot_state'].shape, dtype=np.float32),
                'images': spaces.Dict({
                    'wrist_cam': spaces.Box(0, 255, shape=obs['images']['wrist_cam'].shape, dtype=np.uint8),
                    'env_cam': spaces.Box(0, 255, shape=obs['images']['env_cam'].shape, dtype=np.uint8),
                }),
            })


    def step(self, action: np.ndarray)-> Tuple[dict, float, bool, bool, dict]:
        """
        执行一步环境交互, 并返回符合 Gymnasium API 的5个值.
        p_step表示步进仿真若干时间, 与FPS对齐.
        action的大小需要与FPS耦合, 以保证物理时间上的最大位移变化一致.
        ! 潜在问题: 无法控制夹爪的开合速度 !
        """
        # 预处理动作并驱动机器人、步进仿真 (调用父类 PsmEnv 的辅助方法)
        self.current_step += 1
        if len(action.shape) > 1:
            action = action.squeeze()
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self._set_action(action)
        p_step(self._duration)

        self._step_callback()  # 仿真执行的回调函数
        obs = self._get_obs()

        terminated = False
        truncated = False
        info = {}

        achieved_goal, *waypoints = self._get_task_obs()  # Needle position
        # 没有达到goal, 就是-1, 达到就是0
        reward = self.compute_reward(achieved_goal, self.goal, {})  # 没有用 info

        if reward == 0:
            terminated = True
            info['is_success'] = True
        else:
            info['is_success'] = False
        info['distance'] = np.linalg.norm(achieved_goal - self.goal)    # 无需缩放, scaling作用在阈值

        # gym包装器会自动判断, 外部调用也改为外部判断
        # # 检查是否“截断” (Truncated) - 因时间限制
        # if self.current_step >= self.MAX_EPISODE_STEPS:
        #     truncated = True
        #     if 'is_success' not in info:
        #         info['is_success'] = False

        # # 如果回合结束，填充 final_observation
        # if terminated or truncated:
        #     # 即使回合结束，我们也获取一次“最终”的观测状态(这对于很多算法的价值函数计算至关重要)
        #     final_obs = self._get_obs()
        #     info['final_observation'] = final_obs
        #     info['episode_length'] = self.current_step

        # 返回符合 Gymnasium 标准的 5-元组
        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        """ 初始化环境之后, 就要调用 reset 方法, 返回初始观测和信息 """
        super().reset(seed, options)

        if self._activated != -1:
            self._release(self._activated)      # 放下物体, 解开约束
        self.psm1.move_jaw(np.deg2rad(80.0))  # open jaw to move the needle

        # tray pad也需要重置
        p.resetBasePositionAndOrientation(self.obj_ids['fixed'][0],
                                          np.array(self.POSE_TRAY[0]) * self.SCALING,
                                          p.getQuaternionFromEuler(self.POSE_TRAY[1]))

        # 重置Needle位置, 未给定则随机放置
        if options and  "needle_pose" in options:
            needle_pose = options["needle_pose"]
            p.resetBasePositionAndOrientation(self.needle_id, needle_pose[0],
                                               p.getQuaternionFromEuler(needle_pose[1]))
        else:
            needle_pose = self.np_random.uniform(low=self.NEEDLE_WORKSPACE[:,0], high=self.NEEDLE_WORKSPACE[:,1])
            # yaw = 0.1 * np.pi
            yaw = self.np_random.uniform(-0.5, 0.5) * np.pi
            p.resetBasePositionAndOrientation(self.needle_id,
                                            tuple(needle_pose),
                                            p.getQuaternionFromEuler((0, 0, yaw)))
        # print(f"- Needle init pose: {np.round(needle_pose, 4)}, yaw: {yaw:.4f} rad")

        self.psm1.close_jaw()  # close jaw

        p_step(0.3)  # 让针落到桌面上

        if options and "goal" in options:
            self.goal = options["goal"]
        else:
            self.goal = self._sample_goal()     # 目标最终位置
        self._sample_goal_callback()        # 根据needle位置设置waypoints

        self.current_step = 0

        return self._get_obs(), {}

    def _env_setup(self):
        """ 只加载环境中需要的物体, 并进行必要的设置 """
        super(NeedlePickEnv, self)._env_setup()
        # 将父类的设置清除, 重新设置
        self.psm_1_matrices.clear()
        self.psm_1_jaw.clear()
        # np.random.seed(4)  # for experiment reproduce
        self.has_object = True
        self._waypoint_goal = True  # use waypoints to guide the action

        # reset robot
        workspace_limits = self.workspace_limits1
        pos = (workspace_limits[0][0],
               workspace_limits[1][1],
               (workspace_limits[2][1] + workspace_limits[2][0]) / 2)
        orn = (0.5, 0.5, -0.5, -0.5)
        joint_positions = self.psm1.inverse_kinematics((pos, orn), self.psm1.EEF_LINK_INDEX)
        self.psm1.reset_joint(joint_positions)
        self.block_gripper = False
        # physical interaction
        self._contact_approx = False

        # tray pad
        tray_id = p.loadURDF(os.path.join(ASSET_DIR_PATH, 'tray/tray_pad.urdf'),
                            np.array(self.POSE_TRAY[0]) * self.SCALING,
                            p.getQuaternionFromEuler(self.POSE_TRAY[1]),
                            globalScaling=self.SCALING)
        # p.resetBasePositionAndOrientation(tray_id, np.array(self.POSE_TRAY[0]) * self.SCALING,
        #                                   p.getQuaternionFromEuler(self.POSE_TRAY[1]))
        self.obj_ids['fixed'].append(tray_id)  # 1

        # needle, 此处未做随机化位置
        yaw = 0.1 * np.pi
        needle_id = p.loadURDF(os.path.join(ASSET_DIR_PATH, 'needle/needle_40mm.urdf'),
                            (workspace_limits[0].mean(),  # TODO: scaling
                             workspace_limits[1].mean(),
                             workspace_limits[2][0] + 0.01),
                            p.getQuaternionFromEuler((0, 0, yaw)),
                            useFixedBase=False,
                            globalScaling=self.SCALING)

        # 对needle的操作全部基于inertial坐标系
        # p.resetBasePositionAndOrientation(needle_id, (workspace_limits[0].mean(),  # 重置为初始位置(inertial偏差)
        #                                              workspace_limits[1].mean(),
        #                                              workspace_limits[2][0] + 0.01),
        #                                              p.getQuaternionFromEuler((0, 0, yaw)))
        self.needle_id = needle_id
        p.changeVisualShape(needle_id, -1, specularColor=(80, 80, 80))
        self.obj_ids['rigid'].append(needle_id)  # 0
        self.obj_id, self.obj_link1 = self.obj_ids['rigid'][0], 1       # 明确needle为object

    def _get_task_obs(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """ 获得被抓取物体的抓取位置和姿态 """
        if self.has_object:
            pos, orn = get_link_pose(self.obj_id, -1)
            object_pos = np.array(pos)
            object_orn = np.array(p.getEulerFromQuaternion(np.array(orn)))
            pos, orn = get_link_pose(self.obj_id, self.obj_link1)
            waypoint_pos = np.array(pos)
            waypoint_orn = np.array(p.getEulerFromQuaternion(np.array(orn)))

        # object pos: 物体位置; waypoint pos: 抓取位置
        if self._waypoint_goal:
            achieved_pos, achieved_orn = waypoint_pos, waypoint_orn
        else:
            achieved_pos, achieved_orn = object_pos, object_orn

        if self.has_object:
            return achieved_pos, achieved_orn, object_pos, object_orn
        else:
            raise ValueError("NeedlePick task must have object!")

    def _get_obs(self) -> dict:
        tcp_pose_init = self.tcp_pose_init.copy()
        tcp_orn_init = p.getQuaternionFromEuler(tcp_pose_init[3:6])
        tcp_pose = self._get_robot_state(idx=0)     # 7-dim

        tcp_pos = tcp_pose[0:3]
        tcp_orn_quat = p.getQuaternionFromEuler(tcp_pose[3:6])
        tcp_pos_cam, _ = get_pose_in_camera_frame(tcp_pos, tcp_orn_quat, self._env_view_mat)

        _, tcp_orn_init_inv = p.invertTransform([0,0,0], tcp_orn_init)
        _, tcp_orn_tcp = p.multiplyTransforms([0,0,0], tcp_orn_init_inv, [0,0,0], tcp_orn_quat)

        # tcp position in camera frame + tcp orientation in tcp frame (first two columns) + jaw angle
        tcp_pose_hybrid = np.concatenate([
            tcp_pos_cam, tcp_orn_tcp, np.array([tcp_pose[6]]) # 3+4+1-dim
        ])

        obs_dict = {
            'robot_state': tcp_pose_hybrid.astype(np.float32),
        }

        if self.obs_type == "state":
            achieved_pos, achieved_orn, object_pos, object_orn  = self._get_task_obs()   # 3-dim, 3-dim, 3-dim, 3-dim
            object_rel_pos = object_pos - tcp_pose[0:3]      # 物体相对位置 (3-dim)

            goal = self.goal.copy()      # 任务目标位置(3-dim), 将针放到哪里去
            achieved_goal = achieved_pos.copy() # 被抓点的位置(3-dim)

            # 观测(19-dim)：机器人状态, 物体位置, 物体相对位置, waypoint位置和姿态
            observation = np.concatenate([
                tcp_pose, object_pos.ravel(), object_rel_pos.ravel(),
                achieved_pos.ravel(), achieved_orn.ravel()
            ])

            # return np.concatenate([observation.ravel(), achieved_goal.ravel(), goal.ravel()]).astype('float32')
            obs_dict.update({
                "object_pos": object_pos.astype(np.float32),
                "object_rel_pos": object_rel_pos.astype(np.float32),
                "grasping_pose": np.concatenate([achieved_pos, achieved_orn], dtype=np.float32),  # 针的抓取位置
            })
            # return observation.ravel().astype(np.float32) # (19-dim)

        if self.obs_type == "rgb":
            p.changeVisualShape(self.goal_sphere_id, -1, rgbaColor=[1, 0, 0, 1])
            env_img = self._get_env_camera_image()

            p.changeVisualShape(self.goal_sphere_id, -1, rgbaColor=[1, 0, 0, 0])
            wrist_img = self._get_wrist_camera_image(idx=0)

            obs_dict.update({
                "images": {
                    "env_cam": env_img.astype(np.uint8),
                    "wrist_cam": wrist_img.astype(np.uint8),
                }
            })

        return obs_dict

    def _sample_goal(self, random=True) -> np.ndarray:
        """ Samples a new goal and returns it.
        """
        workspace_limits = self.workspace_limits1
        if random:
            goal = np.array([workspace_limits[0].mean() + 0.01 * self.np_random.standard_normal() * self.SCALING,
                            workspace_limits[1].mean() + 0.01 * self.np_random.standard_normal() * self.SCALING,
                            workspace_limits[2][1] - 0.04 * self.SCALING])
        else:
            goal = np.array([workspace_limits[0].mean() ,
                            workspace_limits[1].mean() ,
                            workspace_limits[2][1] - 0.04 * self.SCALING])
        return goal.copy()

    def _sample_goal_callback(self):
        """ Define waypoints based on the sampled goal and needle position
        """
        super()._sample_goal_callback()
        self._waypoints = [None, None, None, None]  # four waypoints
        pos_obj, orn_obj = get_link_pose(self.obj_id, self.obj_link1)
        self._waypoint_z_init = pos_obj[2]
        orn = p.getEulerFromQuaternion(orn_obj)
        orn_eef = get_link_pose(self.psm1.body, self.psm1.EEF_LINK_INDEX)[1]
        orn_eef = p.getEulerFromQuaternion(orn_eef)
        # yaw表示针姿态中绕Z轴(世界坐标系)的旋转
        yaw = orn[2] if abs(wrap_angle(orn[2] - orn_eef[2])) < abs(wrap_angle(orn[2] + np.pi - orn_eef[2])) \
            else wrap_angle(orn[2] + np.pi)  # minimize the delta yaw

        # # for physical deployment only
        # print(" -> Needle pose: {}, {}".format(np.round(pos_obj, 4), np.round(orn_obj, 4)))
        # qs = self.psm1.get_current_joint_position()
        # joint_positions = self.psm1.inverse_kinematics(
        #     (np.array(pos_obj) + np.array([0, 0, (-0.0007 + 0.0102)]) * self.SCALING,
        #      p.getQuaternionFromEuler([-90 / 180 * np.pi, -0 / 180 * np.pi, yaw])),
        #     self.psm1.EEF_LINK_INDEX)
        # self.psm1.reset_joint(joint_positions)
        # print("qs: {}".format(joint_positions))
        # print("Cartesian: {}".format(self.psm1.get_current_position()))
        # self.psm1.reset_joint(qs)

        # four waypoints, 其中yaw要和针的方向一致(也就是从针的中间垂直向下夹持)
        self._waypoints[0] = np.array([pos_obj[0], pos_obj[1],
                                       pos_obj[2] + (-0.0007 + 0.0102 + 0.005) * self.SCALING, yaw, 0.5])  # approach
        self._waypoints[1] = np.array([pos_obj[0], pos_obj[1],
                                       pos_obj[2] + (-0.0007 + 0.0102) * self.SCALING, yaw, 0.5])  # approach
        self._waypoints[2] = np.array([pos_obj[0], pos_obj[1],
                                       pos_obj[2] + (-0.0007 + 0.0102) * self.SCALING, yaw, -0.5])  # grasp
        self._waypoints[3] = np.array([self.goal[0], self.goal[1],
                                       self.goal[2] + 0.0102 * self.SCALING, yaw, -0.5])  # lift up

    def _meet_contact_constraint_requirement(self):
        # add a contact constraint to the grasped block to make it stable
        if self._contact_approx:
            return True  # mimic the dVRL setting
        else:
            pose = get_link_pose(self.obj_id, self.obj_link1)
            return pose[0][2] > self._waypoint_z_init + 0.005 * self.SCALING

    def get_oracle_action(self, rob_pose) -> np.ndarray:
        """
        Define a human expert strategy
        根据当前姿态和waypoints的误差生成action, 使用P控制器
        根据"名义"最大速度限制每一步发送的action
        """
        # four waypoints executed in sequential order
        action = np.zeros(5)
        action[4] = -0.5
        gain = 10.  # P控制器
        # 依旧有问题，只能大致控制速度，还和FPS耦合
        MAX_DELTA_POS = 0.05 * self.SCALING / self.FPS * 1  # 每次传给机器人控制的最大位移变化
        for i, waypoint in enumerate(self._waypoints):
            if waypoint is None:
                continue
            delta_pos = (waypoint[:3] - rob_pose[:3]) / self.SCALING
            delta_yaw = (waypoint[3] - rob_pose[5]).clip(-0.4, 0.4)
            if np.linalg.norm(delta_pos) < 1e-4 and np.abs(delta_yaw) < 1e-2: # 到达当前waypoint
                self._waypoints[i] = None
            delta_pos *= gain
            if np.linalg.norm(delta_pos) > MAX_DELTA_POS:
                delta_pos *= MAX_DELTA_POS / np.linalg.norm(delta_pos)
            # if np.abs(delta_pos).max() > 1:
            #     delta_pos /= np.abs(delta_pos).max()
            # scale_factor = 0.1 # 0.4      # 控制速度的缩放比例
            # delta_pos *= scale_factor
            action = np.array([delta_pos[0], delta_pos[1], delta_pos[2], delta_yaw, waypoint[4]])
            break

        return action

class GoalNeedlePickEnv(NeedlePickEnv):
    def __init__(self, render_mode=None, cid=-1):
        super().__init__(render_mode, cid)

        original_observation_space = self.observation_space
        goal_shape = self.goal.shape

        self.observation_space = spaces.Dict({
            'observation': original_observation_space,
            'achieved_goal': spaces.Box(-np.inf, np.inf, shape=goal_shape, dtype=np.float32),
            'desired_goal': spaces.Box(-np.inf, np.inf, shape=goal_shape, dtype=np.float32),
        })

    def _create_goal_observation(self, obs):
        """ 创建 goal-based 观测 """
        observation = obs
        achieved_goal, *waypoints = self._get_task_obs()   # 物体位置
        desired_goal = self.goal         # 目标位置

        goal_observation = {
            'observation': observation,
            'achieved_goal': achieved_goal.astype('float32'),
            'desired_goal': desired_goal.astype('float32')
        }
        return goal_observation
    
    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        # 将 observation 包装成字典格式
        goal_obs = self._create_goal_observation(observation)
        return goal_obs, reward, terminated, truncated, info
    
    def reset(self, seed=None, options=None):
        observation, info = super().reset(seed=seed, options=options)

        goal_observation = self._create_goal_observation(observation)

        return goal_observation, info
    

def test(env, horizon=200):
    """
    Run the test simulation without any learning algorithm for debugging purposes
    """
    output_dir = DEBUG_DIR_PATH / "needle_pick"
    import shutil
    shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        obs, info = env.reset()
        terminated = False
        step:int = 0
        total_reward:float = 0
        data_list = []

        grasping_pos = env._get_task_obs()[0].copy()
        print(f"Needle Pos: {grasping_pos}")
        print(f"Way points:\n{env._waypoints}")

        env_cam_view_mat = np.array(env._env_view_mat).reshape(4, 4).T
        print(f"Env cam view mat:\n{np.round(env_cam_view_mat, 4)}")
        tcp_pose_init = env.tcp_pose_init.copy()
        tcp_orn_init = np.array(p.getMatrixFromQuaternion(p.getQuaternionFromEuler(tcp_pose_init[3:6]))).reshape(3, 3)
        print(f"TCP init orn: {np.round(tcp_orn_init, 6)}")

        while not terminated and step < horizon:
            tic = time.time()

            # delta pos: 相机坐标系下的位移；delta_orn: 初始状态下的旋转矩阵（只取前两列）；gripper: 开合角度
            # rob_pose = obs["robot_state"].copy()
            rob_pose = env._get_robot_state(idx=0)  # 7-dim
            action = env.get_oracle_action(rob_pose)
            print(f"\n -> step: {step}, action: {np.round(action, 6)}")
            print(f"Rob pose: {np.round(rob_pose, 6)}")

            rob_state = obs["robot_state"].copy()
            rob_pos_cam = rob_state[0:3]
            rob_orn_tip = rob_state[3:7]
            jaw_angle = rob_state[7]
            print(f" -> Rob pos (cam frame): {np.round(rob_pos_cam, 6)}")
            print(f" -> Rob orn (2 col in tcp frame): {np.round(rob_orn_tip, 6)}")
            # print(f" -> Jaw angle: {jaw_angle:.4f} rad")

            data_list.append({
                "step": step,
                "rob_pose": rob_pose.tolist(),
                "rob_pos_cam": rob_pos_cam.tolist(),
                "rob_orn_tip": rob_orn_tip.tolist(),
                "jaw_angle": float(jaw_angle)
            })

            wrist_img = obs["images"]["wrist_cam"].copy()
            env_img = obs["images"]["env_cam"].copy()
            img = Image.fromarray(wrist_img)
            img.save(output_dir / f"wrist_step_{step:03d}.png")
            img = Image.fromarray(env_img)
            img.save(output_dir / f"env_step_{step:03d}.png")

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            achieved_goal = env._get_task_obs()[0].copy()
            desired_goal = env.goal.copy()
            print(f" -> Needle Pos (Achieved): {np.round(achieved_goal, 6)}")
            print(f" -> Desired Pos:  {np.round(desired_goal, 4)}")
            print(f" -> Reward: {reward:.4f}")

            step += 1
            toc = time.time()
            # print(f" -> step time: {toc - tic:.4f} sec")
            # time.sleep(0.5)

        print("\n--- Demonstration Finished ---")
        if info.get('is_success', False):
            print("Result: ✅ Success!")
        elif truncated:
            print("Result: ⌛️ Truncated (Time Limit Reached).")
        else:
            print("Result: ❌ Failure.")
        print(f"Total steps: {step}, Total reward: {total_reward:.4f}")

    finally:
        env.close()
        traj_data_file = output_dir / "demo_trajectory.json"
        with traj_data_file.open('w') as f:
            import json
            json.dump(data_list, f, indent=4)
        print(f"Demo trajectory data saved to: {traj_data_file}")

if __name__ == "__main__":
    import sim_envs.gymnasium_utils

    # env = gym.make('NeedlePick-v0', render_mode="human")

    # env = NeedlePickEnv(render_mode="human", obs_type="rgb")  # create one process and corresponding env
    env = NeedlePickEnv(render_mode="rgb_array", obs_type="rgb")  # create one process and corresponding env

    env.reset(seed=155)
    test(env=env, horizon=800)
    # env.render()
    # time.sleep(200)
    env.close()
    time.sleep(2)