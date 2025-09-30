""" Needle pick task for RL
"""
import os
import time
from typing import Tuple
import numpy as np
import pybullet as p
from gymnasium import spaces

from sim_envs.gym_set.goal_env import GoalEnvWrapper
from sim_envs.utils.pybullet_utils import (
    p_step,
    get_link_pose,
    wrap_angle,
    UrdfObject
)
from sim_envs.tasks.psm_env import PsmEnv
from sim_envs.const import ASSET_DIR_PATH

class NeedlePickRL(PsmEnv):
    POSE_TRAY = ((0.55, 0, 0.6751), (0, 0, 0))
    WORKSPACE_LIMITS = ((0.50, 0.60), (-0.05, 0.05), (0.685, 0.745))  # reduce tip pad contact
    SCALING = 5.

    # TODO: grasp is sometimes not stable; check how to fix it

    def __init__(self, render_mode=None, cid=-1):
        super().__init__(render_mode, cid)
        self.current_step = 0
        self.max_episode_steps = 200  # max steps per episode
        self.needle_id = None

        self._env_setup()

        # 初始化任务目标, 如果想要读取goal, 调用 self.goal
        self.goal = self._sample_goal()
        self._sample_goal_callback()

        obs = self._get_obs()
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=obs.shape, dtype='float32')

    def step(self, action: np.ndarray):
        """
        执行一步环境交互，并返回符合 Gymnasium API 的5个值。
        """
        # 1. 预处理动作并驱动机器人、步进仿真
        #    这部分可以调用父类 PsmEnv 的辅助方法
        self.current_step += 1
        if len(action.shape) > 1:
            action = action.squeeze()
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self._set_action(action)
        p_step(self._duration)

        self._step_callback()  # 任务相关的回调函数
        obs = self._get_obs()

        terminated = False
        truncated = False
        info = {}

        achieved_goal, _ = self._get_task_obs()  # Needle position
        # 没有达到goal, 就是-1, 达到就是0
        reward = self.compute_reward(achieved_goal, self.goal, {})  # 没有用 info

        if reward == 0:
            terminated = True
            info['is_success'] = True
        else:
            info['is_success'] = False
        info['distance'] = np.linalg.norm(achieved_goal - self.goal)    # 无需缩放, scaling作用在阈值

        # 检查是否“截断” (Truncated) - 因时间限制
        if self.current_step >= self.max_episode_steps:
            truncated = True
            if 'is_success' not in info:
                info['is_success'] = False

        # 如果回合结束，填充 final_observation
        if terminated or truncated:
            # 即使回合结束，我们也获取一次“最终”的观测状态(这对于很多算法的价值函数计算至关重要)
            final_obs = self._get_obs()
            info['final_observation'] = final_obs

        # 返回符合 Gymnasium 标准的 5-元组
        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)

        self.current_step = 0

        # tray pad也需要重置

        # 暂时不随机化针的位置
        workspace_limits = self.workspace_limits1
        yaw = 0.1 * np.pi
        p.resetBasePositionAndOrientation(self.needle_id,
                                          (workspace_limits[0].mean(),
                                           workspace_limits[1].mean(),
                                           workspace_limits[2][0] + 0.01),
                                          p.getQuaternionFromEuler((0, 0, yaw)))

        p_step(0.3)  # 让针落到桌面上

        self.goal = self._sample_goal().copy()     # 目标最终位置
        self._sample_goal_callback()        # 根据needle位置设置waypoints

        return self._get_obs(), {}

    def _env_setup(self):
        """ 加载环境中需要的物体, 并进行必要的设置 """
        super(NeedlePickRL, self)._env_setup()
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
        p.resetBasePositionAndOrientation(needle_id, (workspace_limits[0].mean(),  # 重置为初始位置(inertial偏差)
                                                     workspace_limits[1].mean(),
                                                     workspace_limits[2][0] + 0.01),
                                                     p.getQuaternionFromEuler((0, 0, yaw)))
        self.needle_id = needle_id
        p.changeVisualShape(needle_id, -1, specularColor=(80, 80, 80))
        self.obj_ids['rigid'].append(needle_id)  # 0
        self.obj_id, self.obj_link1 = self.obj_ids['rigid'][0], 1       # 明确needle为object

    def _get_task_obs(self) -> Tuple[np.ndarray, np.ndarray]:
        # 获得被抓取物体的抓取位置和姿态
        if self.has_object:
            pos, orn = get_link_pose(self.obj_id, -1)
            object_pos = np.array(pos)
            object_orn = np.array(orn)
            pos, orn = get_link_pose(self.obj_id, self.obj_link1)
            waypoint_pos = np.array(pos)
            waypoint_orn = np.array(orn)
        else:
            object_pos = np.zeros(3)

        if self._waypoint_goal:
            achieved_pos, achieved_orn = waypoint_pos, waypoint_orn
        else:
            achieved_pos, achieved_orn = object_pos, object_orn

        if self.has_object:
            return achieved_pos, achieved_orn
        else:
            raise ValueError("NeedlePick task must have object!")

    def _get_obs(self) -> np.ndarray:
        robot_state = self._get_robot_state(idx=0)     # 7-dim

        object_pos, object_orn = self._get_task_obs()       # needle position (3-dim)
        object_rel_pos = object_pos - robot_state[0:3]      # 物体相对位置 (3-dim)
        waypoint_pos, waypoint_orn = np.array(object_pos), np.array(p.getEulerFromQuaternion(object_orn))       # np.array可以创建副本

        # 通过 self.goal 访问
        goal = self.goal.copy()      # 任务目标位置(3-dim), 将针放到哪里去

        # 观测包括：机器人状态, 物体位置, 物体相对位置, waypoint位置和姿态
        observation = np.concatenate([
            robot_state, object_pos.ravel(), object_rel_pos.ravel(),
            waypoint_pos.ravel(), waypoint_orn.ravel()  # achieved_goal.copy(),
        ])

        # 将所有信息拼接成一个扁平的向量并返回
        #   顺序：机器人状态, 物体位置, 目标位置
        return np.concatenate([observation, object_pos, goal])

    def _sample_goal(self) -> np.ndarray:
        """ Samples a new goal and returns it.
        """
        workspace_limits = self.workspace_limits1
        goal = np.array([workspace_limits[0].mean() + 0.01 * np.random.randn() * self.SCALING,
                         workspace_limits[1].mean() + 0.01 * np.random.randn() * self.SCALING,
                         workspace_limits[2][1] - 0.04 * self.SCALING])
        goal = np.array([workspace_limits[0].mean() ,
                         workspace_limits[1].mean() ,
                         workspace_limits[2][1] - 0.04 * self.SCALING])
        return goal.copy()

    def _sample_goal_callback(self):
        """ Define waypoints
        """
        super()._sample_goal_callback()
        self._waypoints = [None, None, None, None]  # four waypoints
        pos_obj, orn_obj = get_link_pose(self.obj_id, self.obj_link1)
        self._waypoint_z_init = pos_obj[2]
        orn = p.getEulerFromQuaternion(orn_obj)
        orn_eef = get_link_pose(self.psm1.body, self.psm1.EEF_LINK_INDEX)[1]
        orn_eef = p.getEulerFromQuaternion(orn_eef)
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
        print(f"\n -> Waypoints 2: {self._waypoints[2]}")

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
        """
        # four waypoints executed in sequential order
        action = np.zeros(5)
        action[4] = -0.5
        for i, waypoint in enumerate(self._waypoints):
            if waypoint is None:
                continue
            delta_pos = (waypoint[:3] - rob_pose[:3]) / 0.01 / self.SCALING
            delta_yaw = (waypoint[3] - rob_pose[5]).clip(-0.4, 0.4)
            if np.abs(delta_pos).max() > 1:
                delta_pos /= np.abs(delta_pos).max()
            scale_factor = 0.4      # 控制速度的缩放比例
            delta_pos *= scale_factor
            action = np.array([delta_pos[0], delta_pos[1], delta_pos[2], delta_yaw, waypoint[4]])
            if np.linalg.norm(delta_pos) * 0.01 / scale_factor < 1e-4 and np.abs(delta_yaw) < 1e-2:
                self._waypoints[i] = None
            break

        return action

def test(env, horizon=200):
    """
    Run the test simulation without any learning algorithm for debugging purposes
    """
    try:
        obs, info = env.reset()
        terminated = False
        truncated = False
        step:int = 0
        total_reward:float = 0

        while not (terminated or truncated) and step < horizon:
            tic = time.time()

            rob_pose = obs[0:7]
            action = env.get_oracle_action(rob_pose)
            print(f"\n -> step: {step}, action: {np.round(action, 4)}")
            print(f"Rob pose: {np.round(rob_pose, 4)}")

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            achieved_goal = env._get_task_obs()[0].copy()
            achieved_goal = obs[7:10].copy()
            desired_goal = env.goal.copy()
            print(f" -> Needle Pos (Achieved): {np.round(achieved_goal, 4)}")
            print(f" -> Desired Pos:  {np.round(desired_goal, 4)}")
            print(f" -> Reward: {reward:.4f}")

            step += 1
            toc = time.time()
            print(f" -> step time: {toc - tic:.4f} sec")
            time.sleep(0.5)

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

if __name__ == "__main__":
    env = NeedlePickRL(render_mode="human")  # create one process and corresponding env

    goal_based_env = GoalEnvWrapper(env)

    test(env=env, horizon=50)
    # env.render()
    # time.sleep(200)
    env.close()
    time.sleep(2)