""" Wrapper to convert a standard Gym environment into a GoalEnv-like environment. """
import numpy as np
import gymnasium as gym
from gymnasium import spaces

class GoalEnvWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

        original_observation_space = env.observation_space
        goal_shape = env.goal.shape

        self.observation_space = spaces.Dict({
            'observation': original_observation_space,
            'achieved_goal': spaces.Box(-np.inf, np.inf, shape=goal_shape, dtype=np.float32),
            'desired_goal': spaces.Box(-np.inf, np.inf, shape=goal_shape, dtype=np.float32),
        })

    def _create_goal_observation(self, obs):
        # 创建 goal-based 观测
        observation = obs
        achieved_goal, _ = self.env._get_task_obs()   # 物体位置
        desired_goal = self.env.goal         # 目标位置

        goal_observation = {
            'observation': observation,
            'achieved_goal': achieved_goal,
            'desired_goal': desired_goal
        }
        return goal_observation

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        # 将 observation 包装成字典格式
        goal_obs = self._create_goal_observation(observation)
        return goal_obs, reward, terminated, truncated, info
    
    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)

        goal_observation = self._create_goal_observation(observation)

        return goal_observation, info