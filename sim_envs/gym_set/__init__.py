""" 注册自定义的Gym环境
"""

from gymnasium.envs.registration import register

register(
    id='NeedlePick-v0',                            # 环境的唯一ID
    entry_point='sim_envs.tasks.needle_pick:NeedlePickEnv',  # 入口点，格式为 '包名.模块名:类名'
    max_episode_steps=100,                         # (可选) 每回合最大步数
)