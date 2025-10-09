""" 展示采集到的轨迹数据
"""
import os
import gymnasium as gym
import numpy as np
from sim_envs.const import DATA_DIR_PATH

def main(env_name:str):
    env = gym.make(env_name, render_mode='human')
    env.reset()

    # TODO: not implemented


if __name__ == "__main__":
    main()