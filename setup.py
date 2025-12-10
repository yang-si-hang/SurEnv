from setuptools import setup, find_packages

setup(
    name='SurEnv',  # 包名
    version='0.0.0',
    packages=find_packages(), # 自动查找包内所有模块
    # packages=find_packages(include=['sim_envs'])
)