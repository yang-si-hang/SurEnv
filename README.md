## 文件结构说明

```
SurEnv/
├── sim_envs/
│   ├── assets/     # 几何和文件资产
│   ├── gymnasium_utils/    # 环境设置
│   │   └── sur_env.py  # 环境的基类
│   ├── robots/     # 机器人类
│   ├── tasks/      # 任务类
│   └── utils/      # 辅助函数
├── diffusion_policy/   # dp 算法
├── act/   # ACT 算法
```

## 生成数据

- 用脚本策略生成演示数据：
`sim_envs/data/data_gen.py`

- 转换为zarr格式，用于Diffusion Policy算法：
`sim_envs/data/convert_zarr.py`

## 

- emoj网站： 
https://getemoji.com/