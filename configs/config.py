# 环境配置
ENV_NAME = "PyFlyt/QuadX-Hover-v3"

ENV_CONFIG = {
    "render_mode": "human"
}

# PPO超参数
PPO_CONFIG = {
    "lr_actor": 0.0003,
    "lr_critic": 0.001,
    "gamma": 0.99,
    "k_epochs": 4,
    "eps_clip": 0.2,
    "hidden_dim": 64
}

# 训练配置
TRAIN_CONFIG = {
    "max_training_timesteps": 1000000,
    "max_ep_len": 1000,
    "update_timestep": 2000,
    "log_freq": 2000,
    "save_model_freq": 10000,
    "print_freq": 10
}

# 日志配置
LOG_CONFIG = {
    "log_dir": "logs"
}

# 模型保存配置
MODEL_CONFIG = {
    "save_dir": "models"
}
