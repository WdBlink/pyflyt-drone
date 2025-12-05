import numpy as np
import torch
import os
import glob
import time
from datetime import datetime
import matplotlib.pyplot as plt
from collections import deque

from configs.config import (
    ENV_NAME,
    ENV_CONFIG,
    PPO_CONFIG,
    TRAIN_CONFIG,
    LOG_CONFIG,
    MODEL_CONFIG
)
from envs.hover_env import HoverEnv
from models.ppo_agent import PPOAgent
from utils.logger import Logger

def train():
    """
    训练PPO智能体
    """
    # 创建日志目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{ENV_NAME}_{timestamp}"
    log_dir = os.path.join(LOG_CONFIG["log_dir"], run_name)
    os.makedirs(log_dir, exist_ok=True)
    
    # 初始化环境
    env = HoverEnv(render_mode=None)  # 训练时不渲染
    
    # 获取状态和动作维度
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    # 初始化智能体
    agent = PPOAgent(state_dim, action_dim, PPO_CONFIG)
    
    # 初始化日志记录器
    logger = Logger(log_dir)
    
    # 训练变量
    time_step = 0
    i_episode = 0
    
    # 存储训练数据
    state_history = []
    action_history = []
    reward_history = []
    
    print(f"开始训练: {run_name}")
    print(f"环境: {ENV_NAME}, 状态维度: {state_dim}, 动作维度: {action_dim}")
    
    # 训练循环
    while time_step <= TRAIN_CONFIG["max_training_timesteps"]:
        state, _ = env.reset()
        current_ep_reward = 0
        
        for t in range(1, TRAIN_CONFIG["max_ep_len"] + 1):
            # 选择动作
            action = agent.select_action(state)
            
            # 执行动作
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # 保存奖励和是否结束
            agent.buffer.rewards.append(reward)
            agent.buffer.is_terminals.append(done)
            
            time_step += 1
            current_ep_reward += reward
            
            # 更新状态
            state = next_state
            
            # 更新PPO智能体
            if time_step % TRAIN_CONFIG["update_timestep"] == 0:
                agent.update()
            
            # 打印日志和保存模型
            if time_step % TRAIN_CONFIG["log_freq"] == 0:
                print(f"Time Step: {time_step} / {TRAIN_CONFIG['max_training_timesteps']}")
            
            if time_step % TRAIN_CONFIG["save_model_freq"] == 0:
                print("保存模型...")
                agent.save(os.path.join(log_dir, f"ppo_{ENV_NAME}_{time_step}.pth"))
            
            if done:
                break
        
        i_episode += 1
        
        # 记录每个episode的奖励
        logger.log_scalar("Reward/Episode", current_ep_reward, time_step)
        
        if i_episode % TRAIN_CONFIG["print_freq"] == 0:
            print(f"Episode: {i_episode} \t Timestep: {time_step} \t Reward: {current_ep_reward:.2f}")
            
    # 保存最终模型
    print("训练结束，保存最终模型...")
    agent.save(os.path.join(log_dir, f"ppo_{ENV_NAME}_final.pth"))
    env.close()

if __name__ == "__main__":
    train()
