import gymnasium as gym
import torch
import numpy as np
from envs.hover_env import HoverEnv
from models.ppo_agent import PPOAgent
from configs.config import PPO_CONFIG, ENV_NAME

def inference(model_path):
    """
    加载模型并进行推理
    """
    # 初始化环境
    env = HoverEnv(render_mode="human")
    
    # 获取状态和动作维度
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    # 初始化智能体
    agent = PPOAgent(state_dim, action_dim, PPO_CONFIG)
    
    # 加载模型
    print(f"加载模型: {model_path}")
    agent.load(model_path)
    
    # 推理循环
    for i in range(5):
        state, _ = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            # 选择动作 (不使用探索)
            action = agent.select_action(state)
            
            # 执行动作
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            
            env.render()
            
        print(f"Episode {i+1} Reward: {total_reward:.2f}")
        
    env.close()

if __name__ == "__main__":
    # 这里需要指定模型路径
    model_path = "models/ppo_PyFlyt/QuadX-Hover-v3_final.pth" 
    inference(model_path)
