import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
from collections import deque
import os

class Actor(nn.Module):
    """
    Actor网络：根据状态输出动作的均值
    """
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, state):
        mean = self.net(state)
        std = torch.exp(self.log_std)
        return mean, std

class Critic(nn.Module):
    """
    Critic网络：评估状态的价值
    """
    def __init__(self, state_dim, hidden_dim=64):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state):
        return self.net(state)

class PPOBuffer:
    """
    经验回放缓冲区
    """
    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.is_terminals = []

    def clear(self):
        del self.states[:]
        del self.actions[:]
        del self.log_probs[:]
        del self.rewards[:]
        del self.is_terminals[:]

class PPOAgent:
    """
    PPO算法智能体
    """
    def __init__(self, state_dim, action_dim, config):
        self.lr_actor = config["lr_actor"]
        self.lr_critic = config["lr_critic"]
        self.gamma = config["gamma"]
        self.k_epochs = config["k_epochs"]
        self.eps_clip = config["eps_clip"]
        self.device = torch.device('cpu')
        
        self.buffer = PPOBuffer()
        
        self.policy = Actor(state_dim, action_dim, config["hidden_dim"]).to(self.device)
        self.optimizer = optim.Adam([
            {'params': self.policy.parameters(), 'lr': self.lr_actor}
        ])
        
        self.policy_old = Actor(state_dim, action_dim, config["hidden_dim"]).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.critic = Critic(state_dim, config["hidden_dim"]).to(self.device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr_critic)
        
        self.MseLoss = nn.MSELoss()

    def select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device)
            mean, std = self.policy_old(state)
            dist = Normal(mean, std)
            action = dist.sample()
            action_logprob = dist.log_prob(action)
            
        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.log_probs.append(action_logprob)
        
        return action.detach().cpu().numpy().flatten()

    def update(self):
        # 蒙特卡洛估计回报
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        # 归一化回报
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)
        
        # 将列表转换为张量
        old_states = torch.stack(self.buffer.states).to(self.device).detach()
        old_actions = torch.stack(self.buffer.actions).to(self.device).detach()
        old_log_probs = torch.stack(self.buffer.log_probs).to(self.device).detach()
        
        # 优化策略 k 次
        for _ in range(self.k_epochs):
            # 评估旧动作和值
            mean, std = self.policy(old_states)
            dist = Normal(mean, std)
            log_probs = dist.log_prob(old_actions)
            state_values = self.critic(old_states)
            
            # 计算比率 (pi_theta / pi_theta__old)
            ratios = torch.exp(log_probs - old_log_probs.detach())
            
            # 计算代理损失
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            
            # 最终损失
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist.entropy()
            
            # 梯度下降步骤
            self.optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            self.critic_optimizer.step()
            
        # 复制新权重到旧策略
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        # 清除缓冲区
        self.buffer.clear()

    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
