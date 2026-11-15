from abc import ABC, abstractmethod
import gym
from gym import spaces
import numpy as np

class BaseEnv(gym.Env, ABC):
    def __init__(self, problem):
        super().__init__()
        self.problem = problem
        self.episode_rewards = []
        self.episode_steps = 0
        
    @abstractmethod
    def reset(self):
        """重置环境"""
        pass
        
    @abstractmethod
    def step(self, action):
        """执行动作"""
        pass
        
    def render(self, mode='human'):
        """渲染环境（可选）"""
        pass
        
    @abstractmethod
    def get_valid_actions(self):
        """获取有效动作"""
        pass
        
    @abstractmethod
    def compute_reward(self, objectives):
        """计算奖励"""
        pass
    
    def get_episode_stats(self):
        """获取当前回合的统计信息"""
        return {
            'episode_length': self.episode_steps,
            'episode_reward': sum(self.episode_rewards),
            'average_reward': np.mean(self.episode_rewards) if self.episode_rewards else 0
        }
    
    @abstractmethod
    def get_state_representation(self):
        """获取状态的神经网络输入表示"""
        pass 