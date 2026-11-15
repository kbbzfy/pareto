import numpy as np
import logging
from typing import Dict, List, Tuple, Any
from src.environments.base_env import BaseEnv
import gym
from gym import spaces

logger = logging.getLogger(__name__)

class KnapsackEnv(BaseEnv):
    def __init__(self, problem):
        super().__init__(problem)
        self.num_categories = problem.num_categories
        self.items_per_category = problem.items_per_category
        self.num_objectives = problem.num_objectives
        
        logger.info(f"初始化KnapsackEnv:")
        logger.info(f"类别数: {self.num_categories}")
        logger.info(f"每类物品数: {self.items_per_category}")
        logger.info(f"目标函数数量: {self.num_objectives}")
        
        # 定义动作空间（每个类别中选择哪个物品）
        self.action_space = spaces.Discrete(self.items_per_category)
        
        # 定义观察空间
        self.observation_space = spaces.Dict({
            'items': spaces.Box(
                low=0, 
                high=np.inf, 
                shape=(self.num_categories, self.items_per_category, self.num_objectives),
                dtype=np.float32
            ),
            'current_category': spaces.Discrete(self.num_categories + 1),
            'action_mask': spaces.MultiBinary(self.items_per_category),
            'selected': spaces.Box(
                low=0, 
                high=self.items_per_category-1, 
                shape=(self.num_categories,), 
                dtype=np.int32
            ),
            'remaining_capacity': spaces.Box(
                low=0,
                high=np.inf,
                shape=(self.num_objectives,),
                dtype=np.float32
            )
        })
        
        # 从problem获取目标名称
        self.objective_names = getattr(problem, 'objective_names', 
            ['objective_1'] + [f'objective_{i+1}' for i in range(1, self.num_objectives)])
        
        logger.info(f"目标函数名称: {self.objective_names}")
        
        # 验证物品数据的维度
        if not hasattr(problem, 'items'):
            raise ValueError("Problem对象缺少items属性")
        
        items_shape = problem.items.shape
        expected_shape = (self.num_categories, self.items_per_category, self.num_objectives)
        
        # 检查物品数据维度是否匹配，如果不匹配则调整
        if items_shape != expected_shape:
            logger.warning(f"物品数据维度不匹配: 期望{expected_shape}，实际{items_shape}")
            
            # 创建新的物品数组
            new_items = np.zeros(expected_shape)
            
            # 拷贝可能的数据
            min_categories = min(items_shape[0], expected_shape[0])
            min_items = min(items_shape[1], expected_shape[1])
            min_objectives = min(items_shape[2], expected_shape[2])
            
            # 复制已有数据
            new_items[:min_categories, :min_items, :min_objectives] = \
                problem.items[:min_categories, :min_items, :min_objectives]
            
            # 如果物品数据的目标数量少于环境期望的目标数量，随机生成额外的目标值
            if items_shape[2] < expected_shape[2]:
                for i in range(items_shape[2], expected_shape[2]):
                    # 随机生成与第一个目标类似范围的值
                    base_values = problem.items[:min_categories, :min_items, 0]
                    mean_value = np.mean(base_values)
                    std_value = np.std(base_values) or mean_value * 0.2  # 防止标准差为0
                    
                    new_items[:min_categories, :min_items, i] = \
                        np.random.normal(mean_value, std_value, (min_categories, min_items))
            
            # 更新problem的物品数据
            problem.items = new_items
            logger.info(f"物品数据已调整为维度: {new_items.shape}")
        
        logger.info(f"物品数据维度: {problem.items.shape}")
        
        minimum_totals = problem.items.min(axis=1).sum(axis=0)
        for name, minimum, capacity in zip(self.objective_names, minimum_totals, problem.capacities):
            if minimum > capacity:
                raise ValueError(
                    f"背包实例无可行解：{name} 最少需要 {minimum:.6g}，容量上限为 {capacity:.6g}"
                )

        self.reset()
        
    def reset(self):
        """重置环境"""
        try:
            self.current_state = self.problem.get_initial_state()
            self.episode_rewards = []
            self.episode_steps = 0
            
            # 初始化所有目标的当前值
            self.current_objectives = np.zeros(self.num_objectives)
            if not self.get_valid_actions():
                raise ValueError("背包第一类物品没有同时满足所有容量上限的选项")
            
            state = self.get_state_representation()
            logger.info(f"环境重置成功，状态形状: {state['items'].shape}")
            return state
            
        except Exception as e:
            logger.error(f"重置环境时出错: {str(e)}")
            raise
        
    def step(self, action):
        """执行动作
        Args:
            action: int, 在当前类别中选择的物品索引
        """
        try:
            # 验证动作的有效性
            if not isinstance(action, (int, np.integer)):
                raise ValueError(f"动作必须是整数，实际类型: {type(action)}")
            if action < 0 or action >= self.items_per_category:
                raise ValueError(f"动作超出范围[0, {self.items_per_category}): {action}")
            
            if action not in self.get_valid_actions():
                raise ValueError(f"动作 {action} 违反背包容量约束或回合已结束")

            self.episode_steps += 1
            current_category = self.current_state['current_category']
            self.current_objectives += self.current_state['items'][current_category, action]
            self.current_state = self.problem.get_next_state(self.current_state, action)

            complete = self.current_state['current_category'] == self.num_categories
            dead_end = not complete and not self.get_valid_actions()
            done = complete or dead_end
            reward = self.compute_reward(complete) - float(dead_end)

            self.episode_rewards.append(reward)
            
            # 获取下一个状态
            next_state = self.get_state_representation()
            
            if self.episode_steps % 10 == 0:  # 每10步打印一次状态
                logger.info(f"步骤 {self.episode_steps}:")
                logger.info(f"动作: {action}")
                logger.info(f"奖励: {reward}")
                logger.info(f"是否完成: {done}")
                logger.info(f"当前目标值: {self.current_objectives}")
            
            return next_state, reward, done, {"feasible": complete}
            
        except Exception as e:
            logger.error(f"执行动作时出错: {str(e)}")
            raise

    def get_state_representation(self):
        """获取状态表示"""
        try:
            remaining_capacities = np.maximum(0, self.problem.capacities - self.current_objectives)
            action_mask = np.zeros(self.items_per_category, dtype=np.int8)
            action_mask[self.get_valid_actions()] = 1

            items = self.current_state['items'].astype(np.float32)
            
            # 验证数据
            if items.shape != (self.num_categories, self.items_per_category, self.num_objectives):
                raise ValueError(f"物品数据维度错误: {items.shape}")
            
            return {
                'items': items,
                'current_category': self.current_state['current_category'],
                'selected': self.current_state['selected'],
                'action_mask': action_mask,
                'remaining_capacity': np.array(remaining_capacities, dtype=np.float32)
            }
            
        except Exception as e:
            logger.error(f"获取状态表示时出错: {str(e)}")
            raise
        
    @property
    def capacities(self):
        return self.problem.capacities

    def get_valid_actions(self):
        """返回当前类别中满足所有容量上限的物品索引。"""
        category = self.current_state['current_category']
        if category >= self.num_categories:
            return []
        totals = self.current_objectives + self.current_state['items'][category]
        return np.flatnonzero(np.all(totals <= self.problem.capacities, axis=1)).tolist()

    def compute_reward(self, done):
        """奖励累计价值，对完成全部类别的可行选择额外奖励。"""
        value_ratio = self.current_objectives[0] / (self.problem.items[:, :, 0].sum() + 1e-6)
        return value_ratio + float(done)
