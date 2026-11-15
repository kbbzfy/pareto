import numpy as np
from src.environments.base_env import BaseEnv
import gym
from gym import spaces
import logging

logger = logging.getLogger(__name__)

class TSPEnv(BaseEnv):
    def __init__(self, problem):
        super().__init__(problem)
        # 初始化日志记录器
        self.logger = logging.getLogger(__name__)
        
        # 确保在使用前设置size
        instance = problem.generate_instance(problem.dimension)
        self.size = len(instance['coordinates'])
        
        # 从问题获取目标名称和类型
        self.num_objectives = getattr(problem, 'num_objectives', 4)  # 默认4个目标
        self.objective_names = getattr(problem, 'objective_names', 
                                     ["总距离", "时间窗口", "平衡性", "成本"])  # 默认4个目标名称
        self.objective_types = getattr(problem, 'objective_types', 
                                     [-1] * self.num_objectives)  # 默认全部最小化
        
        # 确保目标数量和名称列表长度一致
        if len(self.objective_names) < self.num_objectives:
            # 如果名称不足，用默认名称补充
            for i in range(len(self.objective_names), self.num_objectives):
                self.objective_names.append(f"目标_{i+1}")
        
        # 确保目标类型列表长度一致
        if len(self.objective_types) < self.num_objectives:
            # 如果类型不足，用-1（最小化）补充
            self.objective_types.extend([-1] * (self.num_objectives - len(self.objective_types)))
        
        logger.info(f"TSPEnv初始化成功:")
        logger.info(f"城市数量: {self.size}")
        logger.info(f"目标数量: {self.num_objectives}")
        logger.info(f"目标名称: {self.objective_names}")
        logger.info(f"目标类型: {self.objective_types}")
        
        # 定义动作空间（选择下一个城市）
        self.action_space = spaces.Discrete(self.size)
        
        # 定义观察空间
        self.observation_space = spaces.Dict({
            'coordinates': spaces.Box(low=0, high=1, shape=(self.size, 2), dtype=np.float32),
            'current_pos': spaces.Discrete(self.size),
            'visited_mask': spaces.Box(low=0, high=1, shape=(self.size,), dtype=np.int32),
            'distance_matrix': spaces.Box(low=0, high=np.inf, shape=(self.size, self.size), dtype=np.float32)
        })
        
        self.reset()
        
    def reset(self):
        """重置环境到初始状态"""
        # 获取问题实例数据
        instance = self.problem.generate_instance(self.size)
        self.coordinates = instance['coordinates']
        self.distance_matrix = instance['distance_matrix']
        
        # 初始化状态
        self.visited = np.zeros(self.size, dtype=np.int32)
        self.tour = []
        self.episode_rewards = []
        self.episode_steps = 0
        
        # 随机选择起始城市
        start_city = np.random.randint(0, self.size)
        self.visited[start_city] = 1
        self.tour.append(start_city)
        
        return self.get_state_representation()
        
    def step(self, action):
        """执行动作"""
        self.episode_steps += 1
        
        # 获取有效动作列表
        valid_actions = self.get_valid_actions()
        
        # 检查动作是否有效
        if action not in valid_actions:
            # 动作无效，返回负面奖励但不终止回合
            self.logger.warning(f"无效动作: {action}，有效动作: {valid_actions}")
            
            # 如果允许，自动选择一个有效动作继续
            if len(valid_actions) > 0:
                # 自动选择一个有效动作（增加随机性以促进探索）
                action = np.random.choice(valid_actions)
                self.logger.info(f"自动选择有效动作: {action}")
            else:
                # 没有有效动作，终止回合
                # 注意：这种情况通常只在所有城市都已访问后发生
                self.logger.warning("没有有效动作可选，终止回合")
                return self.get_state_representation(), -1.0, True, {'invalid_action': True, 'message': '无可用动作'}
        
        # 更新状态
        current_city = self.tour[-1]
        distance = self.distance_matrix[current_city][action]
        self.visited[action] = 1
        self.tour.append(action)
        
        # 判断是否完成
        done = len(self.tour) == self.size
        
        if done:
            # 添加回到起点的路径
            self.tour.append(self.tour[0])
            # 计算多目标函数值
            objectives = self.problem.compute_objectives(np.array(self.tour))
            # 基于多目标计算奖励
            reward = self.compute_reward(objectives)
            # 记录完成路径的总长度和目标值
            self.logger.info(f"回合完成：总步数={self.episode_steps}, 路径长度={len(self.tour)}, 目标值={objectives}")
        else:
            # 使用平滑的局部奖励指导探索
            remaining_cities = self.size - len(self.tour)
            progress = min(1.0, self.episode_steps / self.size)
            
            # 平滑因子：避免早期奖励波动过大
            smooth_factor = 1.0 + progress * 2.0  # 随进度从1.0增加到3.0
            
            # 动态调整奖励权重
            early_weight = 0.5  # 早期探索权重
            late_weight = 1.5    # 后期优化权重
            weight = early_weight + (late_weight - early_weight) * progress
            
            # 计算平滑奖励
            reward = -distance * weight / (remaining_cities * smooth_factor)
            
        self.episode_rewards.append(reward)
        return self.get_state_representation(), reward, done, {}
        
    def get_state_representation(self):
        """获取状态表示"""
        return {
            'coordinates': self.coordinates.astype(np.float32),
            'current_pos': self.tour[-1],
            'visited_mask': self.visited,
            'distance_matrix': self.distance_matrix.astype(np.float32)
        }
        
    def get_valid_actions(self):
        """获取有效动作"""
        return np.where(self.visited == 0)[0]
        
    def compute_reward(self, objectives):
        """计算奖励，基于多个目标函数
        
        Args:
            objectives: 目标函数值数组
            
        Returns:
            float: 综合奖励值
        """
        # 规模感知参数：基于城市数量调整
        scale_factor = np.log(1 + self.size / 5.0)  # 以5城市为基准
        
        # 对于最小化目标，数值越小越好（奖励越高）
        # 对于最大化目标，数值越大越好（奖励越高）
        reward = 0.0
        
        # 动态权重调整：基于当前步数平滑变化
        progress = min(1.0, self.episode_steps / self.size)
        
        # 基础权重设置（考虑规模因素）
        base_weights = [2.0 * scale_factor, 0.5, 0.3, 0.2]  # 前4个目标的权重
        default_weight = 0.1  # 其他目标的默认权重
        
        # 更平滑的权重过渡
        weights = [w * (0.3 + 0.7 * np.tanh(progress * 3)) for w in base_weights]
        
        # 根据目标类型计算奖励贡献
        for i, obj_value in enumerate(objectives):
            # 获取当前目标的类型和权重
            obj_type = -1  # 默认最小化
            if i < len(self.objective_types):
                obj_type = self.objective_types[i]
                
            weight = default_weight
            if i < len(weights):
                weight = weights[i]
            
            # 改进的平滑归一化处理（考虑规模因素）
            normalized_value = 0.0
            scale_param = 10.0 / scale_factor  # 根据规模调整平滑参数
            
            if obj_type > 0:  # 最大化目标
                # 使用改进的sigmoid函数
                normalized_value = 1.0 / (1.0 + np.exp(-obj_value/scale_param))
            else:  # 最小化目标
                # 使用改进的指数衰减
                normalized_value = np.exp(-obj_value/scale_param)
            
            # 加入权重调整
            reward += normalized_value * weight
        
        # 改进的进度奖励（考虑规模因素）
        if len(self.tour) == self.size:
            reward += 1.0 * scale_factor  # 完成路径的额外奖励
        
        return reward / scale_factor  # 最终归一化
    
    def render(self):
        """渲染当前状态（可选）"""
        pass
