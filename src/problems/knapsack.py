from typing import List, Dict, Any
from src.problems.base_problem import BaseProblem
import numpy as np
from src.utils.data_augmentation import DataAugmentation
import torch

class KnapsackProblem(BaseProblem):
    def __init__(self):
        super().__init__()
        self.name = 'multi_objective_knapsack'
        self.num_categories = 5  # 5类物品
        self.items_per_category = 4  # 每类4种物品
        self.dimension = self.num_categories  # 维度为5，因为每类选一个
        self.items = self.generate_items()  # 生成初始物品
        self.augmentor = DataAugmentation(
            augmentation_methods=["noise", "scaling"],
            config={
                'noise_std': 0.1,
                'scale_range': [0.8, 1.2]
            }
        )
        # 目标函数的优化方向，True表示最大化，False表示最小化
        self.maximize_objectives = [True, False, False, False]  # 价值最大化，其他最小化
        
    def generate_items(self):
        """生成物品数据
        Returns:
            numpy.ndarray: shape (num_categories, items_per_category, 4)
                          4个属性分别为 [价值(P), 体积(R), 质量(C), 成本(S)]
        """
        items = np.random.uniform(1, 10, size=(self.num_categories, self.items_per_category, 4))
        return items
        
    def generate_instance(self, size=None):
        """生成问题实例"""
        # 使用数据增强
        data = self.items.reshape(-1, 4)  # 展平为2D数组
        augmented_data = self.augmentor.augment_batch(torch.from_numpy(data).float(), 'knapsack')
        
        # 确保增强后的数据是numpy数组
        if isinstance(augmented_data, torch.Tensor):
            selected_data = augmented_data.numpy()
        else:
            selected_data = augmented_data
            
        # 重塑回原始形状
        if len(selected_data.shape) > 2:
            selected_data = selected_data[0]
        selected_data = selected_data.reshape(self.num_categories, self.items_per_category, 4)
        
        # 确保数据在合理范围内
        self.items = np.clip(selected_data, 1, 10)
        
        # 生成一个随机初始解（每类随机选择一个物品）
        initial_solution = np.array([np.random.randint(0, self.items_per_category) for _ in range(self.num_categories)])
        
        # 将items数组reshape为(20, 4)形状，与data_loader.py中的期望一致
        reshaped_items = self.items.reshape(-1, 4)
        
        return {
            'items': reshaped_items,  # [total_items, (P,R,C,S)]
            'initial_solution': initial_solution,
            'metrics': self.compute_objectives(initial_solution),
            'valid_items': self.num_categories * self.items_per_category  # 5 categories * 4 items per category
        }
        
    def compute_objectives(self, solution):
        """计算四个目标函数值
        Args:
            solution: numpy.ndarray, shape (num_categories,)
                     每个元素表示该类别选择的物品索引
        Returns:
            numpy.ndarray: [总价值, 总体积, 总质量, 总成本]
            注意：第一个目标(总价值)要最大化，其他目标要最小化
        """
        selected_items = self.items[np.arange(self.num_categories), solution]
        total_value = np.sum(selected_items[:, 0])  # 总价值
        total_volume = np.sum(selected_items[:, 1])  # 总体积
        total_mass = np.sum(selected_items[:, 2])  # 总质量
        total_cost = np.sum(selected_items[:, 3])  # 总成本
        return np.array([total_value, total_volume, total_mass, total_cost])
        
    def is_feasible(self, solution):
        """检查解是否可行
        在这个问题中，所有解都是可行的，因为没有约束
        """
        return True
        
    def get_initial_state(self):
        """获取初始状态"""
        return {
            'items': self.items,
            'selected': np.zeros(self.num_categories, dtype=np.int32),
            'current_category': 0
        }
        
    def get_next_state(self, state, action):
        """获取下一个状态
        Args:
            state: 当前状态
            action: 在当前类别中选择的物品索引
        """
        new_state = state.copy()
        new_state['selected'][state['current_category']] = action
        new_state['current_category'] += 1
        return new_state

    def generate_instance_from_data(self, data):
        """从已有数据生成实例"""
        if len(data.shape) != 3:
            raise ValueError("数据应该是一个三维数组：(类别数, 每类物品数, 属性数)")
            
        self.items = data
        initial_solution = np.array([np.random.randint(0, self.items_per_category) for _ in range(self.num_categories)])
        
        return {
            'items': self.items,
            'initial_solution': initial_solution,
            'metrics': self.compute_objectives(initial_solution)
        }

    def is_pareto_dominating(self, solution1, solution2):
        """判断solution1是否帕累托支配solution2
        Args:
            solution1: 第一个解
            solution2: 第二个解
        Returns:
            bool: 如果solution1支配solution2则返回True
        """
        obj1 = self.compute_objectives(solution1)
        obj2 = self.compute_objectives(solution2)
        
        # 检查是否至少在一个目标上严格更好
        at_least_one_better = False
        # 检查每个目标
        for i, (o1, o2) in enumerate(zip(obj1, obj2)):
            if self.maximize_objectives[i]:
                # 如果是最大化目标，solution1应该更大
                if o1 < o2:  # solution1更差
                    return False
                if o1 > o2:  # solution1更好
                    at_least_one_better = True
            else:
                # 如果是最小化目标，solution1应该更小
                if o1 > o2:  # solution1更差
                    return False
                if o1 < o2:  # solution1更好
                    at_least_one_better = True
        
        return at_least_one_better

    def get_solution_details(self, solution):
        """获取解的详细信息
        Args:
            solution: numpy.ndarray, shape (num_categories,)
                     每个类别选择的物品索引
        Returns:
            dict: 包含解的详细信息
        """
        if not self.validate_solution(solution):
            raise ValueError("无效的解")

        selected_items = self.items[np.arange(self.num_categories), solution]
        objectives = self.compute_objectives(solution)
        
        return {
            'solution': solution.tolist(),
            'selected_items': selected_items.tolist(),
            'objectives': {
                'total_value': objectives[0],
                'total_volume': objectives[1],
                'total_mass': objectives[2],
                'total_cost': objectives[3]
            },
            'per_category': [
                {
                    'category': i,
                    'selected_item': int(solution[i]),
                    'value': float(selected_items[i, 0]),
                    'volume': float(selected_items[i, 1]),
                    'mass': float(selected_items[i, 2]),
                    'cost': float(selected_items[i, 3])
                }
                for i in range(self.num_categories)
            ]
        }

    def validate_solution(self, solution):
        """验证解的有效性
        Args:
            solution: numpy.ndarray, 解决方案
        Returns:
            bool: 解是否有效
        """
        # 检查解的维度
        if not isinstance(solution, np.ndarray) or solution.shape != (self.num_categories,):
            return False
        
        # 检查每个类别的选择是否有效
        if not np.all((solution >= 0) & (solution < self.items_per_category)):
            return False
        
        # 检查解的类型
        if not np.issubdtype(solution.dtype, np.integer):
            return False
            
        return True

    def get_random_solution(self):
        """生成一个随机解
        Returns:
            numpy.ndarray: 随机解
        """
        return np.array([np.random.randint(0, self.items_per_category) 
                        for _ in range(self.num_categories)])
