import numpy as np
from typing import Dict, Any, List, Tuple
import torch

class ProblemAbstraction:
    def __init__(self, max_nodes: int):
        self.max_nodes = max_nodes
        
    def transform_problem(self, data: Dict) -> Tuple[torch.Tensor, Dict]:
        """
        将问题数据转换为模型可用的格式
        
        Args:
            data: 原始问题数据
            
        Returns:
            transformed_data: 转换后的张量
            problem_info: 问题相关信息
        """
        # 根据数据结构判断问题类型
        if 'type' in data:
            problem_type = data['type']
        else:
            # 通过检查数据结构判断问题类型
            if 'items' in data and isinstance(data['items'], list):
                problem_type = 'knapsack'
            elif 'coordinates' in data or 'distance_matrix' in data:
                problem_type = 'tsp'
            else:
                raise ValueError(f"无法确定问题类型: {data.keys()}")
                
        if problem_type == 'knapsack':
            return self._transform_knapsack(data)
        else:
            return self._transform_tsp(data)
            
    def _transform_knapsack(self, data: Dict) -> Tuple[torch.Tensor, Dict]:
        """转换背包问题数据"""
        # 检查数据格式
        if 'items' not in data:
            raise ValueError(f"背包问题数据缺少'items'字段: {data.keys()}")
            
        items = data['items']
        
        # 处理不同格式的items数据
        if isinstance(items[0], list) and isinstance(items[0][0], list):
            # 格式为[[[值,体积,质量],...],...]的3D数组
            items_array = np.array(items)
            # 将3D数组平铺为2D特征矩阵
            n_categories = items_array.shape[0]
            items_per_category = items_array.shape[1]
            n_items = n_categories * items_per_category
            
            # 创建特征矩阵：[价值, 体积, 质量, 类别]
            features = torch.zeros(n_items, 4)
            for cat_idx in range(n_categories):
                for item_idx in range(items_per_category):
                    idx = cat_idx * items_per_category + item_idx
                    features[idx, 0] = items_array[cat_idx, item_idx, 0]  # 价值
                    features[idx, 1] = items_array[cat_idx, item_idx, 1]  # 体积
                    features[idx, 2] = items_array[cat_idx, item_idx, 2]  # 质量
                    features[idx, 3] = cat_idx  # 类别
                    
            # 检查是否有容量约束
            if 'metrics' in data and isinstance(data['metrics'], dict):
                metrics = data['metrics']
                volume_capacity = metrics.get('total_volume', 0) * 1.5  # 默认容量为总体积的1.5倍
                mass_capacity = metrics.get('total_mass', 0) * 1.5  # 默认容量为总质量的1.5倍
            else:
                # 没有明确的容量，使用物品总体积和质量的一半
                volume_capacity = np.sum(items_array[:, :, 1]) * 0.5
                mass_capacity = np.sum(items_array[:, :, 2]) * 0.5
                
            # 如果有初始解决方案，添加到问题信息中
            initial_solution = data.get('initial_solution', None)
                
            return features, {
                'volume_capacity': volume_capacity, 
                'mass_capacity': mass_capacity,
                'initial_solution': initial_solution,
                'n_categories': n_categories,
                'items_per_category': items_per_category
            }
        else:
            # 传统格式，每个物品是一个字典
            n_items = len(items)
            
            # 创建特征矩阵：[weight, value]
            features = torch.zeros(n_items, 2)
            for i, item in enumerate(items):
                if isinstance(item, dict):
                    features[i, 0] = item.get('weight', 0)
                    features[i, 1] = item.get('value', 0)
                    
            # 添加容量约束
            capacity = data.get('capacity', sum(item.get('weight', 0) for item in items) * 0.5)
            
            return features, {'capacity': capacity}
