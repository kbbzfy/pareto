import numpy as np
from typing import List, Tuple, Dict, Any
import torch
import random

class DataAugmentation:
    def __init__(self, augmentation_methods: List[str], config: Dict[str, Any] = None):
        """
        初始化数据增强模块
        
        Args:
            augmentation_methods: 数据增强方法列表
            config: 配置参数
        """
        self.methods = augmentation_methods
        self.config = config or {
            'noise_std': 0.1,
            'scale_range': [0.8, 1.2],
            'rotation_angles': [90, 180, 270],
            'swap_prob': 0.2
        }
        
    def augment(self, data: np.ndarray, problem_type: str) -> Tuple[np.ndarray, Dict]:
        """
        对数据进行增强
        
        Args:
            data: 输入数据 shape: (n_items, 3) for knapsack [价值,体积,质量]
            problem_type: 问题类型
            
        Returns:
            augmented_data: 增强后的数据列表
            info: 增强信息
        """
        if problem_type == 'knapsack':
            # 应用噪声
            noise = np.random.normal(0, self.config['noise_std'], data.shape)
            noisy_data = np.clip(data + noise, 1, 10)
            
            # 应用缩放
            scale = np.random.uniform(*self.config['scale_range'])
            scaled_data = np.clip(data * scale, 1, 10)
            
            # 对于3D数据，我们不进行旋转，而是进行随机扰动
            perturbed_data = data.copy()
            # 随机选择30%的数据点进行扰动
            mask = np.random.random(data.shape) < 0.3
            perturbed_data[mask] *= np.random.uniform(0.9, 1.1, size=np.sum(mask))
            perturbed_data = np.clip(perturbed_data, 1, 10)
            
            # 返回原始数据和三种增强后的数据
            augmented_data = np.stack([data, noisy_data, scaled_data, perturbed_data])
            
            return augmented_data, {'augmentation_types': ['original', 'noise', 'scale', 'perturb']}
        else:
            return data, {}
        
    def _rotate_coordinates(self, data: np.ndarray) -> np.ndarray:
        """
        旋转城市坐标（仅适用于TSP）
        """
        rotated_data = []
        for angle in self.config['rotation_angles']:
            rad = np.radians(angle)
            rot_matrix = np.array([
                [np.cos(rad), -np.sin(rad)],
                [np.sin(rad), np.cos(rad)]
            ])
            rotated = np.dot(data.reshape(-1, 2), rot_matrix)
            rotated_data.append(rotated.reshape(data.shape))
        return np.concatenate(rotated_data)
        
    def _add_noise(self, data: np.ndarray) -> np.ndarray:
        """
        添加高斯噪声
        """
        noise = np.random.normal(0, self.config['noise_std'], data.shape)
        return data + noise
        
    def _scale_data(self, data: np.ndarray) -> np.ndarray:
        """
        尺度变换
        """
        scaled_data = []
        for scale in np.linspace(self.config['scale_range'][0], 
                               self.config['scale_range'][1], 3):
            scaled_data.append(data * scale)
        return np.concatenate(scaled_data)
        
    def _swap_cities(self, data: np.ndarray) -> np.ndarray:
        """
        随机交换城市位置（仅适用于TSP）
        """
        swapped_data = data.copy()
        n_cities = data.shape[0] // 2  # 假设数据是成对的坐标
        
        # 随机选择城市对进行交换
        for i in range(n_cities):
            if np.random.random() < self.config['swap_prob']:
                j = np.random.randint(n_cities)
                swapped_data[i*2:(i+1)*2], swapped_data[j*2:(j+1)*2] = \
                    swapped_data[j*2:(j+1)*2].copy(), swapped_data[i*2:(i+1)*2].copy()
                    
        return swapped_data
        
    def augment_batch(self, batch_data: torch.Tensor, problem_type: str) -> np.ndarray:
        """对批量数据进行增强"""
        numpy_data = batch_data.numpy()
        aug_data, _ = self.augment(numpy_data, problem_type)
        return aug_data

class DataAugmentor:
    """数据增强器,用于背包问题和TSP问题的数据增强"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.noise_std = config['data_augmentation'].get('noise_std', 0.1)
        self.scale_range = config['data_augmentation'].get('scale_range', [0.8, 1.2])
        
    def augment_knapsack_data(self, data: Dict) -> List[Dict]:
        """对背包问题数据进行增强"""
        augmented_data = []
        items = data['items']
        
        # 检查数据格式，适配不同格式
        if 'capacity' in data:
            capacity = data['capacity']
        elif 'metrics' in data and isinstance(data['metrics'], dict):
            metrics = data['metrics']
            # 使用总体积作为容量
            capacity = metrics.get('total_volume', 0)
        else:
            # 如果没有容量信息，跳过增强
            return [data]
        
        # 1. 添加高斯噪声
        try:
            noisy_items = self._add_gaussian_noise(items)
            augmented_data.append({
                'items': noisy_items,
                'capacity': capacity
            })
        except Exception as e:
            print(f"添加噪声时出错: {e}")
        
        # 2. 缩放特征
        try:
            scaled_items = self._scale_features(items)
            augmented_data.append({
                'items': scaled_items,
                'capacity': capacity
            })
        except Exception as e:
            print(f"缩放特征时出错: {e}")
            
        # 3. 交换特征
        try:
            swapped_items = self._swap_features(items)
            augmented_data.append({
                'items': swapped_items,
                'capacity': capacity
            })
        except Exception as e:
            print(f"交换特征时出错: {e}")
            
        # 4. 局部扰动
        try:
            perturbed_items = self._local_perturbation(items)
            augmented_data.append({
                'items': perturbed_items,
                'capacity': capacity
            })
        except Exception as e:
            print(f"局部扰动时出错: {e}")
            
        return augmented_data
        
    def _add_gaussian_noise(self, items: List[Dict]) -> List[Dict]:
        """添加高斯噪声"""
        noisy_items = []
        for item in items:
            weight = max(0.1, item['weight'] + np.random.normal(0, self.noise_std))
            value = max(0.1, item['value'] + np.random.normal(0, self.noise_std))
            noisy_items.append({
                'weight': weight,
                'value': value
            })
        return noisy_items
        
    def _scale_features(self, items: List[Dict]) -> List[Dict]:
        """特征缩放"""
        scale = np.random.uniform(*self.scale_range)
        scaled_items = []
        for item in items:
            scaled_items.append({
                'weight': item['weight'] * scale,
                'value': item['value'] * scale
            })
        return scaled_items
        
    def _swap_features(self, items: List[Dict]) -> List[Dict]:
        """特征交换增强"""
        swapped_items = []
        n = len(items)
        # 生成要交换的索引对
        n_swaps = n // 4  # 交换1/4的物品
        swap_pairs = np.random.choice(n, size=(n_swaps, 2), replace=False)
        
        # 创建物品副本
        swapped_items = [item.copy() for item in items]
        
        # 执行交换
        for i, j in swap_pairs:
            # 交换权重
            swapped_items[i]['weight'], swapped_items[j]['weight'] = \
                swapped_items[j]['weight'], swapped_items[i]['weight']
            
        return swapped_items
        
    def _local_perturbation(self, items: List[Dict]) -> List[Dict]:
        """局部扰动"""
        perturbed_items = []
        n = len(items)
        perturb_indices = np.random.choice(n, size=n//3, replace=False)
        
        for i in range(n):
            if i in perturb_indices:
                # 对选中的物品进行局部扰动
                perturb_ratio = np.random.uniform(0.9, 1.1)
                perturbed_items.append({
                    'weight': items[i]['weight'] * perturb_ratio,
                    'value': items[i]['value'] * perturb_ratio
                })
            else:
                perturbed_items.append(items[i].copy())
        return perturbed_items
