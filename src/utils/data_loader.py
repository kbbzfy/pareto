import os
import json
import random
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Tuple, Dict, List, Any
from .data_augmentation import DataAugmentor
from src.utils.problem_factory import ProblemFactory

class OptimizationDataset(Dataset):
    def __init__(self, data_dir: str, problem_type: str, problem_abstraction, config: Dict):
        """
        初始化数据集
        
        Args:
            data_dir: 数据目录路径
            problem_type: 问题类型 ('tsp' 或 'knapsack')
            problem_abstraction: 问题抽象化实例
            config: 配置字典
        """
        self.data_dir = data_dir
        self.problem_type = problem_type
        self.problem_abstraction = problem_abstraction
        self.data_files = self._load_data_files()
        self.max_items = self._get_max_items()
        self.config = config
        
        # 添加数据增强器
        self.augmentor = None  # 暂时禁用数据增强
        
    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.data_files)
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict]:
        """获取数据集中的一个样本"""
        try:
            with open(self.data_files[idx], 'r') as f:
                data = json.load(f)
                
            # 转换问题数据为模型可用的格式
            features, problem_info = self.problem_abstraction.transform_problem(data)
            
            # 确保特征维度一致
            if self.problem_type == 'knapsack':
                if features.size(1) == 4:  # 如果是4维特征(价值,重量,体积,成本)
                    # 添加类别信息作为第5维
                    category_info = torch.zeros(features.size(0), 1)
                    if 'categories' in data:
                        for i, cat in enumerate(data['categories']):
                            category_info[i] = float(cat)
                    features = torch.cat([features, category_info], dim=1)
                else:
                    raise ValueError(f"Knapsack features should have 4 dimensions before adding category, got {features.size(1)}")
                    
            # 添加问题类型信息
            if 'type' not in problem_info:
                problem_info['type'] = self.problem_type
                
            return features, problem_info
            
        except Exception as e:
            print(f"数据加载出错 (file: {self.data_files[idx]}): {e}")
            # 创建一个安全的默认特征张量
            dummy_features = torch.zeros(1, 5)  # 使用5维特征(4个属性 + 1个类别)
            dummy_info = {'type': self.problem_type, 'error': str(e)}
            return dummy_features, dummy_info
        
    def _load_data_files(self) -> List[str]:
        """加载数据文件列表"""
        try:
            if not os.path.exists(self.data_dir):
                print(f"警告：目录不存在: {self.data_dir}")
                return []
                
            files = [f for f in os.listdir(self.data_dir) 
                    if f.endswith('.json')]
            
            # 如果没有文件，尝试生成数据
            if len(files) == 0:
                print(f"警告：目录 {self.data_dir} 中没有找到数据文件")
                return []
                
            print(f"找到 {len(files)} 个数据文件在 {self.data_dir}")
            return [os.path.join(self.data_dir, f) for f in files]
        except Exception as e:
            print(f"加载数据文件时出错: {e}")
            return []
        
    def _get_max_items(self) -> int:
        """获取数据集中最大的物品数量"""
        max_items = 0
        for file in self.data_files:
            with open(file, 'r') as f:
                data = json.load(f)
                if self.problem_type == 'knapsack':
                    max_items = max(max_items, len(data['items']))
                else:  # tsp
                    max_items = max(max_items, len(data['cities']))
        return max_items

def create_data_loaders(config: Dict[str, Any], problem_abstraction) -> Tuple[DataLoader, DataLoader]:
    """
    创建训练和验证数据加载器
    
    Args:
        config: 配置字典
        problem_abstraction: 问题抽象化实例
        
    Returns:
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
    """
    # 创建训练集
    train_dataset = OptimizationDataset(
        data_dir=config['data']['train_dir'],
        problem_type=config['problem']['type'],
        problem_abstraction=problem_abstraction,
        config=config
    )
    
    # 创建验证集
    val_dataset = OptimizationDataset(
        data_dir=config['data']['val_dir'],
        problem_type=config['problem']['type'],
        problem_abstraction=problem_abstraction,
        config=config
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['model']['batch_size'],
        shuffle=True,
        num_workers=config['data'].get('num_workers', 0)
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['model']['batch_size'],
        shuffle=False,
        num_workers=config['data'].get('num_workers', 0)
    )
    
    return train_loader, val_loader 

def custom_collate_fn(batch):
    """自定义的数据批处理函数
    
    Args:
        batch: 一批数据样本
        
    Returns:
        features: 批量特征数据
        problem_infos: 批量问题信息
    """
    # 提取特征和问题信息
    features = [item[0] for item in batch]
    problem_infos = [item[1] for item in batch]
    
    # 找到这批数据中最大的物品数量
    max_items = max(feat.size(0) for feat in features)
    
    # 填充所有样本到相同长度
    padded_features = []
    for feat in features:
        n_items = feat.size(0)
        if n_items < max_items:
            # 创建填充张量
            padding = torch.zeros(max_items - n_items, feat.size(1))
            # 连接原始特征和填充
            padded_feat = torch.cat([feat, padding], dim=0)
            padded_features.append(padded_feat)
        else:
            padded_features.append(feat)
    
    # 将列表转换为张量
    padded_features = torch.stack(padded_features)
    
    return padded_features, problem_infos 

class DataLoader:
    def __init__(self, problem_type, data_dir='data'):
        self.problem_type = problem_type
        self.data_dir = os.path.join(data_dir, problem_type)
        self.problem = ProblemFactory.create_problem(problem_type)
        
        # 确保数据目录存在
        os.makedirs(os.path.join(self.data_dir, 'train'), exist_ok=True)
        os.makedirs(os.path.join(self.data_dir, 'val'), exist_ok=True)
        
    def load_data(self, split='train'):
        """加载数据"""
        data_path = os.path.join(self.data_dir, split)
        instances = []
        
        if not os.path.exists(data_path):
            return []
            
        for file in os.listdir(data_path):
            if self.problem_type == 'knapsack' and file.endswith('.json'):
                with open(os.path.join(data_path, file), 'r') as f:
                    data = json.load(f)
                    items = np.array(data['items'])
                    capacity = data['capacity']
                    instances.append((items, capacity))
            elif file.endswith('.npy'):
                data = np.load(os.path.join(data_path, file), allow_pickle=True)
                instances.append(data)
                
        return instances
        
    def save_data(self, instances, split='train'):
        """保存数据"""
        data_path = os.path.join(self.data_dir, split)
        os.makedirs(data_path, exist_ok=True)
        
        for i, instance in enumerate(instances):
            if self.problem_type == 'knapsack':
                # 适应新的数据格式
                file_path = os.path.join(data_path, f'instance_{i}.json')
                data = {
                    'items': instance['items'].tolist(),
                    'capacity': float(instance['capacity']),
                    'greedy_solution': instance['greedy_solution'].tolist(),
                    'metrics': {
                        'total_value': float(instance['metrics']['total_value']),
                        'total_weight': float(instance['metrics']['total_weight']),
                        'utilization': float(instance['metrics']['utilization'])
                    }
                }
                with open(file_path, 'w') as f:
                    json.dump(data, f)
            else:
                file_path = os.path.join(data_path, f'instance_{i}.npy')
                np.save(file_path, instance)
