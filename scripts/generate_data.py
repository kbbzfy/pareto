import os
import numpy as np
import yaml
from src.utils.problem_factory import problem_factory
from src.utils.data_loader import DataLoader
import argparse
from src.utils.visualization import ProblemVisualizer
import json
from src.utils.data_augmentation import DataAugmentation
import torch
from datetime import datetime

def generate_knapsack_instance(config):
    """生成背包问题实例"""
    # 从配置中获取基本参数，设置默认值
    try:
        num_categories = config['problem'].get('num_categories', 5)
        items_per_category = config['problem'].get('items_per_category', 4)
        
        # 支持两种可能的目标数量键名
        num_objectives = config['problem'].get('num_objectives', 
                                             config['problem'].get('n_objectives', 3))
        
        # 打印调试信息
        print(f"生成背包问题实例：")
        print(f"  类别数量: {num_categories}")
        print(f"  每类物品数: {items_per_category}")
        print(f"  目标数量: {num_objectives}")
        
        # 获取目标类型和名称
        objective_names = config.get('model', {}).get('objective_names', None)
        if not objective_names:
            objective_names = config['problem'].get('objective_names', 
                                                  ['价值', '重量', '体积', '成本', '时间'][:num_objectives])
        
        objective_types = config.get('model', {}).get('objective_types', None)
        if not objective_types:
            objective_types = config['problem'].get('objective_types', 
                                                  [1] + [-1] * (num_objectives - 1))
        
        print(f"  目标名称: {objective_names[:num_objectives]}")
        print(f"  目标类型: {objective_types[:num_objectives]}")
        
        # 获取目标范围，如果没有则使用默认值
        if 'objective_ranges' in config['problem']:
            objective_ranges = config['problem']['objective_ranges']
        else:
            # 创建默认范围：价值(50-200)，其他约束(10-50)
            objective_ranges = [(50, 200)] + [(10, 50)] * (num_objectives - 1)
            print(f"  使用默认目标范围: {objective_ranges}")
        
    except KeyError as e:
        print(f"配置缺少必要的键: {e}")
        # 使用默认值
        num_categories = 5
        items_per_category = 4
        num_objectives = 3
        objective_names = ['价值', '重量', '体积']
        objective_types = [1, -1, -1]
        objective_ranges = [(50, 200), (10, 50), (10, 50)]
        print(f"使用默认配置生成实例")
    
    # 创建num_objectives维的物品数组
    objective_dimensions = config['problem']['n_objectives']
    items = np.zeros((num_categories, items_per_category, objective_dimensions))
    
    # 为每个目标生成随机值
    for i, obj_name in enumerate(config['problem']['objective_names']):
        if i < objective_dimensions:
            # 确保有足够的范围
            if i < len(objective_ranges):
                min_val, max_val = objective_ranges[i]
            else:
                # 使用默认范围
                min_val, max_val = (10, 50)
            
            items[:, :, i] = np.random.uniform(
                min_val, max_val, 
                size=(num_categories, items_per_category)
            )
    
    # 生成初始解（可选）
    initial_solution = np.random.randint(0, items_per_category, size=num_categories)
    
    # 创建数据字典
    data = {
        'items': items.tolist(),
        'initial_solution': initial_solution.tolist(),
        'objective_names': objective_names[:objective_dimensions],
        'objective_types': objective_types[:objective_dimensions],
        'num_objectives': objective_dimensions
    }
    
    # 添加约束信息
    if 'objective_constraints' in config['problem']:
        for name, value in config['problem']['objective_constraints'].items():
            data[f'{name}_capacity'] = value
    else:
        # 计算默认约束：总物品属性值的一半
        for i in range(1, objective_dimensions):  # 跳过第一个目标（价值）
            obj_name = objective_names[i] if i < len(objective_names) else f'objective_{i}'
            total_value = np.sum(items[:, :, i])
            capacity = total_value * 0.5  # 设置为总和的一半
            data[f'{obj_name}_capacity'] = float(capacity)
    
    return data

def generate_tsp_instance(config):
    """生成TSP问题实例"""
    try:
        # 获取城市数量
        size = config['problem'].get('size', 20)
        
        # 获取目标数量
        num_objectives = config['problem'].get('num_objectives', 
                                             config['problem'].get('n_objectives', 3))
        
        # 获取目标名称
        default_names = ["总距离", "时间窗口", "平衡性", "最大段长", "其他目标"]
        objective_names = config['problem'].get('objective_names', 
                                               default_names[:num_objectives])
        
        # 确保目标名称数量匹配
        while len(objective_names) < num_objectives:
            objective_names.append(f"目标{len(objective_names)+1}")
        
        # 获取目标类型
        default_types = [-1] * num_objectives  # 默认全部最小化
        objective_types = config['problem'].get('objective_types', default_types)
        
        # 确保目标类型数量匹配
        while len(objective_types) < num_objectives:
            objective_types.append(-1)  # 默认添加最小化目标
        
        print(f"生成TSP问题实例：")
        print(f"  城市数量: {size}")
        print(f"  目标数量: {num_objectives}")
        print(f"  目标名称: {objective_names}")
        print(f"  目标类型: {objective_types}")
        
    except KeyError as e:
        print(f"配置缺少必要的键: {e}")
        # 使用默认值
        size = 20
        num_objectives = 3
        objective_names = ["总距离", "时间窗口", "平衡性"]
        objective_types = [-1, -1, -1]
        print(f"使用默认配置生成实例")
    
    # 创建TSP问题实例
    tsp_problem = problem_factory.create_problem('tsp', config)
    
    # 生成问题实例
    instance = tsp_problem.generate_instance(size)
    
    # 创建数据字典
    data = {
        'coordinates': instance['coordinates'].tolist(),
        'distance_matrix': instance['distance_matrix'].tolist(),
        'nearest_neighbor_tour': instance['nearest_neighbor_tour'].tolist(),
        'objective_names': objective_names,
        'objective_types': objective_types,
        'num_objectives': num_objectives,
        'size': size
    }
    
    return data

def generate_dataset(problem_type, config, split='train'):
    """生成数据集"""
    # 确保输出目录存在
    output_dir = os.path.join('data', problem_type, split)
    os.makedirs(output_dir, exist_ok=True)
    
    # 根据问题类型选择相应的生成函数
    if problem_type == 'knapsack':
        data = generate_knapsack_instance(config)
    elif problem_type == 'tsp':
        data = generate_tsp_instance(config)
    else:
        raise ValueError(f"不支持的问题类型: {problem_type}")
    
    # 保存到文件
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = os.path.join(output_dir, f'instance_{timestamp}.json')
    
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)

def main(args):
    """生成数据集的主函数"""
    # 尝试使用命令行参数中的配置文件路径，如果没有则使用默认值
    config_file = args.config_file if hasattr(args, 'config_file') else 'configs/knapsack_config.yaml'
    
    if args.problem == 'tsp':
        config_file = 'configs/tsp_config.yaml'
    
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"配置文件 {config_file} 不存在，使用默认配置")
        config = {
            'problem': {
                'type': args.problem,
                'size': 20,
                'n_objectives': 3
            },
            'data': {
                'train_size': 1000,
                'val_size': 200,
                'test_size': 200
            }
        }
    
    # 生成数据集
    generate_dataset(args.problem, config)
    
    if args.visualize:
        print(f"Visualizing {args.problem} instances...")
        visualizer = ProblemVisualizer()
        visualizer.visualize_dataset_samples(f'data/{args.problem}/train', args.problem)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--problem', type=str, required=True, help='Problem type (knapsack/tsp)')
    parser.add_argument('--visualize', action='store_true', help='Visualize some examples')
    parser.add_argument('--config-file', type=str, help='配置文件路径')
    args = parser.parse_args()
    main(args)
