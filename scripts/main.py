import os
import sys

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import argparse
import logging
import yaml
from datetime import datetime
import torch
import random
import json
from typing import Dict, List
import numpy as np
from src.utils.problem_factory import problem_factory
from src.environments.knapsack_env import KnapsackEnv
from scripts.train import RLTrainer
from src.utils.visualization import ProblemVisualizer

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.fast_solver import FastSolver
from src.utils.data_loader import create_data_loaders
from src.utils.problem_abstraction import ProblemAbstraction
from src.utils.visualization import ProblemVisualizer
from scripts.train import Trainer, RLTrainer
from scripts.generate_data import generate_dataset
from src.environments.knapsack_env import KnapsackEnv
from src.environments.tsp_env import TSPEnv
from src.utils.problem_factory import ProblemFactory

def setup_logging(log_dir: str) -> None:
    """设置日志"""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'run_{timestamp}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def ensure_checkpoint_dirs():
    """确保checkpoint目录存在"""
    os.makedirs(os.path.join('checkpoints', 'knapsack'), exist_ok=True)
    os.makedirs(os.path.join('checkpoints', 'tsp'), exist_ok=True)
    # 不再创建嵌套目录

def get_checkpoint_path(problem_type, checkpoint_name):
    """获取checkpoint完整路径"""
    return os.path.join('checkpoints', problem_type, checkpoint_name)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, required=True,
                      choices=['generate', 'train', 'test'],
                      help='运行模式：generate-生成数据，train-训练模型，test-测试模型')
    parser.add_argument('--problem', type=str, required=True,
                      choices=['knapsack', 'tsp'],
                      help='问题类型：knapsack-背包问题，tsp-旅行商问题')
    parser.add_argument('--rl', action='store_true',
                      help='是否使用强化学习方法')
    parser.add_argument('--visualize', action='store_true',
                      help='是否可视化结果')
    parser.add_argument('--checkpoint', type=str,
                      help='模型检查点路径（仅在test模式下使用）')
    parser.add_argument('--n-objectives', type=int, default=4,
                      help='目标函数的数量（默认为3）')
    parser.add_argument('--objective-types', type=str, default=None,
                      help='目标函数类型，格式为逗号分隔的1/-1列表，1表示最大化，-1表示最小化（例如：1,-1,-1）')
    parser.add_argument('--objective-names', type=str, default=None,
                      help='目标函数名称，格式为逗号分隔的名称列表（例如：价值,重量,体积）')
    parser.add_argument('--config-file', type=str, default=None,
                      help='配置文件路径')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s'
    )
    
    # 根据问题类型选择配置文件路径
    if args.config_file is None:
        if args.problem == 'knapsack':
            args.config_file = 'configs/knapsack_config.yaml'
        elif args.problem == 'tsp':
            args.config_file = 'configs/tsp_config.yaml'
        else:
            raise ValueError(f"不支持的问题类型: {args.problem}")
    
    # 初始化配置
    if args.config_file and os.path.exists(args.config_file):
        problem_factory.initialize(config_path=args.config_file)
    else:
        print(f"警告：找不到配置文件 {args.config_file}，尝试使用默认配置")
    
    # 加载配置
    with open(args.config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 更新问题类型
    config['problem']['type'] = args.problem
    
    # 更新多目标设置
    if 'model' not in config:
        config['model'] = {}
    
    # 设置目标数量
    config['model']['n_objectives'] = args.n_objectives
    
    # 处理命令行参数中的目标名称和类型
    if args.objective_names:
        objective_names = args.objective_names.split(',')
        config['problem']['objective_names'] = objective_names
        
        # 如果启用了英文标签，尝试将中文目标名称转换为英文
        if config.get('visualization', {}).get('use_english_labels', False):
            # 使用已导入的ProblemVisualizer
            visualizer = ProblemVisualizer(use_english_labels=True)
            config['problem']['objective_names'] = visualizer._translate_objective_names(objective_names)
            logging.info(f"转换后的目标名称: {config['problem']['objective_names']}")
    
    if args.objective_types:
        # 添加引号保护，确保正确解析
        objective_types_str = args.objective_types.replace(' ', '')
        if objective_types_str.startswith('"') and objective_types_str.endswith('"'):
            objective_types_str = objective_types_str[1:-1]
        elif objective_types_str.startswith("'") and objective_types_str.endswith("'"):
            objective_types_str = objective_types_str[1:-1]
            
        try:
            objective_types = [int(t) for t in objective_types_str.split(',')]
            config['problem']['objective_types'] = objective_types
        except ValueError as e:
            logging.error(f"解析目标类型时出错: {e}")
            logging.error(f"原始输入: {args.objective_types}")
            logging.error(f"处理后输入: {objective_types_str}")
            config['problem']['objective_types'] = [-1] * args.n_objectives  # 默认全部为最小化
            logging.info(f"使用默认目标类型: {config['problem']['objective_types']}")
    
    # 确保数据目录存在
    for split in ['train', 'val', 'test']:
        data_dir = os.path.join('data', args.problem, split)
        os.makedirs(data_dir, exist_ok=True)
        if not os.path.exists(data_dir) or len(os.listdir(data_dir)) == 0:
            print(f"正在生成{split}数据集...")
            generate_dataset(args.problem, config, split=split)
    
    if args.mode == 'generate':
        generate_dataset(args.problem, config)
        if args.visualize:
            try:
                # 创建可视化器
                use_english_labels = config.get('visualization', {}).get('use_english_labels', False)
                visualizer = ProblemVisualizer(use_english_labels=use_english_labels)
                logging.info(f"使用英文标签: {use_english_labels}")
                visualizer.visualize_dataset_samples(f'data/{args.problem}/train', args.problem)
            except Exception as e:
                logging.error(f"可视化数据时出错: {e}")
    
    # 如果是非RL模式，确保数据集存在
    if not args.rl and args.mode in ['train', 'test']:
        data_dir = os.path.join('data', args.problem, 'train')
        if not os.path.exists(data_dir) or len(os.listdir(data_dir)) == 0:
            print("数据集不存在或为空，正在生成...")
            generate_dataset(args.problem, config)
    
    if args.mode in ['train', 'test']:
        if args.rl:
            # 创建环境和训练器
            if args.problem == 'knapsack':
                # 确保配置中有正确的问题参数
                problem_config = {
                    'problem': {
                        'type': 'knapsack',
                        'num_categories': config['problem'].get('num_categories', 5),
                        'items_per_category': config['problem'].get('items_per_category', 4),
                        'num_objectives': args.n_objectives,  # 使用命令行参数指定的目标数量
                        'objective_names': config['problem'].get('objective_names', [])  # 使用problem中的目标名称
                    }
                }
                
                # 添加目标类型，如果存在
                if 'objective_types' in config['problem']:
                    problem_config['problem']['objective_types'] = config['problem']['objective_types']
                
                # 使用更新后的配置创建问题实例
                problem = problem_factory.create_problem(args.problem, problem_config)
                env = KnapsackEnv(problem)
                
                # 打印调试信息
                logging.info(f"问题类型: {args.problem}")
                logging.info(f"目标数量: {args.n_objectives}")
                logging.info(f"目标名称: {config['problem'].get('objective_names', [])}")
                
                # 确保环境的目标数量与配置一致
                if hasattr(env, 'num_objectives') and env.num_objectives != args.n_objectives:
                    logging.warning(f"环境目标数量({env.num_objectives})与命令行参数({args.n_objectives})不匹配，以命令行参数为准")
                    env.num_objectives = args.n_objectives
                
                # 确保环境的目标名称与配置一致
                if hasattr(env, 'objective_names') and len(config['problem'].get('objective_names', [])) > 0:
                    logging.info(f"更新环境目标名称: {config['problem']['objective_names']}")
                    env.objective_names = config['problem']['objective_names']
            elif config['problem']['type'] == 'tsp':
                # 为TSP问题准备专门的配置
                problem_config = {
                    'problem': {
                        'type': 'tsp',
                        'size': config['problem'].get('size', 20),
                        'num_objectives': config['problem'].get('n_objectives', 4),
                        'objective_names': config['problem'].get('objective_names', 
                            ["总距离", "时间窗口", "平衡性", "成本"]),
                        'objective_types': config['problem'].get('objective_types', 
                            [-1] * config['problem'].get('n_objectives', 4))
                    },
                    'model': {
                        'objective_names': config.get('model', {}).get('objective_names', []),
                        'objective_types': config.get('model', {}).get('objective_types', 
                            [-1] * config['problem'].get('n_objectives', 4))
                    }
                }
                
                # 记录问题配置
                logging.info(f"创建问题实例，配置:")
                logging.info(f"  - 问题类型: TSP问题")
                logging.info(f"  - 城市数量: {problem_config['problem']['size']}")
                logging.info(f"  - 目标数量: {problem_config['problem']['num_objectives']}")
                logging.info(f"  - 目标名称: {problem_config['problem']['objective_names']}")
                logging.info(f"  - 目标类型: {problem_config['problem']['objective_types']}")
                
                # 使用专门配置创建问题实例
                problem = problem_factory.create_problem('tsp', problem_config)
                
                # 使用问题实例创建环境
                env = TSPEnv(problem)
                logging.info("成功创建TSP问题环境")
            else:
                # 为其他问题类型创建配置和问题实例
                problem_config = {
                    'problem': {
                        'type': args.problem,
                        'size': config['problem'].get('size', 20),
                        'num_objectives': args.n_objectives
                    }
                }
                problem = problem_factory.create_problem(args.problem, problem_config)
                env = TSPEnv(problem)
                
            trainer = RLTrainer(env, config)
        else:
            config['data']['train_dir'] = os.path.join('data', args.problem, 'train')
            config['data']['val_dir'] = os.path.join('data', args.problem, 'val')
            config['data']['test_dir'] = os.path.join('data', args.problem, 'test')
            trainer = Trainer(config)
    
    if args.mode == 'train':
        ensure_checkpoint_dirs()
        trainer.train()

    elif args.mode == 'test':
        if not args.checkpoint:
            # 只在正确的目录中查找模型
            checkpoint_dir = os.path.join('checkpoints', args.problem)
            
            if os.path.exists(checkpoint_dir):
                models = [os.path.join(checkpoint_dir, f) for f in os.listdir(checkpoint_dir) 
                         if f.endswith('.pt')]
                
                if models:
                    # 按修改时间排序，选择最新的模型
                    latest_model = max(models, key=os.path.getmtime)
                    logging.info(f"自动选择最新模型: {latest_model}")
                    args.checkpoint = latest_model
                else:
                    raise ValueError("未找到可用模型，请先运行训练或手动指定模型路径")
            else:
                raise ValueError(f"模型目录 {checkpoint_dir} 不存在，请先运行训练")
        
        checkpoint_name = os.path.basename(args.checkpoint)
        
        # 只检查两个合理的路径
        possible_paths = [
            args.checkpoint,  # 用户直接指定的路径
            os.path.join('checkpoints', args.problem, checkpoint_name)  # 标准路径
        ]
        
        checkpoint_path = None
        for path in possible_paths:
            if os.path.exists(path):
                checkpoint_path = path
                logging.info(f"找到模型文件: {path}")
                break
        
        if not checkpoint_path:
            raise ValueError(f"无法找到模型文件: {args.checkpoint}")
        
        try:
            trainer.load_model(checkpoint_path)
            
            # 如果是RL模式，使用环境进行测试
            if args.rl:
                results, solution, objectives = trainer.test(visualize=args.visualize)
            else:
                # 使用数据集进行测试
                results, solution, objectives = trainer.test_on_dataset(visualize=args.visualize)
            
            # 输出主要指标
            logging.info("\n测试结果:")
            if isinstance(results, dict):
                if 'error' in results:
                    logging.error(f"测试过程出错: {results['error']}")
                else:
                    if 'total_value' in results:
                        logging.info(f"总价值: {results['total_value']:.2f}")
                    if 'total_weight' in results:
                        logging.info(f"总重量: {results['total_weight']:.2f}")
                    if 'selected_items' in results:
                        logging.info(f"选取数量: {results['selected_items']}")
                    if 'mean_reward' in results:
                        logging.info(f"平均奖励: {results['mean_reward']:.2f}")
            
        except Exception as e:
            logging.error(f"测试过程中出错: {str(e)}")
            raise e

if __name__ == '__main__':
    main()
