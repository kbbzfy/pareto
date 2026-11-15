import torch
import yaml
import os
import logging
from torch.utils.data import DataLoader
from src.models.fast_solver import FastSolver
from src.utils.data_augmentation import DataAugmentation
from src.utils.problem_abstraction import ProblemAbstraction
from datetime import datetime
from src.utils.data_loader import OptimizationDataset, custom_collate_fn
from typing import Dict, List, Tuple
import torch.nn as nn
import numpy as np
import random
from src.environments.knapsack_env import KnapsackEnv
from src.environments.tsp_env import TSPEnv
from src.utils.replay_buffer import ReplayBuffer, Experience
from src.utils.visualization import ProblemVisualizer
from collections import namedtuple
import time
import torch.nn.functional as F
import gc
import traceback
import argparse
import platform
import psutil
import sys
from src.utils.problem_factory import problem_factory
import json
import matplotlib.pyplot as plt

# 确保必要目录存在
os.makedirs('logs', exist_ok=True)
os.makedirs('solutions', exist_ok=True)
os.makedirs('checkpoints', exist_ok=True)

# 设置日志格式
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(f'logs/run_{time.strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)

def custom_collate_fn(batch: List[Tuple[torch.Tensor, Dict]]) -> Tuple[torch.Tensor, List[Dict]]:
    """自定义的数据批处理函数"""
    # 提取特征和问题信息
    features = [item[0] for item in batch]
    problem_infos = [item[1] for item in batch]
    
    # 找到这批数据中最大的物品数量和特征维度
    max_items = max(feat.size(0) for feat in features)
    feature_dim = features[0].size(1)  # 使用第一个样本的特征维度
    
    # 创建填充张量
    padded_features = torch.zeros(len(features), max_items, feature_dim)
    
    # 填充所有样本到相同长度
    for i, feat in enumerate(features):
        n_items = feat.size(0)
        padded_features[i, :n_items] = feat
    
    # 确保problem_info中的所有张量数据具有一致的长度
    for i, info in enumerate(problem_infos):
        # 记录原始物品数量
        original_num_items = None
        for key, value in info.items():
            if isinstance(value, torch.Tensor) and value.dim() == 1:
                if original_num_items is None:
                    original_num_items = value.size(0)
                
                # 确保所有张量属性具有相同的大小
                if value.size(0) != max_items:
                    # 创建一个新的填充张量
                    padded_value = torch.zeros(max_items).to(value.device)
                    # 复制原始数据
                    padded_value[:value.size(0)] = value
                    # 用最后一个值填充剩余部分
                    if value.size(0) > 0:
                        padded_value[value.size(0):] = value[-1]
                    # 更新属性
                    info[key] = padded_value
        
        # 设置有效物品数量
        if original_num_items is not None:
            info['valid_items'] = original_num_items
        else:
            # 如果没有找到任何属性，使用最大填充长度
            info['valid_items'] = max_items
    
    return padded_features, problem_infos

class Trainer:
    def __init__(self, config: Dict):
        """初始化训练器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.current_epoch = 0  # 添加当前epoch计数器
        
        # 创建模型
        self.model = FastSolver(
            input_dim=config['model']['input_dim'],
            hidden_dim=config['model']['hidden_dim'],
            problem_type=config['problem']['type']
        ).to(self.device)
        
        # 创建优化器
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config['training']['learning_rate']
        )
        
        # 创建数据集
        problem_abstraction = ProblemAbstraction(max_nodes=config['data']['max_size'])
        train_dataset = OptimizationDataset(
            config['data']['train_dir'],
            config['problem']['type'],
            problem_abstraction,
            config
        )
        
        val_dataset = OptimizationDataset(
            config['data']['val_dir'],
            config['problem']['type'],
            problem_abstraction,
            config
        )
        
        # 创建数据加载器
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=True,
            num_workers=config['data']['num_workers'],
            collate_fn=custom_collate_fn
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=False,
            num_workers=config['data']['num_workers'],
            collate_fn=custom_collate_fn
        )
        
        # 初始化最佳验证损失
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        # 添加训练损失记录
        self.train_losses = []
        self.val_losses = []
        
        # 在Trainer类的__init__方法或代码开始处添加
        os.makedirs('solutions', exist_ok=True)
        
    def train(self):
        """训练模型"""
        logging.info("Starting training...")
        
        try:
            for epoch in range(self.config['training']['epochs']):
                self.current_epoch = epoch
                
                # 训练一个epoch
                train_loss = self._train_epoch()
                self.train_losses.append(train_loss)  # 记录训练损失
                
                # 验证
                val_loss = self._validate()
                if isinstance(val_loss, dict):
                    self.val_losses.append(val_loss.get('val_loss', 0))  # 记录验证损失
                else:
                    self.val_losses.append(val_loss)  # 记录验证损失
                
                # 早停检查
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.patience_counter = 0
                    self._save_checkpoint()
                else:
                    self.patience_counter += 1
                    
                if self.patience_counter >= self.config['training']['early_stopping_patience']:
                    logging.info("Early stopping triggered")
                    break
                    
                logging.info(f'Epoch {epoch+1}/{self.config["training"]["epochs"]}, '
                            f'Train Loss: {train_loss:.4f}, '
                            f'Val Loss: {val_loss:.4f}')
                
            # 训练完成后，保存训练损失
            self._save_training_losses()
            
        except KeyboardInterrupt:
            print("\n训练被用户中断")
            # 保存当前模型和训练损失
            self._save_checkpoint(filename="interrupted_checkpoint.pt")
            self._save_training_losses()
        
        except Exception as e:
            print(f"训练过程中发生严重错误: {str(e)}")
            traceback.print_exc()
            # 尝试保存模型和训练损失
            try:
                self._save_checkpoint(filename="error_checkpoint.pt")
                self._save_training_losses()
            except:
                print("无法保存错误时的模型状态")

    def _save_training_losses(self):
        """保存训练损失图表"""
        # 确保训练有损失记录
        if not self.train_losses:
            return
        
        try:
            # 确保目录存在
            os.makedirs('solutions', exist_ok=True)
            
            # 设置图表样式
            plt.figure(figsize=(10, 6))
            
            # 使用英文标题和标签，避免中文乱码
            plt.title('Training Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            
            # 绘制损失曲线
            plt.plot(self.train_losses, label='Train Loss')
            if self.val_losses:
                plt.plot(self.val_losses, label='Validation Loss')
            plt.legend()
            
            # 只保存图表到solutions目录
            plt.savefig('solutions/latest_training_loss.png')
            plt.close()
            
        except Exception as e:
            logging.warning(f"保存训练损失图表时出错: {str(e)}")
    
    def _train_epoch(self):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        
        for batch_idx, (data, problem_info) in enumerate(self.train_loader):
            # 移动数据到设备
            data = data.to(self.device)
            
            self.optimizer.zero_grad()
            
            # 前向传播
            action_probs, state_value, objectives = self.model(data)
            solution = action_probs  # 使用动作概率作为解决方案
            
            # 计算损失
            loss = self._calculate_loss(solution, objectives, problem_info)
            
            # 反向传播
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                logging.info(f'Train Epoch: {self.current_epoch} '
                           f'[{batch_idx}/{len(self.train_loader)} '
                           f'({100. * batch_idx / len(self.train_loader):.0f}%)]\t'
                           f'Loss: {loss.item():.6f}')
        
        return total_loss / len(self.train_loader)
    
    def _validate(self):
        """验证模型"""
        self.model.eval()
        total_loss = 0
        total_value = 0
        total_feasible = 0
        n_samples = 0
        
        with torch.no_grad():
            for data, problem_info in self.val_loader:
                try:
                    data = data.to(self.device)
                    action_probs, state_value, objectives = self.model(data)
                    solution = action_probs  # 使用动作概率作为解决方案
                    loss = self._calculate_loss(solution, objectives, problem_info)
                    
                    # 计算额外的指标
                    batch_value, batch_feasible = self._calculate_metrics(solution, problem_info)
                    
                    # 累加统计
                    total_loss += loss.item()
                    total_value += batch_value
                    total_feasible += batch_feasible
                    n_samples += data.size(0)
                except Exception as e:
                    logging.warning(f"验证过程中出错: {str(e)}")
                    # 验证错误不应该中断整个过程
                    continue
        
        # 确保有有效样本
        if n_samples == 0:
            n_samples = 1  # 避免除零错误
            
        metrics = {
            'val_loss': total_loss / len(self.val_loader),
            'avg_value': total_value / n_samples,
            'feasibility_rate': total_feasible / n_samples
        }
        
        # 限制异常值
        if metrics['val_loss'] < -1000:
            metrics['val_loss'] = -1000
        elif metrics['val_loss'] > 1000:
            metrics['val_loss'] = 1000
        
        logging.info(f"Validation metrics: {metrics}")
        return metrics['val_loss']
    
    def _calculate_loss(self, solution: torch.Tensor, objectives: torch.Tensor, 
                       problem_info: List[Dict]) -> torch.Tensor:
        """计算损失函数
        
        Args:
            solution: 预测的解决方案 [batch_size, n_items]
            objectives: 预测的目标函数值 [batch_size, n_objectives]
            problem_info: 问题信息列表
            
        Returns:
            loss: 损失值
        """
        batch_size = solution.size(0)
        device = solution.device
        
        # 计算可行性损失
        feasibility_loss = self._calculate_feasibility_loss(solution, problem_info)
        
        # 计算目标函数损失
        objective_loss = torch.zeros(1, device=device)
        objective_weights = torch.tensor(self.config['model']['objective_types'], 
                                      device=device, dtype=torch.float32)
        
        # 对每个目标进行归一化和加权
        for i in range(objectives.size(1)):
            objective_values = objectives[:, i]
            weight = objective_weights[i]
            
            # 归一化目标值到[0,1]范围
            if objective_values.abs().max() > 0:
                normalized_obj = objective_values / (objective_values.abs().max() + 1e-8)
            else:
                normalized_obj = objective_values
            
            # 根据目标类型（最大化/最小化）计算损失
            objective_loss -= weight * normalized_obj.mean()
        
        # 添加零解惩罚项
        zero_solution_penalty = torch.exp(-solution.sum(dim=1).mean())
        
        # 总损失
        total_loss = (objective_loss + 
                     self.config['training'].get('feasibility_weight', 1.0) * feasibility_loss +
                     self.config['training'].get('zero_penalty_weight', 0.5) * zero_solution_penalty)
        
        # 限制损失范围
        total_loss = torch.clamp(total_loss, min=-100.0, max=100.0)
        
        return total_loss
    
    def _calculate_feasibility_loss(self, solution: torch.Tensor, 
                                  problem_info: List[Dict]) -> torch.Tensor:
        """计算可行性损失"""
        batch_size = solution.size(0)
        total_loss = torch.zeros(1).to(self.device)
        
        for i in range(batch_size):
            try:
                # 获取有效物品数量
                valid_items = problem_info[i]['valid_items']
                
                # 获取解决方案并确保维度正确
                if i < solution.size(0):
                    if valid_items <= solution.size(1):
                        # 只考虑有效的物品
                        valid_solution = solution[i, :valid_items]
                    else:
                        # 如果valid_items超出了解决方案的维度，调整valid_items
                        valid_items = solution.size(1)
                        valid_solution = solution[i, :valid_items]
                        # 更新problem_info中的valid_items
                        problem_info[i]['valid_items'] = valid_items
                else:
                    # 批次索引超出范围，跳过此样本
                    continue
                
                # 计算约束违反程度
                capacity = problem_info[i].get('capacity', 100.0)
                
                # 获取权重，确保维度匹配
                if 'weights' in problem_info[i]:
                    weights = problem_info[i]['weights']
                    if isinstance(weights, torch.Tensor):
                        # 记录权重和解决方案的维度，以便调试
                        weights_size = weights.size(0)
                        solution_size = valid_solution.size(0)
                        
                        # 确保weights长度和valid_solution匹配
                        if weights_size != solution_size:
                            logging.debug(f"调整weights大小从 {weights_size} 到 {solution_size}")
                            # 如果长度不匹配，裁剪或填充
                            if weights_size > solution_size:
                                weights = weights[:solution_size]
                            else:
                                # 如果weights太短，用最后一个值填充
                                if weights_size > 0:
                                    padding = torch.ones(solution_size - weights_size).to(self.device) * weights[-1]
                                    weights = torch.cat([weights, padding])
                                else:
                                    # 如果weights为空，使用全1张量
                                    weights = torch.ones(solution_size).to(self.device)
                    else:
                        # 如果weights不是张量，转换并确保长度正确
                        weights = torch.ones(valid_items).to(self.device) * weights
                else:
                    # 如果没有weights，使用默认值
                    weights = torch.ones(valid_items).to(self.device)
                
                # 确保weights在设备上
                weights = weights.to(self.device)
                
                # 最后检查一次维度是否匹配
                if weights.size(0) != valid_solution.size(0):
                    # 如果仍然不匹配，截断长的那个
                    min_size = min(weights.size(0), valid_solution.size(0))
                    weights = weights[:min_size]
                    valid_solution = valid_solution[:min_size]
                
                # 计算总权重
                total_weight = (valid_solution * weights).sum()
                
                # 如果超过容量，添加惩罚
                if total_weight > capacity:
                    total_loss += (total_weight - capacity) ** 2
                    
            except Exception as e:
                logging.warning(f"计算可行性损失时出错: {str(e)}")
                # 出错时使用默认损失值
                total_loss += torch.tensor(1.0).to(self.device)
        
        return total_loss / max(batch_size, 1)  # 避免除零错误
    
    def _calculate_metrics(self, solution: torch.Tensor, problem_info: List[Dict]) -> Tuple[float, int]:
        """
        计算评估指标
        
        Args:
            solution: 模型生成的解决方案
            problem_info: 问题信息
            
        Returns:
            total_value: 总价值
            n_feasible: 可行解的数量
        """
        batch_size = solution.size(0)
        total_value = 0
        n_feasible = 0
        valid_samples = 0
        
        for i in range(batch_size):
            try:
                # 获取有效物品数量
                valid_items = problem_info[i]['valid_items']
                
                # 获取解决方案并确保维度正确
                if i < solution.size(0):
                    if valid_items <= solution.size(1):
                        # 只考虑有效的物品
                        valid_solution = solution[i, :valid_items]
                    else:
                        # 如果valid_items超出了解决方案的维度，调整valid_items
                        valid_items = solution.size(1)
                        valid_solution = solution[i, :valid_items]
                        # 更新problem_info中的valid_items
                        problem_info[i]['valid_items'] = valid_items
                else:
                    # 批次索引超出范围，跳过此样本
                    continue
                
                # 处理价值属性
                if 'values' in problem_info[i]:
                    values = problem_info[i]['values']
                    if isinstance(values, torch.Tensor):
                        # 确保values长度和valid_solution匹配
                        values_size = values.size(0)
                        solution_size = valid_solution.size(0)
                        
                        if values_size != solution_size:
                            # 如果长度不匹配，裁剪或填充
                            if values_size > solution_size:
                                values = values[:solution_size]
                            else:
                                # 如果values太短，用最后一个值填充
                                if values_size > 0:
                                    padding = torch.ones(solution_size - values_size).to(self.device) * values[-1]
                                    values = torch.cat([values, padding])
                                else:
                                    # 如果values为空，使用全1张量
                                    values = torch.ones(solution_size).to(self.device)
                    else:
                        # 如果values不是张量，转换并确保长度正确
                        values = torch.ones(valid_items).to(self.device) * values
                else:
                    # 如果没有values，使用默认值
                    values = torch.ones(valid_items).to(self.device)
                
                # 确保values在设备上
                values = values.to(self.device)
                
                # 最后检查一次维度是否匹配
                if values.size(0) != valid_solution.size(0):
                    # 如果仍然不匹配，截断长的那个
                    min_size = min(values.size(0), valid_solution.size(0))
                    values = values[:min_size]
                    valid_solution = valid_solution[:min_size]
                
                # 计算总价值
                batch_value = (valid_solution * values).sum().item()
                total_value += batch_value
                
                # 处理权重属性
                if 'weights' in problem_info[i]:
                    weights = problem_info[i]['weights']
                    if isinstance(weights, torch.Tensor):
                        # 确保weights长度和valid_solution匹配
                        weights_size = weights.size(0)
                        solution_size = valid_solution.size(0)
                        
                        if weights_size != solution_size:
                            # 如果长度不匹配，裁剪或填充
                            if weights_size > solution_size:
                                weights = weights[:solution_size]
                            else:
                                # 如果weights太短，用最后一个值填充
                                if weights_size > 0:
                                    padding = torch.ones(solution_size - weights_size).to(self.device) * weights[-1]
                                    weights = torch.cat([weights, padding])
                                else:
                                    # 如果weights为空，使用全1张量
                                    weights = torch.ones(solution_size).to(self.device)
                    else:
                        # 如果weights不是张量，转换并确保长度正确
                        weights = torch.ones(valid_items).to(self.device) * weights
                else:
                    # 如果没有weights，使用默认值
                    weights = torch.ones(valid_items).to(self.device)
                
                # 确保weights在设备上
                weights = weights.to(self.device)
                
                # 最后检查一次维度是否匹配
                if weights.size(0) != valid_solution.size(0):
                    # 如果仍然不匹配，截断长的那个
                    min_size = min(weights.size(0), valid_solution.size(0))
                    weights = weights[:min_size]
                    valid_solution = valid_solution[:min_size]
                
                # 检查可行性
                capacity = problem_info[i].get('capacity', 100.0)
                total_weight = (valid_solution * weights).sum().item()
                
                if total_weight <= capacity:
                    n_feasible += 1
                
                # 计算有效样本数
                valid_samples += 1
                    
            except Exception as e:
                logging.warning(f"计算评估指标时出错: {str(e)}")
                # 出错时不增加计数
                continue
            
        # 防止除零错误
        if valid_samples == 0:
            return 0.0, 0
            
        return total_value, n_feasible
    
    def _save_checkpoint(self, filename="latest_model.pt"):
        """保存模型检查点"""
        # 确保目录存在
        checkpoint_dir = os.path.join('checkpoints', self.config['problem']['type'])
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # 直接在problem目录下保存模型
        checkpoint_path = os.path.join(checkpoint_dir, filename)
        
        # 保存模型状态
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if hasattr(self, 'optimizer') else None,
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict() if hasattr(self, 'policy_optimizer') else None,
            'value_optimizer_state_dict': self.value_optimizer.state_dict() if hasattr(self, 'value_optimizer') else None,
            'config': self.config,
            'episode': self.current_episode if hasattr(self, 'current_episode') else 0
        }
        
        torch.save(checkpoint, checkpoint_path)
        logging.info(f"模型已保存到: {checkpoint_path}")
    
    def load_model(self, checkpoint_path):
        """加载模型检查点
        
        Args:
            checkpoint_path: 检查点文件路径
        """
        try:
            logging.info(f"加载模型检查点: {checkpoint_path}")
            
            # 添加numpy相关类型到安全全局变量
            import torch.serialization
            import numpy
            torch.serialization.add_safe_globals([
                numpy.core.multiarray.scalar,
                numpy.dtype,
                numpy.ndarray,
                numpy._globals._NoValue,
                numpy.ufunc,
                numpy.generic,
                numpy.core._multiarray_umath._get_implementing_args
            ])
            
            # 加载检查点，设置weights_only=False以支持numpy类型
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            
            # 尝试加载模型状态
            try:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            except Exception as e:
                logging.warning(f"使用原始模型结构加载失败，尝试适配新结构: {e}")
                
                # 创建状态字典映射
                old_state_dict = checkpoint['model_state_dict']
                new_state_dict = {}
                
                # 映射特征网络的权重
                for key in old_state_dict:
                    if 'feature_network' in key:
                        new_state_dict[key] = old_state_dict[key]
                    elif 'policy_network' in key:
                        # 将策略网络的权重映射到决策网络
                        new_key = key.replace('policy_network', 'decision_network')
                        new_state_dict[new_key] = old_state_dict[key]
                
                # 加载映射后的状态字典
                self.model.load_state_dict(new_state_dict, strict=False)
                logging.info("成功适配并加载新的模型结构")
            
            # 加载优化器状态（如果存在）
            if 'optimizer_state_dict' in checkpoint and hasattr(self, 'optimizer'):
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # 加载其他信息
            self.epoch = checkpoint.get('epoch', 0)
            
            logging.info("模型加载成功")
            
        except Exception as e:
            logging.error(f"加载模型时出错: {str(e)}")
            raise e

    def test(self, visualize=False):
        """测试模型"""
        self.model.eval()
        test_rewards = []
        solutions = []
        objectives_list = []
        
        for episode in range(10):  # 测试10个episode
            state = self.env.reset()
            done = False
            episode_reward = 0
            current_solution = []
            
            while not done:
                # 处理状态
                state_tensor = self._process_state(state)
                
                # 获取动作概率
                with torch.no_grad():
                    action_probs, _, _ = self.model(state_tensor)
                
                # 选择动作
                action = torch.argmax(action_probs).item()
                
                # 执行动作
                next_state, reward, done, _ = self.env.step(action)
                episode_reward += reward
                state = next_state
                current_solution.append(action)
            
            test_rewards.append(episode_reward)
            solutions.append(current_solution)
            
            # 计算目标值
            if isinstance(self.env, KnapsackEnv):
                # 使用current_objectives替代单独的current_value等属性
                objectives = {}
                
                # 确保目标与名称一致
                for i, name in enumerate(self.env.objective_names):
                    if i < len(self.env.current_objectives):
                        objectives[name.lower()] = self.env.current_objectives[i]
                    else:
                        objectives[name.lower()] = 0
                
                # 为兼容性保留原始键名
                if 'value' not in objectives and len(self.env.current_objectives) > 0:
                    objectives['value'] = self.env.current_objectives[0]
                if 'weight' not in objectives and len(self.env.current_objectives) > 1:
                    objectives['weight'] = self.env.current_objectives[1]
                if 'volume' not in objectives and len(self.env.current_objectives) > 2:
                    objectives['volume'] = self.env.current_objectives[2]
                if 'cost' not in objectives and len(self.env.current_objectives) > 3:
                    objectives['cost'] = self.env.current_objectives[3]
                if 'time' not in objectives and len(self.env.current_objectives) > 4:
                    objectives['time'] = self.env.current_objectives[4]
                
                # 添加已选择物品数量
                objectives['n_selected'] = len(current_solution)
                
                # 记录所有目标函数值
                logging.info(f"初始化了{len(self.env.current_objectives)}个目标函数：{self.env.objective_names}")
                for i, name in enumerate(self.env.objective_names):
                    if i < len(self.env.current_objectives):
                        logging.info(f"目标{i} ({name})值: {self.env.current_objectives[i]}")
                        if i > 0 and i-1 < len(self.env.capacities):
                            logging.info(f"目标{i} ({name})容量: {self.env.capacities[i-1]}")
                
                objectives_list.append(objectives)
        
        # 找到最佳解决方案
        best_idx = int(np.argmax(test_rewards))
        best_solution = solutions[best_idx]
        best_objectives = objectives_list[best_idx] if objectives_list else None
        
        results = {
            'mean_reward': float(np.mean(test_rewards)),
            'std_reward': float(np.std(test_rewards))
        }
        
        if best_objectives:
            results.update({
                'total_value': float(best_objectives['value']),
                'total_weight': float(best_objectives['weight']),
                'total_volume': float(best_objectives['volume']),
                'total_cost': float(best_objectives['cost']),
                'selected_items': int(best_objectives['n_selected'])
            })
        
        # 添加可视化处理
        if visualize and best_objectives:
            try:
                # 创建solutions目录
                os.makedirs('solutions', exist_ok=True)
                
                # 准备可视化所需数据
                from src.utils.visualization import ProblemVisualizer
                visualizer = ProblemVisualizer(use_english_labels=True)  # 强制使用英文标签，避免中文字体问题
                
                # 获取环境的当前状态
                state = self.env.reset()
                for action in best_solution:
                    state, _, _, _ = self.env.step(action)
                
                # 从state中构建问题实例
                problem_data = {}
                
                if isinstance(self.env, KnapsackEnv):
                    # 获取实际目标名称和约束
                    objective_names = self.env.objective_names
                    # 构建容量字典，将每个约束与其名称关联
                    capacities = {}
                    for i in range(1, len(self.env.objective_names)):  # 从1开始，跳过价值目标
                        if i-1 < len(self.env.capacities):
                            name = self.env.objective_names[i]
                            capacities[f"{name}_capacity"] = self.env.capacities[i-1]
                    
                    problem_data = {
                        'items': self.env.problem.items.reshape(-1, self.env.num_objectives),
                        'objective_names': objective_names,
                        'objective_types': self.config.get('model', {}).get('objective_types',
                                          [1] + [-1] * (len(objective_names) - 1))
                    }
                    
                    # 添加所有容量约束
                    problem_data.update(capacities)
                
                # 提取所有解的目标值
                # 确保支持动态数量的目标函数
                num_objectives = len(self.env.current_objectives)
                all_objectives = []
                
                for obj in objectives_list:
                    obj_values = []
                    # 从环境中获取所有目标名称
                    for i, name in enumerate(self.env.objective_names):
                        name_lower = name.lower()
                        # 尝试使用目标名称获取值
                        if name_lower in obj:
                            obj_values.append(obj[name_lower])
                        # 尝试使用通用名称获取值
                        elif i == 0 and 'value' in obj:
                            obj_values.append(obj['value'])
                        elif i == 1 and 'weight' in obj:
                            obj_values.append(obj['weight'])
                        elif i == 2 and 'volume' in obj:
                            obj_values.append(obj['volume'])
                        elif i == 3 and 'cost' in obj:
                            obj_values.append(obj['cost'])
                        elif i == 4 and 'time' in obj:
                            obj_values.append(obj['time'])
                        else:
                            # 对于额外的目标，如果没有具体名称，使用索引
                            obj_key = f'objective_{i}'
                            if obj_key in obj:
                                obj_values.append(obj[obj_key])
                            else:
                                obj_values.append(0)  # 默认值
                    
                    # 确保目标值数量与目标数量匹配
                    while len(obj_values) < num_objectives:
                        obj_values.append(0)  # 添加默认值
                    
                    all_objectives.append(obj_values[:num_objectives])  # 只保留需要的目标数量
                
                if all_objectives:
                    all_objectives = np.array(all_objectives)
                    
                    # 添加当前解
                    solution_obj = []
                    for i in range(num_objectives):
                        if i < len(self.env.current_objectives):
                            solution_obj.append(self.env.current_objectives[i])
                        else:
                            solution_obj.append(0)  # 默认值
                    
                    solution_obj = np.array(solution_obj)
                    all_objectives = np.vstack([all_objectives, solution_obj])
                    
                    # 可视化Pareto前沿
                    visualizer.visualize_pareto_front(all_objectives)
                    
                    # 高维可视化
                    if num_objectives > 3:
                        try:
                            visualizer.visualize_high_dimensional_solution(
                                problem={
                                    'items': problem_data.get('items', self.env.problem.items) if hasattr(self.env, 'problem') else [],  # 正确传递物品数据
                                    'objective_names': self.env.objective_names if hasattr(self.env, 'objective_names') else None,
                                    'objective_types': self.env.problem.objective_types if hasattr(self.env.problem, 'objective_types') else None
                                },
                                solution=np.array(best_solution, dtype=np.int32),  # 使用已转换的best_solution
                                objectives=np.array(all_objectives)
                            )
                        except Exception as e:
                            logging.error(f"高维可视化失败: {e}")
                    else:
                        try:
                            visualizer.visualize_problem_and_solution(problem_data, best_solution)
                        except Exception as e:
                            logging.error(f"问题可视化失败: {e}")
                    
                    logging.info("可视化结果已保存到 solutions 目录")
            except Exception as e:
                logging.error(f"生成可视化时出错: {str(e)}")
        
        return results, best_solution, best_objectives

class RLTrainer:
    def __init__(self, env, config: Dict):
        """初始化RL训练器
        
        Args:
            env: 强化学习环境
            config: 配置字典
        """
        # 绑定环境
        self.env = env
        self.config = config
        
        # 验证配置
        self._validate_config(config)
        
        # 设置日志记录器
        self.logger = logging.getLogger("RLTrainer")
        
        # 设备配置
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"训练使用设备: {self.device}")
        
        # 从环境中获取size
        if hasattr(env, 'size'):
            self.size = env.size
        else:
            # 从配置中获取，如果都没有则使用默认值
            self.size = config['problem'].get('size', 20)
        
        # 从配置中提取参数
        problem_config = config['problem']
        model_config = config['model']
        training_config = config['training']
        
        # 设置训练参数
        self.gamma = training_config.get('gamma', 0.99)  # 折扣因子
        self.batch_size = training_config.get('batch_size', 64)  # 批次大小
        self.buffer_size = training_config.get('buffer_size', 10000)  # 经验回放缓冲区大小
        
        # 探索设置
        self.epsilon = training_config.get('epsilon_start', 1.0)  # 初始探索率
        self.epsilon_end = training_config.get('epsilon_end', 0.01)  # 最终探索率 
        self.epsilon_decay = training_config.get('epsilon_decay', 0.995)  # 探索率衰减
        
        # 更高级的训练选项
        self.entropy_weight = training_config.get('entropy_weight', 0.01)  # 熵权重
        self.value_weight = training_config.get('value_weight', 0.5)  # 价值函数权重
        
        # 梯度裁剪值（如果启用）
        self.grad_clip = training_config.get('gradient_clip', None)
        if self.grad_clip is not None:
            self.logger.info(f"启用梯度裁剪，阈值: {self.grad_clip}")
        
        # 创建经验回放缓冲区
        self.replay_buffer = ReplayBuffer(self.buffer_size)
        
        # 创建策略网络
        self.model = self._create_model(model_config).to(self.device)
        self.logger.info(f"模型结构:\n{self.model}")
        
        # 创建优化器
        self.lr = training_config.get('learning_rate', 0.001)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        
        # 学习率调度器（如果启用）
        self.lr_scheduler = None
        if training_config.get('lr_decay', False):
            lr_decay_step = training_config.get('lr_decay_step', 500)
            lr_decay_rate = training_config.get('lr_decay_rate', 0.95)
            self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, 
                step_size=lr_decay_step, 
                gamma=lr_decay_rate
            )
            self.logger.info(f"启用学习率调度，每 {lr_decay_step} 步衰减到 {lr_decay_rate} 倍")
        
        # 初始化训练指标跟踪
        self.policy_losses = []
        self.value_losses = []
        self.entropies = []
        self.total_losses = []
        self.training_history = []
        
        # 获取问题维度信息
        self.n_objectives = model_config.get('n_objectives', 1)
        self.input_dim = model_config.get('input_dim', 5)
        
        # 早停参数
        self.use_early_stopping = training_config.get('early_stopping', False)
        self.patience = training_config.get('patience', 20)
        self.best_reward = -float('inf')
        self.no_improvement_count = 0

    def _validate_config(self, config: Dict):
        """验证配置参数的完整性"""
        required_params = {
            'training': ['num_episodes', 'buffer_size', 'batch_size', 'gamma', 
                        'learning_rate', 'entropy_weight', 'value_weight'],
            'model': ['input_dim', 'hidden_dim'],
            'problem': ['type']
        }
        
        for section, params in required_params.items():
            if section not in config:
                raise ValueError(f"配置中缺少 {section} 部分")
            for param in params:
                if param not in config[section]:
                    raise ValueError(f"配置中缺少 {section}.{param} 参数")
        
    def _create_model(self, model_config: Dict) -> FastSolver:
        """创建模型实例
        
        Args:
            model_config: 模型配置字典
            
        Returns:
            FastSolver: 模型实例
        """
        try:
            # 获取模型参数
            input_dim = model_config.get('input_dim', 4)  # 默认4维特征
            hidden_dim = model_config.get('hidden_dim', 256)
            n_objectives = model_config.get('n_objectives', 3)
            problem_type = self.config['problem']['type']
            
            # 获取目标类型
            objective_types = self.config['problem'].get('objective_types', None)
            
            self.logger.info(f"创建模型:")
            self.logger.info(f"- 输入维度: {input_dim}")
            self.logger.info(f"- 隐藏层维度: {hidden_dim}")
            self.logger.info(f"- 目标数量: {n_objectives}")
            self.logger.info(f"- 问题类型: {problem_type}")
            self.logger.info(f"- 目标类型: {objective_types}")
            
            # 创建模型实例
            model = FastSolver(
                hidden_dim=hidden_dim,
                problem_type=problem_type,
                input_dim=input_dim,
                n_objectives=n_objectives,
                objective_types=objective_types
            )
            
            self.logger.info("模型创建成功")
            return model
            
        except Exception as e:
            self.logger.error(f"创建模型时出错: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise
    
    def train(self):
        """训练模型"""
        logger = logging.getLogger(__name__)
        logger.info("Starting training...")
        
        try:
            for episode in range(self.config['training']['num_episodes']):
                try:
                    logger.info(f"\n=== 开始第 {episode+1}/{self.config['training']['num_episodes']} 轮训练 ===")
                    
                    # 重置环境
                    try:
                        state = self.env.reset()
                        # 根据状态字典中的键来适配不同环境类型
                        if 'items' in state:  # 背包问题
                            logger.info(f"环境重置成功，状态形状: {state['items'].shape}")
                        elif 'coordinates' in state:  # TSP问题
                            logger.info(f"环境重置成功，状态形状: 坐标={state['coordinates'].shape}，距离矩阵={state['distance_matrix'].shape}")
                        else:
                            logger.info(f"环境重置成功，状态键: {list(state.keys())}")
                    except Exception as e:
                        logger.error(f"重置环境时出错: {str(e)}")
                        raise
                    
                    episode_reward = 0
                    done = False
                    
                    while not done:
                        try:
                            # 处理状态
                            state_tensor = self._process_state(state)
                            logger.debug(f"状态处理成功，张量形状: {state_tensor.shape}")
                            
                            # 获取动作概率和状态值
                            try:
                                action_probs, state_value, _ = self.model(state_tensor)
                                logger.debug(f"模型前向传播成功，动作概率形状: {action_probs.shape}")
                            except Exception as e:
                                logger.error(f"模型前向传播时出错: {str(e)}")
                                logger.error(f"输入状态形状: {state_tensor.shape}")
                                logger.error(f"模型输入维度: {self.model.input_dim}")
                                raise
                            
                            # 创建动作分布并采样
                            try:
                                action_dist = torch.distributions.Categorical(action_probs)
                                action = action_dist.sample()
                                logger.debug(f"动作采样成功: {action.item()}")
                            except Exception as e:
                                logger.error(f"动作采样时出错: {str(e)}")
                                raise
                            
                            # 执行动作
                            try:
                                next_state, reward, done, info = self.env.step(action.item())
                                logger.debug(f"动作执行成功，奖励: {reward}")
                            except Exception as e:
                                logger.error(f"执行动作时出错: {str(e)}")
                                raise
                            
                            # 存储经验
                            try:
                                self.replay_buffer.push(Experience(
                                    state, action.item(), reward, next_state, done
                                ))
                                logger.debug("经验存储成功")
                            except Exception as e:
                                logger.error(f"存储经验时出错: {str(e)}")
                                raise
                            
                            state = next_state
                            episode_reward += reward
                            
                        except Exception as e:
                            logger.error(f"训练步骤执行出错: {str(e)}")
                            logger.error(traceback.format_exc())
                            raise
                    
                    # 处理episode结束
                    try:
                        self._handle_episode_end(episode, episode_reward, info)
                        logger.info(f"第 {episode+1} 轮训练完成，总奖励: {episode_reward}")
                    except Exception as e:
                        logger.error(f"处理episode结束时出错: {str(e)}")
                        raise
                    
                    # 更新模型
                    if len(self.replay_buffer) >= self.batch_size:
                        try:
                            self._update_model()
                            logger.debug("模型更新成功")
                        except Exception as e:
                            logger.error(f"更新模型时出错: {str(e)}")
                            raise
                    
                    # 定期保存检查点
                    if (episode + 1) % self.config['training']['save_interval'] == 0:
                        try:
                            self._save_checkpoint()
                            logger.info(f"保存检查点成功: episode {episode+1}")
                        except Exception as e:
                            logger.error(f"保存检查点时出错: {str(e)}")
                            raise
                    
                    # 清理内存
                    if (episode + 1) % 10 == 0:
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        logger.debug("内存清理完成")
                
                except Exception as e:
                    logger.error(f"第 {episode+1} 轮训练出错: {str(e)}")
                    logger.error(traceback.format_exc())
                    continue
            
            # 保存最终模型
            try:
                self._save_checkpoint()
                logger.info("训练完成，最终模型保存成功")
            except Exception as e:
                logger.error(f"保存最终模型时出错: {str(e)}")
                raise
                
            # 在训练结束时添加：
            self._save_training_losses()
            
        except KeyboardInterrupt:
            logger.info("\n训练被用户中断")
            try:
                self._save_checkpoint(filename="interrupted_checkpoint.pt")
                logger.info("已保存中断时的模型状态到 'interrupted_checkpoint.pt'")
            except Exception as e:
                logger.error(f"保存中断检查点时出错: {str(e)}")
        
        except Exception as e:
            logger.error(f"训练过程中发生严重错误: {str(e)}")
            logger.error(traceback.format_exc())
            try:
                self._save_checkpoint(filename="error_checkpoint.pt")
                logger.info("已保存错误时的模型状态到 'error_checkpoint.pt'")
            except:
                logger.error("无法保存错误时的模型状态")
            raise

    def _process_state(self, state):
        """处理状态为模型输入格式"""
        if isinstance(state, dict):
            # 处理字典类型的状态
            if 'items' in state:  # 背包问题
                return self._process_knapsack_state(state)
            elif 'coordinates' in state:  # TSP问题
                return self._process_tsp_state(state)
            else:
                raise ValueError(f"无法识别的状态格式: {state.keys()}")
        elif isinstance(state, np.ndarray):
            state_tensor = torch.FloatTensor(state).to(self.device)
        elif isinstance(state, list):
            state_tensor = torch.FloatTensor(np.array(state)).to(self.device)
        else:
            state_tensor = state
        
        # 如果是单个状态，添加批次维度
        if hasattr(state_tensor, 'shape') and len(state_tensor.shape) == 1:
            state_tensor = state_tensor.unsqueeze(0)
        
        # 检查并调整特征维度
        if hasattr(state_tensor, 'shape') and len(state_tensor.shape) > 0:
            expected_dim = self.config['model']['input_dim']
            if state_tensor.shape[-1] != expected_dim:
                state_tensor = self._adjust_features(state_tensor, expected_dim)
        
        return state_tensor

    def _process_knapsack_state(self, state):
        """处理背包问题的状态"""
        items = torch.FloatTensor(state['items'])  # [num_categories, items_per_category, n_objectives]
        current_category = state['current_category']
        
        # 如果current_category超出范围，说明episode已经结束
        if current_category >= items.shape[0]:
            # 返回一个全零状态，保持正确的维度
            n_features = self.config['model'].get('input_dim', items.shape[2] + 1)
            return torch.zeros(1, items.shape[1], n_features).to(self.device)
        
        # 获取当前类别的所有物品
        current_items = items[current_category]  # [items_per_category, n_objectives]
        
        # 添加类别信息作为额外特征
        category_info = torch.ones(current_items.shape[0], 1) * current_category
        current_items = torch.cat([current_items, category_info], dim=1)  # [items_per_category, n_objectives+1]
        
        # 检查并调整维度
        expected_dim = self.config['model']['input_dim']
        if current_items.shape[1] != expected_dim:
            current_items = self._adjust_features(current_items, expected_dim)
        
        # 扩展维度并返回 [1, items_per_category, n_objectives+1]
        return current_items.unsqueeze(0).to(self.device)

    def _process_tsp_state(self, state):
        """处理TSP问题的状态"""
        try:
            coordinates = np.array(state['coordinates'])
            current_pos = state['current_pos']
            visited_mask = np.array(state['visited_mask'])
            distance_matrix = np.array(state['distance_matrix'])
            
            # 获取当前城市的坐标
            current_coords = coordinates[current_pos]
            
            # 对未访问城市创建特征矩阵
            unvisited_indices = np.where(visited_mask == 0)[0]
            
            # 记录当前回合的状态信息（仅在调试级别）
            if logging.getLogger().level <= logging.DEBUG:
                logging.debug(f"当前位置: {current_pos}, 未访问城市数: {len(unvisited_indices)}")
                logging.debug(f"未访问城市索引: {unvisited_indices}")
            
            # 如果所有城市都已访问，返回一个特殊的"全部完成"状态表示
            if len(unvisited_indices) == 0:
                n_features = self.config['model'].get('input_dim', 4)
                # 返回一个单行特征向量，表示所有城市已访问
                final_state = torch.zeros(1, 1, n_features).to(self.device)
                # 设置一个特殊标志，表示完成状态
                final_state[0, 0, 0] = 1.0  # 第一个特征设为1，表示完成
                return final_state
            
            # 为每个未访问城市创建特征向量
            features_list = []
            for idx in unvisited_indices:
                # 计算相对坐标 (相对于当前城市的位置)
                relative_x = coordinates[idx, 0] - current_coords[0]
                relative_y = coordinates[idx, 1] - current_coords[1]
                
                # 计算距离和其他特征
                distance = distance_matrix[current_pos, idx]
                visited_ratio = np.sum(visited_mask) / len(visited_mask)
                
                # 增加新特征：剩余城市数量和当前步数
                remaining_cities = len(unvisited_indices) / len(visited_mask)  # 归一化
                current_steps = np.sum(visited_mask) / len(visited_mask)  # 归一化
                
                # 构建特征向量 [x坐标, y坐标, 到当前城市的距离, 已访问比例, 剩余城市比例, 当前步数比例]
                feature = [
                    relative_x,
                    relative_y,
                    distance,
                    visited_ratio,
                    remaining_cities,
                    current_steps
                ]
                
                # 确保特征向量的长度与模型输入维度一致
                expected_dim = self.config['model'].get('input_dim', 4)
                if len(feature) < expected_dim:
                    # 如果特征数量不足，填充零
                    feature.extend([0.0] * (expected_dim - len(feature)))
                elif len(feature) > expected_dim:
                    # 如果特征数量过多，截断
                    feature = feature[:expected_dim]
                
                features_list.append(feature)
            
            # 转换为张量
            features = torch.tensor(features_list, dtype=torch.float32)
            
            # 添加批次维度并返回 [1, num_features, n_dims]
            state_tensor = features.unsqueeze(0).to(self.device)
            
            # 保存有效动作信息到模型，这样预测时可以使用
            if hasattr(self, 'model') and hasattr(self.model, 'set_mask'):
                # 创建城市掩码，将已访问的城市标记为True（掩码）
                mask = torch.ones(1, len(coordinates), dtype=torch.bool).to(self.device)
                # 将未访问的城市设置为False（可用）
                for idx in unvisited_indices:
                    mask[0, idx] = False
                # 设置模型的动作掩码
                self.model.set_mask(mask)
                
            return state_tensor
            
        except Exception as e:
            self.logger.error(f"处理TSP状态时出错: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            # 返回一个安全的默认张量
            return torch.zeros(1, 1, self.config['model'].get('input_dim', 4)).to(self.device)

    def _handle_episode_end(self, episode, episode_reward, info):
        """处理回合结束"""
        self.training_history.append({
            'episode': episode,
            'reward': episode_reward,
            'steps': len(self.training_history),
            'info': info
        })
        
        # 打印训练进度
        if (episode + 1) % self.config['training'].get('log_interval', 10) == 0:
            avg_reward = np.mean([h['reward'] for h in self.training_history[-10:]])
            logging.info(f'Episode {episode+1}/{self.config["training"]["num_episodes"]} - '
                        f'Average Reward: {avg_reward:.2f}')
            
            # 检查是否需要早停
            if self.use_early_stopping:
                if avg_reward > self.best_reward:
                    self.best_reward = avg_reward
                    self.no_improvement_count = 0
                    # 保存最佳模型
                    self._save_checkpoint(filename="best_model.pt")
                    logging.info(f"新的最佳平均奖励: {avg_reward:.2f}，已保存最佳模型")
                else:
                    self.no_improvement_count += 1
                    if self.no_improvement_count >= self.patience:
                        logging.info(f"触发早停: {self.patience} 轮无改善")
                        return True  # 触发早停
        
        return False  # 不触发早停

    def _update_model(self):
        """更新模型参数"""
        if len(self.replay_buffer) < self.batch_size:
            return
        
        try:
            # 从缓冲区采样
            batch = self.replay_buffer.sample(self.batch_size)
            
            # 解包经验
            states, actions, rewards, next_states, dones = zip(*batch)
            
            # 转换为张量
            states_tensor = []
            for state in states:
                if isinstance(state, dict):
                    # 处理字典状态
                    state_tensor = self._process_state(state)
                    states_tensor.append(state_tensor)
                    if self.logger.level <= logging.INFO and len(states_tensor) <= 3:
                        self.logger.info(f"状态 {len(states_tensor)-1} 原始类型: {type(state)}")
                        self.logger.info(f"状态 {len(states_tensor)-1} 张量形状: {state_tensor.shape}")
                else:
                    # 处理张量状态
                    states_tensor.append(state)
            
            if not states_tensor:
                self.logger.warning("没有有效状态可以处理")
                return
            
            # 将状态张量堆叠为批次
            states_batch = torch.cat(states_tensor, dim=0)
            
            # 转换其他数据为张量
            input_dim = self.config['model'].get('input_dim', 4)
            self.logger.info(f"模型期望的输入维度: {input_dim}")
            
            # 调整特征维度以匹配模型输入
            states_batch = self._adjust_features(states_batch, input_dim)
            self.logger.info(f"状态批次形状: {states_batch.shape}")
            
            actions_batch = torch.tensor(actions, dtype=torch.int64).to(self.device)
            self.logger.info(f"动作批次形状: {actions_batch.shape}")
            
            rewards_batch = torch.tensor(rewards, dtype=torch.float32).to(self.device)
            self.logger.info(f"奖励批次形状: {rewards_batch.shape}")
            
            # 处理下一个状态
            if any(isinstance(s, dict) for s in next_states):
                next_states_tensor = []
                for next_state in next_states:
                    if isinstance(next_state, dict):
                        next_state_tensor = self._process_state(next_state)
                        next_states_tensor.append(next_state_tensor)
                    else:
                        next_states_tensor.append(next_state)
                next_states_batch = torch.cat(next_states_tensor, dim=0)
                next_states_batch = self._adjust_features(next_states_batch, input_dim)
            else:
                next_states_batch = torch.tensor(next_states, dtype=torch.float32).to(self.device)
            
            dones_batch = torch.tensor(dones, dtype=torch.float32).to(self.device)
            
            # 模型前向传播 - 注意FastSolver返回三个值
            action_probs, state_values, _ = self.model(states_batch)
            self.logger.info(f"模型前向传播成功，动作概率形状: {action_probs.shape}, 状态值形状: {state_values.shape}")
            
            # 计算策略损失 (Actor)
            # 在TSP问题中，动作索引可能会超出范围，需要特殊处理
            log_probs = torch.log(action_probs + 1e-10)
            
            # 确保动作索引在有效范围内
            max_action_idx = action_probs.size(1) - 1
            valid_actions = []
            
            # 优化TSP问题的动作索引处理
            if self.config['problem']['type'] == 'tsp':
                # 对于TSP问题，我们需要特殊处理动作索引
                # 因为随着城市被访问，可选动作数量会减少
                for i, action in enumerate(actions_batch):
                    if action > max_action_idx:
                        # 如果动作索引超出范围，检查是否有其他有效动作
                        self.logger.warning(f"动作索引 {action.item()} 超出范围 [0, {max_action_idx}]")
                        # 将超出范围的动作转换为有效范围内的随机动作
                        valid_action = torch.randint(0, max_action_idx + 1, (1,)).item()
                        self.logger.info(f"替换为随机有效动作: {valid_action}")
                        valid_actions.append(valid_action)
                    else:
                        valid_actions.append(action.item())
            else:
                # 对于背包问题，保持原有处理方式
                for i, action in enumerate(actions_batch):
                    if action > max_action_idx:
                        # 如果动作索引超出范围，裁剪并记录
                        self.logger.warning(f"动作索引 {action.item()} 被裁剪为 {max_action_idx}")
                        valid_actions.append(max_action_idx)
                    else:
                        valid_actions.append(action.item())
            
            valid_actions_batch = torch.tensor(valid_actions, dtype=torch.int64).to(self.device)
            
            # 使用有效动作索引计算策略损失
            selected_log_probs = log_probs.gather(1, valid_actions_batch.unsqueeze(1))
            
            # 获取下一个状态的价值 - 同样处理三个返回值
            _, next_state_values, _ = self.model(next_states_batch)
            next_state_values = next_state_values.detach()
            
            # 计算目标值 (Critic)
            target_values = rewards_batch.unsqueeze(1) + (1 - dones_batch.unsqueeze(1)) * self.gamma * next_state_values
            
            # 价值损失 (Critic)
            value_loss = F.mse_loss(state_values, target_values.detach())
            
            # 优势函数
            advantage = (target_values - state_values).detach()
            
            # 策略损失 (Actor)
            policy_loss = -(selected_log_probs * advantage).mean()
            
            # 熵损失 (用于鼓励探索)
            entropy = -(action_probs * log_probs).sum(dim=1).mean()
            
            # 总损失
            loss = policy_loss + self.value_weight * value_loss - self.entropy_weight * entropy
            
            # 优化
            self.optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪（如果启用）
            if self.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            
            self.optimizer.step()
            
            # 更新学习率（如果启用）
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            
            # 记录损失
            self.policy_losses.append(policy_loss.item())
            self.value_losses.append(value_loss.item())
            self.entropies.append(entropy.item())
            
            # 每10轮打印一次损失
            if len(self.policy_losses) % 10 == 0:
                self.logger.info(
                    f"损失信息 - 策略损失: {policy_loss.item():.4f}, "
                    f"价值损失: {value_loss.item():.4f}, "
                    f"熵: {entropy.item():.4f}"
                )
        except Exception as e:
            self.logger.error(f"更新模型过程中出错: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())

    def _adjust_features(self, tensor, target_dim):
        """调整特征维度以匹配目标维度"""
        current_dim = tensor.shape[-1]
        if current_dim == target_dim:
            return tensor
        
        if current_dim > target_dim:
            # 截断多余特征
            return tensor[..., :target_dim]
        else:
            # 添加零填充
            padding = torch.zeros(*tensor.shape[:-1], target_dim - current_dim, device=tensor.device)
            return torch.cat([tensor, padding], dim=-1)

    def _save_checkpoint(self, filename="latest_model.pt"):
        """保存模型检查点"""
        # 确保目录存在
        checkpoint_dir = os.path.join('checkpoints', self.config['problem']['type'])
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # 直接在problem目录下保存模型
        checkpoint_path = os.path.join(checkpoint_dir, filename)
        
        # 保存模型状态
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if hasattr(self, 'optimizer') else None,
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict() if hasattr(self, 'policy_optimizer') else None,
            'value_optimizer_state_dict': self.value_optimizer.state_dict() if hasattr(self, 'value_optimizer') else None,
            'config': self.config,
            'episode': self.current_episode if hasattr(self, 'current_episode') else 0
        }
        
        torch.save(checkpoint, checkpoint_path)
        logging.info(f"模型已保存到: {checkpoint_path}")

    def load_model(self, checkpoint_path):
        """加载模型检查点
        
        Args:
            checkpoint_path: 检查点文件路径
        """
        try:
            self.logger.info(f"加载模型检查点: {checkpoint_path}")
            
            # 添加numpy相关类型到安全全局变量
            import torch.serialization
            import numpy
            torch.serialization.add_safe_globals([
                numpy.core.multiarray.scalar,
                numpy.dtype,
                numpy.ndarray,
                numpy._globals._NoValue,
                numpy.ufunc,
                numpy.generic,
                numpy.core._multiarray_umath._get_implementing_args
            ])
            
            # 加载检查点，设置weights_only=False以支持numpy类型
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            
            # 尝试加载模型状态
            try:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            except Exception as e:
                self.logger.warning(f"使用原始模型结构加载失败，尝试适配新结构: {e}")
                
                # 创建状态字典映射
                old_state_dict = checkpoint['model_state_dict']
                new_state_dict = {}
                
                # 映射特征网络的权重
                for key in old_state_dict:
                    if 'feature_network' in key:
                        new_state_dict[key] = old_state_dict[key]
                    elif 'policy_network' in key:
                        # 将策略网络的权重映射到决策网络
                        new_key = key.replace('policy_network', 'decision_network')
                        new_state_dict[new_key] = old_state_dict[key]
                
                # 加载映射后的状态字典
                self.model.load_state_dict(new_state_dict, strict=False)
                self.logger.info("成功适配并加载新的模型结构")
            
            # 加载优化器状态（如果存在）
            if 'optimizer_state_dict' in checkpoint and hasattr(self, 'optimizer'):
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # 加载训练历史记录（如果存在）
            if 'training_history' in checkpoint:
                self.training_history = checkpoint['training_history']
            
            if 'policy_losses' in checkpoint:
                self.policy_losses = checkpoint['policy_losses']
                
            if 'value_losses' in checkpoint:
                self.value_losses = checkpoint['value_losses']
                
            if 'total_losses' in checkpoint:
                self.total_losses = checkpoint['total_losses']
                
            if 'episode' in checkpoint:
                self.episode = checkpoint['episode']
            
            self.logger.info("模型加载成功")
            
        except Exception as e:
            self.logger.error(f"加载模型时出错: {str(e)}")
            raise e

    def test(self, visualize=False):
        """测试模型"""
        self.model.eval()
        test_rewards = []
        solutions = []
        objectives_list = []
        
        for episode in range(10):  # 测试10个episode
            state = self.env.reset()
            done = False
            episode_reward = 0
            current_solution = []
            
            while not done:
                # 处理状态
                state_tensor = self._process_state(state)
                
                # 获取动作概率
                with torch.no_grad():
                    action_probs, _, _ = self.model(state_tensor)
                
                # 选择动作
                action = torch.argmax(action_probs).item()
                
                # 执行动作
                next_state, reward, done, _ = self.env.step(action)
                episode_reward += reward
                state = next_state
                current_solution.append(action)
            
            test_rewards.append(episode_reward)
            solutions.append(current_solution)
            
            # 计算目标值
            if isinstance(self.env, KnapsackEnv):
                # 使用current_objectives替代单独的current_value等属性
                objectives = {}
                
                # 确保目标与名称一致
                for i, name in enumerate(self.env.objective_names):
                    if i < len(self.env.current_objectives):
                        objectives[name.lower()] = self.env.current_objectives[i]
                    else:
                        objectives[name.lower()] = 0
                
                # 为兼容性保留原始键名
                if 'value' not in objectives and len(self.env.current_objectives) > 0:
                    objectives['value'] = self.env.current_objectives[0]
                if 'weight' not in objectives and len(self.env.current_objectives) > 1:
                    objectives['weight'] = self.env.current_objectives[1]
                if 'volume' not in objectives and len(self.env.current_objectives) > 2:
                    objectives['volume'] = self.env.current_objectives[2]
                if 'cost' not in objectives and len(self.env.current_objectives) > 3:
                    objectives['cost'] = self.env.current_objectives[3]
                if 'time' not in objectives and len(self.env.current_objectives) > 4:
                    objectives['time'] = self.env.current_objectives[4]
                
                # 添加已选择物品数量
                objectives['n_selected'] = len(current_solution)
                
                # 记录所有目标函数值
                logging.info(f"初始化了{len(self.env.current_objectives)}个目标函数：{self.env.objective_names}")
                for i, name in enumerate(self.env.objective_names):
                    if i < len(self.env.current_objectives):
                        logging.info(f"目标{i} ({name})值: {self.env.current_objectives[i]}")
                        if i > 0 and i-1 < len(self.env.capacities):
                            logging.info(f"目标{i} ({name})容量: {self.env.capacities[i-1]}")
                
                objectives_list.append(objectives)
        
        # 找到最佳解决方案
        best_idx = int(np.argmax(test_rewards))
        best_solution = solutions[best_idx]
        best_objectives = objectives_list[best_idx] if objectives_list else None
        
        results = {
            'mean_reward': float(np.mean(test_rewards)),
            'std_reward': float(np.std(test_rewards))
        }
        
        if best_objectives:
            results.update({
                'total_value': float(best_objectives['value']),
                'total_weight': float(best_objectives['weight']),
                'total_volume': float(best_objectives['volume']),
                'total_cost': float(best_objectives['cost']),
                'selected_items': int(best_objectives['n_selected'])
            })
        
        # 添加可视化处理
        if visualize and best_objectives:
            try:
                # 创建solutions目录
                os.makedirs('solutions', exist_ok=True)
                
                # 准备可视化所需数据
                from src.utils.visualization import ProblemVisualizer
                visualizer = ProblemVisualizer(use_english_labels=True)  # 强制使用英文标签，避免中文字体问题
                
                # 获取环境的当前状态
                state = self.env.reset()
                for action in best_solution:
                    state, _, _, _ = self.env.step(action)
                
                # 从state中构建问题实例
                problem_data = {}
                
                if isinstance(self.env, KnapsackEnv):
                    # 获取实际目标名称和约束
                    objective_names = self.env.objective_names
                    # 构建容量字典，将每个约束与其名称关联
                    capacities = {}
                    for i in range(1, len(self.env.objective_names)):  # 从1开始，跳过价值目标
                        if i-1 < len(self.env.capacities):
                            name = self.env.objective_names[i]
                            capacities[f"{name}_capacity"] = self.env.capacities[i-1]
                    
                    problem_data = {
                        'items': self.env.problem.items.reshape(-1, self.env.num_objectives),
                        'objective_names': objective_names,
                        'objective_types': self.config.get('model', {}).get('objective_types',
                                          [1] + [-1] * (len(objective_names) - 1))
                    }
                    
                    # 添加所有容量约束
                    problem_data.update(capacities)
                
                # 提取所有解的目标值
                # 确保支持动态数量的目标函数
                num_objectives = len(self.env.current_objectives)
                all_objectives = []
                
                for obj in objectives_list:
                    obj_values = []
                    # 从环境中获取所有目标名称
                    for i, name in enumerate(self.env.objective_names):
                        name_lower = name.lower()
                        # 尝试使用目标名称获取值
                        if name_lower in obj:
                            obj_values.append(obj[name_lower])
                        # 尝试使用通用名称获取值
                        elif i == 0 and 'value' in obj:
                            obj_values.append(obj['value'])
                        elif i == 1 and 'weight' in obj:
                            obj_values.append(obj['weight'])
                        elif i == 2 and 'volume' in obj:
                            obj_values.append(obj['volume'])
                        elif i == 3 and 'cost' in obj:
                            obj_values.append(obj['cost'])
                        elif i == 4 and 'time' in obj:
                            obj_values.append(obj['time'])
                        else:
                            # 对于额外的目标，如果没有具体名称，使用索引
                            obj_key = f'objective_{i}'
                            if obj_key in obj:
                                obj_values.append(obj[obj_key])
                            else:
                                obj_values.append(0)  # 默认值
                    
                    # 确保目标值数量与目标数量匹配
                    while len(obj_values) < num_objectives:
                        obj_values.append(0)  # 添加默认值
                    
                    all_objectives.append(obj_values[:num_objectives])  # 只保留需要的目标数量
                
                if all_objectives:
                    all_objectives = np.array(all_objectives)
                    
                    # 添加当前解
                    solution_obj = []
                    for i in range(num_objectives):
                        if i < len(self.env.current_objectives):
                            solution_obj.append(self.env.current_objectives[i])
                        else:
                            solution_obj.append(0)  # 默认值
                    
                    solution_obj = np.array(solution_obj)
                    all_objectives = np.vstack([all_objectives, solution_obj])
                    
                    # 可视化Pareto前沿
                    visualizer.visualize_pareto_front(all_objectives)
                    
                    # 高维可视化
                    if num_objectives > 3:
                        try:
                            visualizer.visualize_high_dimensional_solution(
                                problem={
                                    'items': problem_data.get('items', self.env.problem.items) if hasattr(self.env, 'problem') else [],  # 正确传递物品数据
                                    'objective_names': self.env.objective_names if hasattr(self.env, 'objective_names') else None,
                                    'objective_types': self.env.problem.objective_types if hasattr(self.env.problem, 'objective_types') else None
                                },
                                solution=np.array(best_solution, dtype=np.int32),  # 使用已转换的best_solution
                                objectives=np.array(all_objectives)
                            )
                        except Exception as e:
                            logging.error(f"高维可视化失败: {e}")
                    else:
                        try:
                            visualizer.visualize_problem_and_solution(problem_data, best_solution)
                        except Exception as e:
                            logging.error(f"问题可视化失败: {e}")
                    
                    logging.info("可视化结果已保存到 solutions 目录")
            except Exception as e:
                logging.error(f"生成可视化时出错: {str(e)}")
        
        return results, best_solution, best_objectives

    def _save_training_losses(self):
        """保存训练损失图表"""
        # 确保训练有损失记录
        if not hasattr(self, 'policy_losses') or not self.policy_losses:
            self.logger.warning("没有可用的训练损失记录")
            return
        
        try:
            # 确保目录存在
            os.makedirs('solutions', exist_ok=True)
            
            # 设置图表样式
            plt.figure(figsize=(10, 6))
            
            # 使用英文标题和标签，避免中文乱码
            plt.title('Reinforcement Learning Training Loss')
            plt.xlabel('Episode')
            plt.ylabel('Loss')
            
            # 绘制策略损失和价值损失曲线
            if self.policy_losses:
                plt.plot(self.policy_losses, label='Policy Loss')
            if hasattr(self, 'value_losses') and self.value_losses:
                plt.plot(self.value_losses, label='Value Loss')
            if hasattr(self, 'total_losses') and self.total_losses:
                plt.plot(self.total_losses, label='Total Loss')
            plt.legend()
            
            # 绘制奖励曲线（如果有）
            if hasattr(self, 'rewards') and self.rewards:
                ax2 = plt.twinx()
                ax2.set_ylabel('Average Reward', color='g')
                ax2.plot(self.rewards, 'g-', label='Average Reward')
                ax2.tick_params(axis='y', labelcolor='g')
                
            # 只保存图表到solutions目录
            plt.savefig('solutions/latest_training_loss.png')
            plt.close()
            
            self.logger.info("训练损失图表已保存到solutions/latest_training_loss.png")
            
        except Exception as e:
            self.logger.warning(f"保存训练损失图表时出错: {str(e)}")
            import traceback
            self.logger.warning(traceback.format_exc())

def print_system_info():
    """打印系统信息，用于调试"""
    print("\n" + "="*50)
    print("系统信息:")
    print(f"操作系统: {platform.system()} {platform.version()}")
    print(f"Python版本: {sys.version}")
    print(f"PyTorch版本: {torch.__version__}")
    
    # GPU信息
    if torch.cuda.is_available():
        print(f"CUDA是否可用: 是")
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"GPU数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"当前GPU内存使用: {torch.cuda.memory_allocated(i)/1024**2:.2f} MB")
            print(f"当前GPU内存缓存: {torch.cuda.memory_reserved(i)/1024**2:.2f} MB")
    else:
        print("CUDA是否可用: 否")
    
    # CPU和内存信息
    print(f"CPU核心数: {psutil.cpu_count(logical=True)}")
    print(f"物理CPU核心数: {psutil.cpu_count(logical=False)}")
    print(f"系统内存: {psutil.virtual_memory().total / (1024**3):.2f} GB")
    print(f"可用内存: {psutil.virtual_memory().available / (1024**3):.2f} GB")
    print("="*50 + "\n")

def check_config(config):
    """检查配置是否有问题，并打印警告"""
    print("\n" + "="*50)
    print("配置检查:")
    
    # 检查输入维度
    if config['model']['input_dim'] != 6 and config['problem']['type'] == 'knapsack':
        print(f"警告: 模型输入维度({config['model']['input_dim']})与状态批次形状(6)不匹配!")
        print(f"建议将config.yaml中的model.input_dim修改为6")
    
    # 检查批次大小
    if config['training']['batch_size'] < 8:
        print(f"警告: 批次大小({config['training']['batch_size']})过小，可能影响训练效果")
    
    # 检查学习率
    if config['training']['learning_rate'] > 0.01:
        print(f"警告: 学习率({config['training']['learning_rate']})可能过大")
    
    # 检查问题配置
    if config['problem']['type'] == 'knapsack':
        expected_size = config['problem']['num_categories'] * config['problem']['items_per_category']
        if config['problem']['size'] != expected_size:
            print(f"警告: 问题规模({config['problem']['size']})与类别数*每类物品数({expected_size})不匹配")
    
    print("="*50 + "\n")

def main():
    """主函数"""
    try:
        # 设置日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'logs/train_{time.strftime("%Y%m%d_%H%M%S")}.log'),
                logging.StreamHandler()
            ]
        )
        logger = logging.getLogger(__name__)
        
        # 打印系统信息
        try:
            print_system_info()
        except Exception as e:
            logger.error(f"打印系统信息时出错: {str(e)}")
        
        # 加载配置
        parser = argparse.ArgumentParser(description='Train a model for the given problem.')
        parser.add_argument('--config', type=str, default='config.yaml', help='Path to the config file')
        args = parser.parse_args()
        
        try:
            with open(args.config, 'r') as f:
                config = yaml.safe_load(f)
            logger.info("成功加载配置文件")
            logger.info(f"配置内容: {config}")
        except Exception as e:
            logger.error(f"加载配置文件时出错: {str(e)}")
            raise
        
        # 检查配置
        try:
            check_config(config)
        except Exception as e:
            logger.error(f"检查配置时出错: {str(e)}")
        
        # 创建环境
        try:
            if config['problem']['type'] == 'knapsack':
                # 为问题实例准备专门的配置
                problem_config = {
                    'problem': {
                        'type': 'knapsack',
                        'num_categories': config['problem'].get('num_categories', 5),
                        'items_per_category': config['problem'].get('items_per_category', 4),
                        'num_objectives': config['problem'].get('num_objectives', 3)  # 确保总是有一个默认值
                    },
                    'model': {
                        'objective_names': config.get('model', {}).get('objective_names', []),
                        'objective_types': config.get('model', {}).get('objective_types', [1] + [-1] * (config['problem'].get('num_objectives', 3) - 1))
                    }
                }
                
                # 记录问题配置
                logger.info(f"创建问题实例，配置:")
                logger.info(f"  - 问题类型: 背包问题")
                logger.info(f"  - 目标数量: {problem_config['problem']['num_objectives']}")
                logger.info(f"  - 目标名称: {problem_config['model']['objective_names']}")
                logger.info(f"  - 目标类型: {problem_config['model']['objective_types']}")
                
                # 使用专门配置创建问题实例
                problem = problem_factory.create_problem('knapsack', problem_config)
                
                # 使用问题实例创建环境
                env = KnapsackEnv(problem)
                
                logger.info("成功创建背包问题环境")
                logger.info(f"环境目标数量: {problem.num_objectives}")
                logger.info(f"环境目标名称: {problem.objective_names}")
                
            elif config['problem']['type'] == 'tsp':
                # 为TSP问题准备专门的配置
                problem_config = {
                    'problem': {
                        'type': 'tsp',
                        'size': config['problem'].get('size', 20),
                        'num_objectives': config['problem'].get('num_objectives', 3)
                    },
                    'model': {
                        'objective_names': config.get('model', {}).get('objective_names', []),
                        'objective_types': config.get('model', {}).get('objective_types', [1] + [-1] * (config['problem'].get('num_objectives', 3) - 1))
                    }
                }
                
                # 记录问题配置
                logger.info(f"创建问题实例，配置:")
                logger.info(f"  - 问题类型: TSP问题")
                logger.info(f"  - 城市数量: {problem_config['problem']['size']}")
                logger.info(f"  - 目标数量: {problem_config['problem']['num_objectives']}")
                
                # 为TSP创建环境（此处可能需要修改成基于问题的实例化方式）
                env = TSPEnv(
                    num_cities=problem_config['problem']['size'],
                    num_objectives=problem_config['problem']['num_objectives']
                )
                logger.info("成功创建TSP问题环境")
            else:
                raise ValueError(f"不支持的问题类型: {config['problem']['type']}")
        except Exception as e:
            logger.error(f"创建环境时出错: {str(e)}")
            logger.error(f"配置详情: {config}")
            logger.error(traceback.format_exc())
            raise
        
        # 创建训练器并训练
        try:
            trainer = RLTrainer(env, config)
            logger.info("成功创建训练器")
            
            # 开始训练
            logger.info("开始训练...")
            trainer.train()
            
        except Exception as e:
            logger.error(f"训练过程中出错: {str(e)}")
            logger.error("详细错误信息:")
            logger.error(traceback.format_exc())
            raise
            
    except Exception as e:
        logger.error(f"程序执行过程中发生严重错误: {str(e)}")
        logger.error("详细错误信息:")
        logger.error(traceback.format_exc())
        raise

if __name__ == '__main__':
    main()
