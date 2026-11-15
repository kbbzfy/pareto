import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict
import numpy as np
import logging
import traceback

class FastSolver(nn.Module):
    def __init__(self, hidden_dim: int, problem_type: str, input_dim: int = 4, n_objectives: int = 3, objective_types: list = None):
        """
        初始化多目标求解模型
        
        Args:
            input_dim: 输入维度 (物品属性数量 + 类别信息)，默认80对应knapsack问题
            hidden_dim: 隐藏层维度
            problem_type: 问题类型 ('tsp' 或 'knapsack')
            n_objectives: 目标函数的数量
            objective_types: 目标函数类型列表 (1表示最大化，-1表示最小化)
        """
        super().__init__()
        
        # 设置日志记录器
        self.logger = logging.getLogger(__name__)
        
        try:
            self.input_dim = input_dim
            self.hidden_dim = hidden_dim
            self.problem_type = problem_type
            self.n_objectives = n_objectives
            
            # 设置目标类型（最大化/最小化）
            if objective_types is None:
                # 默认：第一个目标（价值）最大化，其他目标最小化
                self.objective_types = [1] + [-1] * (n_objectives - 1)
            else:
                self.objective_types = objective_types
            
            # 添加设备属性
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # 记录初始化信息
            self.logger.info(f"初始化FastSolver模型:")
            self.logger.info(f"- 输入维度: {input_dim}")
            self.logger.info(f"- 隐藏层维度: {hidden_dim}")
            self.logger.info(f"- 问题类型: {problem_type}")
            self.logger.info(f"- 目标函数数量: {n_objectives}")
            self.logger.info(f"- 目标类型: {self.objective_types}")
            self.logger.info(f"- 使用设备: {self.device}")
            
            # 编码器：初步编码到高维特征
            try:
                self.encoder = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                )
                self.logger.debug("编码器创建成功")
            except Exception as e:
                self.logger.error(f"创建编码器时出错: {str(e)}")
                raise
            
            # 特征处理：高维特征处理
            try:
                self.feature_processor = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.1),  # 添加dropout以增强泛化能力
                )
                self.logger.debug("特征处理器创建成功")
            except Exception as e:
                self.logger.error(f"创建特征处理器时出错: {str(e)}")
                raise
            
            # 解码器：前馈神经网络
            try:
                self.decoder = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, 1),  # 输出一个值，表示选择该物品的概率
                )
                self.logger.debug("解码器创建成功")
            except Exception as e:
                self.logger.error(f"创建解码器时出错: {str(e)}")
                raise
            
            # 多目标输出层：预测每个目标函数的值
            try:
                self.objective_head = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.ReLU(),
                    nn.Linear(hidden_dim // 2, n_objectives)
                )
                self.logger.debug("目标函数输出层创建成功")
            except Exception as e:
                self.logger.error(f"创建目标函数输出层时出错: {str(e)}")
                raise
            
            # 将模型移动到正确的设备
            try:
                self.to(self.device)
                self.logger.info(f"模型成功移动到设备: {self.device}")
            except Exception as e:
                self.logger.error(f"将模型移动到设备时出错: {str(e)}")
                raise
                
        except Exception as e:
            self.logger.error(f"FastSolver初始化失败: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise
            
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: 输入特征 [batch_size, n_items, input_feat_dim]
               input_feat_dim可能与self.input_dim不同
            
        Returns:
            action_probs: 动作概率 [batch_size, n_items]
            state_value: 状态值 [batch_size, 1]
            objectives: 预测的目标函数值 [batch_size, n_objectives]
        """
        try:
            # 打印输入张量信息
            self.logger.debug(f"\n前向传播:")
            self.logger.debug(f"输入张量形状: {x.shape}")
            self.logger.debug(f"输入张量类型: {x.dtype}")
            self.logger.debug(f"输入张量设备: {x.device}")
            self.logger.debug(f"模型期望输入维度: {self.input_dim}")
            
            # 确保输入维度正确
            if len(x.shape) == 2:  # 如果输入是[batch_size, input_dim]
                self.logger.debug("输入形状为2维，扩展为3维")
                x = x.unsqueeze(1)  # 转换为[batch_size, 1, input_dim]
            
            batch_size, n_items, feat_dim = x.shape
            self.logger.debug(f"处理后的形状: 批次大小={batch_size}, 项目数={n_items}, 特征维度={feat_dim}")
            
            # 检查输入特征维度是否与模型期望的一致
            if feat_dim != self.input_dim:
                self.logger.warning(f"特征维度不匹配: 输入={feat_dim}, 期望={self.input_dim}")
                # 如果特征维度不匹配，需要进行调整
                device = x.device
                if feat_dim < self.input_dim:
                    # 特征数量不足，用零填充
                    self.logger.debug(f"特征维度不足，进行零填充: {feat_dim} -> {self.input_dim}")
                    padding = torch.zeros(batch_size, n_items, self.input_dim - feat_dim).to(device)
                    x = torch.cat([x, padding], dim=2)
                else:
                    # 特征数量过多，截断
                    self.logger.debug(f"特征维度过多，进行截断: {feat_dim} -> {self.input_dim}")
                    x = x[:, :, :self.input_dim]
                
                self.logger.debug(f"调整后的张量形状: {x.shape}")
            
            # 重塑输入以便处理
            try:
                x_flat = x.reshape(-1, self.input_dim)  # [batch_size * n_items, input_dim]
                self.logger.debug(f"重塑后的张量形状: {x_flat.shape}")
            except RuntimeError as e:
                self.logger.error(f"重塑张量时出错: {str(e)}")
                self.logger.error(f"当前张量形状: {x.shape}, 目标形状: [{batch_size * n_items}, {self.input_dim}]")
                raise
            
            # 编码器：初步编码
            try:
                encoded_features = self.encoder(x_flat)  # [batch_size * n_items, hidden_dim]
                self.logger.debug(f"编码器输出形状: {encoded_features.shape}")
            except Exception as e:
                self.logger.error(f"编码器处理时出错: {str(e)}")
                self.logger.error(f"编码器输入形状: {x_flat.shape}")
                raise
            
            # 特征处理：高维特征处理
            try:
                processed_features = self.feature_processor(encoded_features)  # [batch_size * n_items, hidden_dim]
                self.logger.debug(f"特征处理器输出形状: {processed_features.shape}")
            except Exception as e:
                self.logger.error(f"特征处理时出错: {str(e)}")
                raise
            
            # 解码器：生成选择概率
            try:
                decisions = self.decoder(processed_features)  # [batch_size * n_items, 1]
                self.logger.debug(f"解码器输出形状: {decisions.shape}")
            except Exception as e:
                self.logger.error(f"解码器处理时出错: {str(e)}")
                raise
            
            # 检查重塑前的实际大小
            total_elements = decisions.numel()
            expected_elements = batch_size * n_items
            self.logger.debug(f"决策张量元素数: {total_elements}, 期望元素数: {expected_elements}")
            
            # 如果大小不匹配，调整维度
            if total_elements != expected_elements:
                self.logger.warning(f"警告: 实际元素数与期望元素数不匹配")
                # 计算实际应该的n_items
                actual_n_items = total_elements // batch_size
                self.logger.debug(f"重新计算的项目数: {actual_n_items}")
                decisions = decisions.view(batch_size, actual_n_items)
            else:
                # 重塑回原始维度
                try:
                    decisions = decisions.view(batch_size, n_items)  # [batch_size, n_items]
                    self.logger.debug(f"重塑后的决策张量形状: {decisions.shape}")
                except RuntimeError as e:
                    self.logger.error(f"重塑决策张量时出错: {str(e)}")
                    self.logger.error(f"当前决策张量形状: {decisions.shape}, 目标形状: [{batch_size}, {n_items}]")
                    raise
            
            # 对于背包问题使用sigmoid，对于TSP使用softmax
            try:
                if self.problem_type == 'knapsack':
                    action_probs = torch.sigmoid(decisions)  # 每个物品独立的选择概率
                else:
                    # TSP问题使用softmax，但需要考虑已访问城市的掩码
                    # 应用动作掩码（如果存在）
                    if hasattr(self, 'mask') and self.mask is not None:
                        # 确保掩码和决策形状匹配
                        if self.mask.shape[1] == decisions.shape[1]:
                            # 将已访问城市的logits设置为一个很小的值（负无穷）
                            decisions = decisions.masked_fill(self.mask, float('-inf'))
                        else:
                            self.logger.warning(f"掩码形状({self.mask.shape})与决策形状({decisions.shape})不匹配，跳过掩码应用")
                    
                    # 应用softmax得到概率分布
                    action_probs = F.softmax(decisions, dim=-1)  # TSP需要选择下一个城市
                
                self.logger.debug(f"动作概率张量形状: {action_probs.shape}")
            except Exception as e:
                self.logger.error(f"计算动作概率时出错: {str(e)}")
                raise
            
            # 使用平均池化得到状态值
            try:
                state_value = decisions.mean(dim=1, keepdim=True)
                self.logger.debug(f"状态值张量形状: {state_value.shape}")
            except Exception as e:
                self.logger.error(f"计算状态值时出错: {str(e)}")
                raise
            
            # 预测目标函数值
            try:
                # 使用全局平均池化聚合特征
                global_features = processed_features.view(batch_size, -1, self.hidden_dim).mean(dim=1)  # [batch_size, hidden_dim]
                self.logger.debug(f"全局特征张量形状: {global_features.shape}")
                objectives = self.objective_head(global_features)  # [batch_size, n_objectives]
                self.logger.debug(f"目标函数值张量形状: {objectives.shape}")
            except Exception as e:
                self.logger.error(f"计算目标函数值时出错: {str(e)}")
                raise
            
            return action_probs, state_value, objectives
            
        except Exception as e:
            self.logger.error(f"前向传播过程中发生未捕获的错误: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise
        
    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        预测最佳动作
        
        Args:
            x: 输入状态 [batch_size, input_dim]
            
        Returns:
            action: 预测的最佳动作 [batch_size]
            objectives: 预测的目标函数值 [batch_size, n_objectives]
        """
        self.eval()
        with torch.no_grad():
            action_probs, state_value, objectives = self.forward(x)
            
            if self.problem_type == 'knapsack':
                # 使用阈值方法选择物品，而不是argmax
                threshold = 0.5  # 可以根据需要调整这个阈值
                action = (action_probs > threshold).float()
                
                # 确保至少选择一个物品
                if action.sum() == 0:
                    # 如果没有物品被选中，选择概率最高的物品
                    max_prob_item = torch.argmax(action_probs, dim=-1)
                    action = torch.zeros_like(action_probs)
                    action.scatter_(1, max_prob_item.unsqueeze(1), 1)
            else:
                # 对于TSP问题，确保选择的动作是有效的（未访问的城市）
                if hasattr(self, 'mask') and self.mask is not None:
                    # 创建有效动作的概率分布（已经在forward中应用了掩码）
                    valid_probs = action_probs.clone()
                    
                    # 检查如果所有概率都为零（可能是数值问题），则使用均匀分布
                    if torch.all(valid_probs == 0).item():
                        # 创建未访问城市的均匀分布
                        valid_probs = torch.ones_like(valid_probs)
                        valid_probs.masked_fill_(self.mask, 0)  # 将已访问城市的概率设为0
                        # 重新归一化概率
                        sum_probs = valid_probs.sum(dim=-1, keepdim=True)
                        valid_probs = valid_probs / (sum_probs + 1e-10)
                        
                    # 选择概率最高的动作
                    action = torch.argmax(valid_probs, dim=-1)
                else:
                    # 没有掩码信息，直接选择概率最高的动作
                    action = torch.argmax(action_probs, dim=-1)
        
        return action, objectives
        
    def _post_process_solution(self, solution: torch.Tensor) -> torch.Tensor:
        """
        对解决方案进行后处理，确保满足约束条件
        
        Args:
            solution: 原始预测解决方案
            
        Returns:
            处理后的可行解决方案
        """
        # 对于TSP问题，确保是有效的排列
        if self.problem_type == 'tsp':
            # 使用softmax将输出转换为概率分布
            probs = torch.softmax(solution, dim=-1)
            # 使用贪婪策略选择城市
            _, indices = torch.sort(probs, descending=True)
            return indices
        
        # 对于背包问题，确保满足容量约束
        elif self.problem_type == 'knapsack':
            # 使用sigmoid将输出转换为0-1之间的值
            binary_sol = torch.sigmoid(solution)
            # 根据阈值确定是否选择物品
            return (binary_sol > 0.5).float()
            
        return solution
        
    def calculate_objectives(self, solution: torch.Tensor, problem_data: Dict) -> torch.Tensor:
        """
        计算解决方案的实际目标函数值
        
        Args:
            solution: 预测的解决方案
            problem_data: 问题实例数据
            
        Returns:
            目标函数值 [n_objectives]
        """
        if self.problem_type == 'tsp':
            return self._calculate_tsp_objectives(solution, problem_data)
        elif self.problem_type == 'knapsack':
            return self._calculate_knapsack_objectives(solution, problem_data)
        
    def _calculate_tsp_objectives(self, solution: torch.Tensor, problem_data: Dict) -> torch.Tensor:
        """计算TSP问题的目标函数值"""
        total_distance = 0.0
        time_penalty = 0.0
        
        # 计算总距离和时间窗口惩罚
        cities = problem_data['cities']
        route = solution.cpu().numpy()
        
        for i in range(len(route)):
            curr_city = cities[route[i]]
            next_city = cities[route[(i + 1) % len(route)]]
            
            # 计算距离
            dist = np.sqrt((curr_city['x'] - next_city['x'])**2 + 
                          (curr_city['y'] - next_city['y'])**2)
            total_distance += dist
            
            # 计算时间窗口惩罚
            if 'time_window' in curr_city:
                arrival_time = curr_city.get('arrival_time', 0)
                window_start, window_end = curr_city['time_window']
                if arrival_time < window_start:
                    time_penalty += window_start - arrival_time
                elif arrival_time > window_end:
                    time_penalty += arrival_time - window_end
                    
        return torch.tensor([total_distance, time_penalty])
        
    def _calculate_knapsack_objectives(self, solution: torch.Tensor, problem_data: Dict) -> torch.Tensor:
        """计算多目标背包问题的目标函数值"""
        # 处理维度，确保solution是一维的
        if len(solution.shape) > 1:
            solution = solution.squeeze()
            
        if 'categories' in problem_data:
            # 多类别背包问题
            items = problem_data['items']  # [num_categories, items_per_category, n_properties]
            selected = solution.cpu().numpy()  # [num_categories]
            
            # 获取选中的物品
            selected_items = items[np.arange(len(selected)), selected]
            
            # 计算目标函数值
            objectives = np.zeros(self.n_objectives)
            for i in range(min(self.n_objectives, selected_items.shape[1])):
                objectives[i] = np.sum(selected_items[:, i])
                
        else:
            # 标准背包问题（物品无类别）
            properties = []
            
            # 获取物品属性
            # 1. 先检查objective_names是否在配置中
            objective_names = getattr(self, 'objective_names', None)
            
            # 优先使用命名属性
            if objective_names and len(objective_names) >= self.n_objectives:
                for i in range(self.n_objectives):
                    name = objective_names[i]
                    if name in problem_data:
                        properties.append(problem_data[name])
                    else:
                        # 2. 然后尝试使用传统命名
                        if i == 0 and 'values' in problem_data:
                            properties.append(problem_data['values'])
                        elif i == 1 and 'weights' in problem_data:
                            properties.append(problem_data['weights'])
                        elif i == 2 and 'volumes' in problem_data:
                            properties.append(problem_data['volumes'])
                        elif i == 3 and 'costs' in problem_data:
                            properties.append(problem_data['costs'])
                        else:
                            # 3. 然后尝试使用通用命名
                            prop_name = f'property_{i+1}'
                            if prop_name in problem_data:
                                properties.append(problem_data[prop_name])
                            else:
                                # 最后，使用随机值
                                properties.append(torch.rand_like(solution))
            else:
                # 如果没有objective_names，使用通用属性名
                for i in range(self.n_objectives):
                    prop_name = f'property_{i+1}'
                    if prop_name in problem_data:
                        properties.append(problem_data[prop_name])
                    elif i == 0 and 'values' in problem_data:
                        properties.append(problem_data['values'])
                    elif i == 1 and 'weights' in problem_data:
                        properties.append(problem_data['weights'])
                    elif i == 2 and 'volumes' in problem_data:
                        properties.append(problem_data['volumes'])
                    elif i == 3 and 'costs' in problem_data:
                        properties.append(problem_data['costs'])
                    else:
                        # 如果缺少属性，使用随机值
                        properties.append(torch.rand_like(solution))
            
            # 计算目标函数值
            objectives = np.zeros(self.n_objectives)
            for i, prop in enumerate(properties):
                if isinstance(prop, torch.Tensor):
                    prop = prop.cpu().numpy()
                objectives[i] = np.sum(solution.cpu().numpy() * prop)
                
        return torch.tensor(objectives)

    def set_mask(self, mask):
        """设置动作mask（用于TSP）"""
        self.mask = mask
