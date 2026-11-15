import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict
import json
from itertools import combinations
import os
import pandas as pd
from pandas.plotting import parallel_coordinates
import logging
import scipy.stats as stats
import tempfile
import requests
import shutil
import subprocess

# 添加字体下载和安装功能
def download_and_install_fonts():
    """下载并安装中文字体"""
    logging.info("检查中文字体...")
    
    # 检查matplotlib字体目录
    import matplotlib as mpl
    from matplotlib.font_manager import fontManager, FontManager
    
    # 获取系统字体列表
    fm = FontManager()
    font_names = [f.name for f in fm.ttflist]
    
    # 检查是否有中文字体
    chinese_fonts = [f for f in font_names if any(name in f for name in 
                 ['Microsoft YaHei', 'SimHei', 'SimSun', 'NSimSun', 'KaiTi', 'FangSong',
                  'Noto Sans CJK', 'Source Han Sans', 'Hiragino', 'STHeiti', 'STSong'])]
    
    if not chinese_fonts:
        logging.info("未找到中文字体，正在下载...")
        
        # 创建临时目录
        temp_dir = tempfile.mkdtemp()
        font_dir = os.path.join(os.path.expanduser("~"), ".fonts")
        os.makedirs(font_dir, exist_ok=True)
        
        try:
            # 从GitHub下载Google Noto Sans CJK字体
            url = "https://github.com/googlefonts/noto-cjk/raw/main/Sans/OTF/SimplifiedChinese/NotoSansCJKsc-Regular.otf"
            font_path = os.path.join(temp_dir, "NotoSansCJKsc-Regular.otf")
            
            logging.info(f"下载字体: {url}")
            response = requests.get(url, stream=True)
            if response.status_code == 200:
                with open(font_path, 'wb') as f:
                    response.raw.decode_content = True
                    shutil.copyfileobj(response.raw, f)
                
                # 复制到用户字体目录
                user_font_path = os.path.join(font_dir, "NotoSansCJKsc-Regular.otf")
                shutil.copy(font_path, user_font_path)
                logging.info(f"字体已复制到: {user_font_path}")
                
                # 刷新字体缓存
                try:
                    # Linux系统刷新字体缓存
                    if os.name != 'nt':  # 非Windows系统
                        subprocess.run(['fc-cache', '-f', '-v'], check=True)
                        logging.info("已刷新系统字体缓存")
                except Exception as e:
                    logging.warning(f"刷新字体缓存失败: {str(e)}")
                
                # 添加字体到matplotlib
                try:
                    fontManager.addfont(user_font_path)
                    logging.info("字体已添加到matplotlib")
                except Exception as e:
                    logging.warning(f"添加字体到matplotlib失败: {str(e)}")
                
                # 如果是Windows，还需要安装字体
                if os.name == 'nt':
                    logging.info("检测到Windows系统，请手动安装字体")
                    logging.info(f"字体文件位置: {user_font_path}")
                    
                # 重新加载字体管理器
                from matplotlib.font_manager import _rebuild
                _rebuild()
                logging.info("已重新加载字体管理器")
            else:
                logging.error(f"下载字体失败，HTTP状态码: {response.status_code}")
        except Exception as e:
            logging.error(f"字体安装过程中出错: {str(e)}")
        finally:
            shutil.rmtree(temp_dir)
    else:
        logging.info(f"已找到中文字体: {chinese_fonts}")
    
    return chinese_fonts

class ProblemVisualizer:
    def __init__(self, problem_type=None, use_english_labels=False):
        """初始化可视化器
        
        Args:
            problem_type: 问题类型，如'knapsack'或'tsp'
            use_english_labels: 是否使用英文标签（解决中文字体显示问题）
        """
        # 确保可视化目录存在
        self.vis_dir = 'solutions'
        os.makedirs(self.vis_dir, exist_ok=True)
        os.makedirs('solutions', exist_ok=True)
        
        # 保存问题类型
        self.problem_type = problem_type
        # 设置标签语言
        self.use_english_labels = use_english_labels
        
        # 下载并安装中文字体
        self.chinese_fonts = [] if use_english_labels else download_and_install_fonts()
        
        # 设置matplotlib支持中文显示
        self._setup_matplotlib_fonts()
        
        # 中文目标名称到英文的映射
        self.cn_to_en_objectives = {
            '价值': 'Value',
            '重量': 'Weight',
            '体积': 'Volume',
            '成本': 'Cost',
            '时间': 'Time',
            '目标1': 'Objective1',
            '目标2': 'Objective2',
            '目标3': 'Objective3',
            '目标4': 'Objective4',
            '目标5': 'Objective5',
            '目标6': 'Objective6',
            '目标7': 'Objective7',
            '目标8': 'Objective8',
            '目标9': 'Objective9',
            '目标10': 'Objective10'
        }
        
    def _setup_matplotlib_fonts(self):
        """设置matplotlib字体配置"""
        # 优先使用英文标签解决中文显示问题
        if not self.chinese_fonts or self.use_english_labels:
            self.use_english_labels = True
            logging.info("由于未找到中文字体或配置指定，将使用英文标签")
            return
            
        # 设置matplotlib支持中文显示 - 基于已安装的字体
        plt.rcParams['font.sans-serif'] = [
            'Noto Sans CJK SC', 'Noto Sans CJK TC', 'Source Han Sans CN', 'Source Han Sans TW',
            'Microsoft YaHei', 'SimHei', 'SimSun', 'NSimSun', 'FangSong', 'KaiTi', 
            'SimSun-ExtB', 'Microsoft YaHei UI', 'Microsoft JhengHei',
            'Hiragino Sans GB', 'STHeiti', 'STSong', 'STFangsong', 'STKaiti',
            'Arial Unicode MS', 'DejaVu Sans', 'sans-serif'
        ]
        
        # 如果找到了中文字体，把它们放在首位
        if self.chinese_fonts:
            plt.rcParams['font.sans-serif'] = self.chinese_fonts + plt.rcParams['font.sans-serif']
        
        plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
        plt.rcParams['font.family'] = 'sans-serif'
        
        # 设置字体回退机制
        plt.rcParams['svg.fonttype'] = 'none'
        
        # 尝试提前加载字体文件，减少字形丢失警告
        try:
            import matplotlib.font_manager as fm
            
            # 针对不同操作系统设置字体
            import platform
            system = platform.system()
            
            if system == 'Windows':
                # Windows系统上的默认中文字体
                windows_fonts = ['SimSun', 'Microsoft YaHei', 'SimHei', 'KaiTi']
                for font in windows_fonts:
                    try:
                        fm.findfont(font, fallback_to_default=False)
                        logging.info(f"成功找到并加载字体: {font}")
                        # 将此字体设为首选
                        plt.rcParams['font.sans-serif'].insert(0, font)
                        break
                    except Exception:
                        continue
            elif system == 'Darwin':  # macOS
                # macOS系统上的默认中文字体
                macos_fonts = ['PingFang SC', 'STHeiti', 'Heiti SC']
                for font in macos_fonts:
                    try:
                        fm.findfont(font, fallback_to_default=False)
                        plt.rcParams['font.sans-serif'].insert(0, font)
                        break
                    except Exception:
                        continue
            elif system == 'Linux':
                # Linux系统上的默认中文字体
                linux_fonts = ['Noto Sans CJK SC', 'Source Han Sans CN', 'WenQuanYi Micro Hei']
                for font in linux_fonts:
                    try:
                        fm.findfont(font, fallback_to_default=False)
                        plt.rcParams['font.sans-serif'].insert(0, font)
                        break
                    except Exception:
                        continue
        except Exception as e:
            logging.warning(f"加载字体过程中出错: {e}")
            # 如果字体加载失败，强制使用英文标签
            self.use_english_labels = True
        
    def _translate_objective_names(self, objective_names):
        """将中文目标名称转换为英文（如果启用了英文标签）"""
        if not self.use_english_labels:
            return objective_names
            
        translated_names = []
        for name in objective_names:
            # 如果是中文名称且在映射表中，使用英文替代
            if name in self.cn_to_en_objectives:
                translated_names.append(self.cn_to_en_objectives[name])
            else:
                # 如果不在映射表中但包含中文字符，尝试通用替换
                has_chinese = False
                for char in name:
                    if '\u4e00' <= char <= '\u9fff':
                        has_chinese = True
                        break
                
                if has_chinese:
                    # 对于包含中文的名称，使用"Objective"前缀加数字
                    translated_names.append(f"Objective_{len(translated_names)+1}")
                else:
                    # 非中文名称保持不变
                    translated_names.append(name)
                    
        return translated_names
        
    def _compute_pareto_ranks(self, objectives: np.ndarray) -> np.ndarray:
        """计算解的Pareto等级"""
        n_solutions = objectives.shape[0]
        dominated_count = np.zeros(n_solutions)
        
        # 计算每个解被支配的次数
        for i in range(n_solutions):
            for j in range(n_solutions):
                if i != j:
                    # 检查j是否支配i
                    if np.all(objectives[j] >= objectives[i]) and np.any(objectives[j] > objectives[i]):
                        dominated_count[i] += 1
        
        return dominated_count
        
    def visualize_dataset_samples(self, data_dir, problem_type, num_samples=5, split='train'):
        """可视化数据集样本"""
        target_dir = os.path.join('data', problem_type, split)
        if not os.path.exists(target_dir):
            print(f"目录不存在: {target_dir}")
            return
        
        files = [f for f in os.listdir(target_dir) if f.endswith('.json')]
        if not files:
            print(f"没有找到JSON文件在: {target_dir}")
            return
        
        np.random.shuffle(files)
        
        for i, file in enumerate(files[:num_samples]):
            try:
                with open(os.path.join(target_dir, file), 'r') as f:
                    instance = json.load(f)
                    
                if problem_type == 'knapsack':
                    self.visualize_knapsack_instance(instance, 
                        save_path=os.path.join(self.vis_dir, f'knapsack_sample_{split}_{i}.png'))
                else:  # tsp
                    self.visualize_tsp_instance(instance, 
                        save_path=os.path.join(self.vis_dir, f'tsp_sample_{split}_{i}.png'))
            except Exception as e:
                print(f"处理文件时出错 {file}: {str(e)}")
    
    def visualize_knapsack_instance(self, instance, save_path=None):
        """可视化多目标背包问题实例
        
        Args:
            instance: 包含items和initial_solution的字典
            save_path: 保存图像的路径
        """
        # 创建3D图
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        items = np.array(instance['items'])  # shape: (num_categories, items_per_category, 3)
        initial_solution = np.array(instance['initial_solution'])
        
        # 获取所有物品的属性
        values = items[:, :, 0].flatten()  # 价值
        volumes = items[:, :, 1].flatten()  # 体积
        masses = items[:, :, 2].flatten()  # 质量
        
        # 绘制所有物品
        scatter = ax.scatter(values, volumes, masses, 
                           c='blue', alpha=0.3, label='All Items')
        
        # 获取选中物品的属性
        selected_items = items[np.arange(len(initial_solution)), initial_solution]
        selected_values = selected_items[:, 0]
        selected_volumes = selected_items[:, 1]
        selected_masses = selected_items[:, 2]
        
        # 绘制选中的物品
        ax.scatter(selected_values, selected_volumes, selected_masses,
                  c='red', s=100, label='Selected Items')
        
        # 计算目标函数值
        total_value = np.sum(selected_values)
        total_volume = np.sum(selected_volumes)
        total_mass = np.sum(selected_masses)
        
        ax.set_xlabel('Value (P)')
        ax.set_ylabel('Volume (R)')
        ax.set_zlabel('Mass (C)')
        ax.set_title(f'Multi-Objective Knapsack Instance\n' + 
                    f'Total Value: {total_value:.2f} (max)\n' +
                    f'Total Volume: {total_volume:.2f} (min)\n' +
                    f'Total Mass: {total_mass:.2f} (min)')
        
        plt.legend()
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
    
    def visualize_tsp_instance(self, instance, save_path=None):
        """可视化TSP问题实例"""
        plt.figure(figsize=(8, 8))
        coordinates = np.array(instance['coordinates'])
        tour = instance['nearest_neighbor_tour']
        
        # 绘制所有城市点
        plt.scatter(coordinates[:, 0], coordinates[:, 1], c='blue', label='Cities')
        
        # 绘制路径
        for i in range(len(tour)):
            start = coordinates[tour[i]]
            end = coordinates[tour[(i + 1) % len(tour)]]  # 使用模运算处理最后一个城市到第一个城市的连接
            plt.plot([start[0], end[0]], [start[1], end[1]], 'r-')
        
        # 添加城市编号
        for i, (x, y) in enumerate(coordinates):
            plt.annotate(f'City {i}', (x, y))
        
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title(f'TSP Instance (Tour Length: {instance["metrics"]["tour_length"]:.2f})')
        plt.legend()
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

    def visualize_problem_and_solution(self, problem: Dict, solution: List[int]):
        """可视化模型的解决方案，包括3D Pareto前沿"""
        # 创建一个图形，包含两个子图
        fig = plt.figure(figsize=(15, 6))
        
        # 1. 2D散点图（原有的可视化）
        ax1 = fig.add_subplot(121)
        items = problem['items']
        weights = [item['weight'] for item in items]
        values = [item['value'] for item in items]
        
        # 分离选中和未选中的物品
        selected_weights = [w for w, s in zip(weights, solution) if s]
        selected_values = [v for v, s in zip(values, solution) if s]
        unselected_weights = [w for w, s in zip(weights, solution) if not s]
        unselected_values = [v for v, s in zip(values, solution) if not s]
        
        ax1.scatter(unselected_weights, unselected_values, c='blue', label='Not Selected')
        ax1.scatter(selected_weights, selected_values, c='red', label='Selected')
        ax1.legend()
        ax1.set_xlabel('Weight')
        ax1.set_ylabel('Value')
        ax1.set_title('2D Solution View')
        
        # 2. 3D Pareto前沿图
        ax2 = fig.add_subplot(122, projection='3d')
        
        # 生成一些随机扰动的解，以显示Pareto前沿
        n_variations = 50
        variations = []
        objectives_list = []
        
        # 基于当前解生成变体
        for _ in range(n_variations):
            # 随机修改一些物品的选择状态
            variation = solution.copy()
            n_changes = np.random.randint(1, 5)
            change_indices = np.random.choice(len(variation), n_changes, replace=False)
            for idx in change_indices:
                variation[idx] = 1 - variation[idx]  # 翻转选择状态
                
            # 计算目标值
            total_value = sum(v for v, s in zip(values, variation) if s)
            total_weight = sum(w for w, s in zip(weights, variation) if s)
            total_volume = sum(w for w, s in zip(weights, variation) if s)  # 假设体积与重量相同，如果有实际体积数据应该使用实际数据
            
            variations.append(variation)
            objectives_list.append([total_value, total_weight, total_volume])
        
        # 将当前解添加到列表中
        current_value = sum(v for v, s in zip(values, solution) if s)
        current_weight = sum(w for w, s in zip(weights, solution) if s)
        current_volume = sum(w for w, s in zip(weights, solution) if s)
        objectives_list.append([current_value, current_weight, current_volume])
        
        # 转换为numpy数组
        objectives = np.array(objectives_list)
        
        # 计算Pareto等级
        ranks = self._compute_pareto_ranks(objectives)
        
        # 绘制3D散点图
        scatter = ax2.scatter(
            objectives[:, 0],  # 价值（最大化）
            objectives[:, 1],  # 重量（最小化）
            objectives[:, 2],  # 体积（最小化）
            c=ranks,          # 颜色表示Pareto等级
            cmap='viridis'
        )
        
        # 突出显示当前解
        ax2.scatter([current_value], [current_weight], [current_volume], 
                    c='red', s=100, marker='*', label='Current Solution')
        
        ax2.set_xlabel('Total Value (max)')
        ax2.set_ylabel('Total Weight (min)')
        ax2.set_zlabel('Total Volume (min)')
        ax2.set_title('3D Pareto Front')
        plt.colorbar(scatter, label='Pareto Rank')
        
        plt.tight_layout()
        plt.savefig('solutions/latest_solution.png')
        plt.show()
        plt.close()

    def visualize_pareto_front(self, objectives: np.ndarray, save_path: str = 'solutions/pareto_front.png'):
        """可视化Pareto前沿"""
        n_objectives = objectives.shape[1]
        
        # 如果是2D或3D问题，使用传统的散点图或3D散点图
        if n_objectives == 2:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111)
            
            # 计算Pareto等级
            pareto_ranks = self._compute_pareto_ranks(objectives)
            
            # 绘制所有解
            scatter = ax.scatter(
                objectives[:, 0],  # 第一个目标
                objectives[:, 1],  # 第二个目标
                c=pareto_ranks,    # 颜色表示Pareto等级
                cmap='viridis'
            )
            
            ax.set_xlabel('目标1')
            ax.set_ylabel('目标2')
            ax.set_title('2D Pareto前沿')
            plt.colorbar(scatter, label='Pareto等级')
            
        elif n_objectives == 3:
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            # 绘制所有解
            scatter = ax.scatter(
                objectives[:, 0],  # 第一个目标
                objectives[:, 1],  # 第二个目标
                objectives[:, 2],  # 第三个目标
                c=self._compute_pareto_ranks(objectives),  # 颜色表示Pareto等级
                cmap='viridis'
            )
            
            ax.set_xlabel('目标1')
            ax.set_ylabel('目标2')
            ax.set_zlabel('目标3')
            plt.colorbar(scatter, label='Pareto等级')
            
        else:  # 高维问题
            # 创建散点矩阵(Scatterplot Matrix)
            import matplotlib.gridspec as gridspec
            
            # 确定图表大小
            fig = plt.figure(figsize=(4 * n_objectives, 4 * n_objectives))
            gs = gridspec.GridSpec(n_objectives, n_objectives)
            
            # 计算Pareto等级
            pareto_ranks = self._compute_pareto_ranks(objectives)
            
            # 创建轴标签
            axis_labels = [f'目标{i+1}' for i in range(n_objectives)]
            
            # 为每对目标创建散点图
            for i in range(n_objectives):
                for j in range(n_objectives):
                    ax = fig.add_subplot(gs[i, j])
                    
                    if i == j:  # 对角线上放直方图
                        ax.hist(objectives[:, i], bins=20, alpha=0.7)
                        ax.set_title(axis_labels[i])
                        
                        # 只显示第一行和最后一列的坐标轴标签
                        if i != 0:  # 不是第一行
                            ax.set_xticklabels([])
                        if j != n_objectives - 1:  # 不是最后一列
                            ax.set_yticklabels([])
                    else:  # 非对角线放散点图
                        scatter = ax.scatter(
                            objectives[:, j],  # x轴
                            objectives[:, i],  # y轴
                            c=pareto_ranks,
                            cmap='viridis',
                            alpha=0.7,
                            s=30
                        )
                        
                        # 只显示第一行和最后一列的坐标轴标签
                        if i != 0:  # 不是第一行
                            ax.set_xticklabels([])
                        if j != n_objectives - 1:  # 不是最后一列
                            ax.set_yticklabels([])
                        
                        # 第一列和最后一行显示标签
                        if j == 0:
                            ax.set_ylabel(axis_labels[i])
                        if i == n_objectives - 1:
                            ax.set_xlabel(axis_labels[j])
            
            plt.tight_layout()
            plt.subplots_adjust(top=0.95)
            fig.suptitle(f'{n_objectives}维Pareto前沿 - 散点矩阵', fontsize=16)
            
            # 添加颜色条
            cbar_ax = fig.add_axes([0.92, 0.1, 0.02, 0.8])
            cbar = fig.colorbar(scatter, cax=cbar_ax)
            cbar.set_label('Pareto等级')
        
        plt.savefig(save_path)

    def visualize_rl_training(self, training_history: List[Dict], save_path: str = 'visualizations/rl_training.png'):
        """可视化强化学习训练过程
        
        Args:
            training_history: 训练历史记录，包含每个episode的奖励、价值和重量
            save_path: 保存路径
        """
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))
        
        # 提取数据
        episodes = range(len(training_history))
        rewards = [h['reward'] for h in training_history]
        values = [h['value'] for h in training_history]
        weights = [h['weight'] for h in training_history]
        capacities = [h['capacity'] for h in training_history]
        
        # 1. 奖励曲线
        ax1.plot(episodes, rewards, 'b-', label='Episode Reward')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Total Reward')
        ax1.set_title('Training Rewards over Episodes')
        ax1.grid(True)
        
        # 2. 价值曲线
        ax2.plot(episodes, values, 'g-', label='Total Value')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Total Value')
        ax2.set_title('Solution Values over Episodes')
        ax2.grid(True)
        
        # 3. 重量和容量约束
        ax3.plot(episodes, weights, 'r-', label='Total Weight')
        ax3.plot(episodes, capacities, 'k--', label='Capacity')
        ax3.fill_between(episodes, capacities, alpha=0.2)
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Weight')
        ax3.set_title('Weight vs Capacity over Episodes')
        ax3.legend()
        ax3.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    def visualize_rl_solution(self, state_history: List[Dict], solution: List[int], 
                             save_path: str = 'visualizations/rl_solution.png'):
        """可视化强化学习的解决方案
        
        Args:
            state_history: 状态历史记录
            solution: 最终解决方案
            save_path: 保存路径
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 1. 状态值变化
        steps = range(len(state_history))
        remaining_capacities = [s['remaining_capacity'] for s in state_history]
        current_values = [s['current_value'] for s in state_history]
        
        ax1.plot(steps, remaining_capacities, 'b-', label='Remaining Capacity')
        ax1.plot(steps, current_values, 'r-', label='Accumulated Value')
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Value')
        ax1.set_title('State Values during Solution Construction')
        ax1.legend()
        ax1.grid(True)
        
        # 2. 最终解决方案的价值-重量分布
        weights = [s['item_weight'] for s in state_history]
        values = [s['item_value'] for s in state_history]
        
        selected_weights = [w for w, s in zip(weights, solution) if s]
        selected_values = [v for v, s in zip(values, solution) if s]
        unselected_weights = [w for w, s in zip(weights, solution) if not s]
        unselected_values = [v for v, s in zip(values, solution) if not s]
        
        ax2.scatter(unselected_weights, unselected_values, c='blue', label='Not Selected')
        ax2.scatter(selected_weights, selected_values, c='red', label='Selected')
        ax2.set_xlabel('Weight')
        ax2.set_ylabel('Value')
        ax2.set_title('Final Solution Distribution')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    def visualize_tsp_solution(self, coordinates: np.ndarray, solution: List[int]):
        """绘制求解器返回的闭合路线。"""
        fig, ax = plt.subplots(figsize=(8, 6))
        route = coordinates[np.asarray(solution)]
        ax.plot(route[:, 0], route[:, 1], color='tab:red', alpha=0.65, label='Tour')
        ax.scatter(coordinates[:, 0], coordinates[:, 1], color='tab:blue', label='Cities')
        ax.scatter(*route[0], color='tab:red', s=120, marker='*', label='Start city')
        ax.set_title('TSP Solution Path')
        ax.set_aspect('equal')
        ax.legend()
        ax.grid(True)
        fig.tight_layout()
        fig.savefig('solutions/tsp_solution.png')
        plt.close(fig)

    def visualize_candidate_pareto(self, values, names, directions, frontier, problem_type):
        """Plot pairwise projections of the candidate frontier calculated in all dimensions."""
        translations = {
            **self.cn_to_en_objectives,
            "总距离": "Total distance", "时间窗口迟到量": "Time-window lateness",
            "路段长度变异系数": "Segment length CV", "最长路段": "Longest segment",
        }
        labels = [f"{translations.get(name, name)} ({'max' if direction == 1 else 'min'})"
                  for name, direction in zip(names, directions)]
        pairs = list(combinations(range(len(names)), 2)) or [(0, None)]
        cols = min(3, len(pairs))
        rows = (len(pairs) + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, squeeze=False, figsize=(4.2 * cols, 3.4 * rows))
        for ax, (x_index, y_index) in zip(axes.flat, pairs):
            y_values = values[:, y_index] if y_index is not None else np.zeros(len(values))
            ax.scatter(values[~frontier, x_index], y_values[~frontier], color='0.65', label='Dominated')
            ax.scatter(values[frontier, x_index], y_values[frontier], color='tab:blue', label='Nondominated')
            ax.set_xlabel(labels[x_index])
            if y_index is not None:
                ax.set_ylabel(labels[y_index])
            else:
                ax.set_yticks([])
            ax.grid(alpha=0.25)
        axes[0, 0].legend()
        title = 'TSP' if problem_type == 'tsp' else 'Knapsack'
        fig.suptitle(f'{title} candidate Pareto front (all objectives)')
        fig.tight_layout()
        fig.savefig(f'solutions/{problem_type}_pareto_front.png')
        plt.close(fig)

    def visualize_high_dimensional_solution(self, problem: Dict, solution: List[int], objectives: np.ndarray):
        """Visualize high-dimensional solution with analysis
        
        Args:
            problem: Problem instance data
            solution: Current solution
            objectives: Objective function values [n_solutions, n_objectives]
        """
        try:
            # 导入所需库
            import os
            import pandas as pd
            from pandas.plotting import parallel_coordinates
            import matplotlib.pyplot as plt
            import numpy as np
            import logging
            
            # 重新确保设置字体配置，确保中文显示正常
            self._setup_matplotlib_fonts()
            
            # 如果遇到字体问题，尝试切换到英文标签
            if not self.chinese_fonts and not self.use_english_labels:
                logging.warning("未找到中文字体，自动切换到英文标签模式")
                self.use_english_labels = True
            
            # 强制使用英文标签避免字体问题（可在配置中设置为False重新启用中文）
            if os.name == 'nt':  # Windows系统
                logging.info("检测到Windows系统，使用英文标签以避免字体问题")
                self.use_english_labels = True  # 在Windows上默认使用英文标签
            
            n_objectives = objectives.shape[1]
            logging.info(f"可视化包含{n_objectives}个目标的高维解决方案")
            
            if n_objectives <= 3:
                # Use existing 3D visualization
                self.visualize_problem_and_solution(problem, solution)
                logging.info("使用3D可视化展示3个或更少目标的结果")
                return
            
            # 获取目标名称和类型
            logging.info(f"从problem中获取目标信息，problem字典内包含的键: {list(problem.keys())}")
            objective_names = problem.get('objective_names', [f'Objective {i+1}' for i in range(n_objectives)])
            objective_types = problem.get('objective_types', [1] + [-1] * (n_objectives - 1))
            
            logging.info(f"初始objective_names: {objective_names}")
            logging.info(f"初始objective_types: {objective_types}")
            
            # 确保objective_types不为None
            if objective_types is None:
                # 默认第一个目标最大化，其他目标最小化
                objective_types = [1] + [-1] * (n_objectives - 1)
                logging.warning(f"objective_types为None，使用默认值: {objective_types}")
            
            # 确保objective_names的长度与n_objectives匹配
            if len(objective_names) < n_objectives:
                # 如果名称数量不足，添加默认名称
                for i in range(len(objective_names), n_objectives):
                    objective_names.append(f'Objective {i+1}')
                logging.warning(f"objective_names长度不足，已补充至: {objective_names}")
            elif len(objective_names) > n_objectives:
                # 如果名称数量过多，截断
                objective_names = objective_names[:n_objectives]
                logging.warning(f"objective_names长度过多，已截断至: {objective_names}")
            
            # 检查目标类型数量是否匹配
            if len(objective_types) < n_objectives:
                # 如果类型数量不足，添加默认值（最小化）
                objective_types.extend([-1] * (n_objectives - len(objective_types)))
                logging.warning(f"objective_types长度不足，已补充至: {objective_types}")
                
            # 如果类型数量过多，截断
            if len(objective_types) > n_objectives:
                objective_types = objective_types[:n_objectives]
                logging.warning(f"objective_types长度过多，已截断至: {objective_types}")
                
            logging.info(f"处理后的objective_names: {objective_names}")
            logging.info(f"处理后的objective_types: {objective_types}")
            
            # 如果启用了英文标签，翻译目标名称
            if self.use_english_labels:
                objective_names = self._translate_objective_names(objective_names)
            
            # 确保目标名称和类型的长度与目标数量匹配
            if len(objective_names) < n_objectives:
                logging.warning(f"目标名称数量({len(objective_names)})少于目标数量({n_objectives})，使用默认名称补充")
                for i in range(len(objective_names), n_objectives):
                    if self.use_english_labels:
                        objective_names.append(f'Objective_{i+1}')
                    else:
                        objective_names.append(f'目标{i+1}')
            
            if len(objective_types) < n_objectives:
                logging.warning(f"目标类型数量({len(objective_types)})少于目标数量({n_objectives})，假设剩余目标为最小化")
                for i in range(len(objective_types), n_objectives):
                    objective_types.append(-1)  # 默认为最小化目标
            
            # 截断超出的部分
            objective_names = objective_names[:n_objectives]
            objective_types = objective_types[:n_objectives]
            
            logging.info(f"目标名称: {objective_names}")
            logging.info(f"目标类型: {objective_types}")
            
            # Add maximize/minimize labels with language option
            if self.use_english_labels:
                objective_labels = [f"{name}\n({'Maximize' if typ == 1 else 'Minimize'})" 
                                for name, typ in zip(objective_names, objective_types)]
            else:
                # 使用中文标签
                objective_labels = [f"{name}\n({'最大化' if typ == 1 else '最小化'})" 
                                for name, typ in zip(objective_names, objective_types)]
            
            # Generate variants by perturbing the objectives
            n_variants = 10
            variants = []
            
            # Add current solution
            if len(objectives) > 0:
                variants.append(objectives[-1])  # Current solution
            
            # Generate variants with random perturbations
            base_solution = objectives[-1].copy()  # Use current solution as base
            for _ in range(n_variants - 1):
                variant = base_solution.copy()
                for j in range(n_objectives):
                    perturbation = np.random.uniform(-0.3, 0.3)
                    variant[j] *= (1 + perturbation)
                variants.append(variant)
            else:
                # 如果没有提供objectives，创建一些随机数据用于示例
                logging.warning("没有提供objectives数据，使用随机数据进行可视化")
                # 创建一个基础变体
                base_variant = np.random.rand(n_objectives)
                variants.append(base_variant)
                
                # 生成其他变体
                for _ in range(n_variants - 1):
                    variant = base_variant.copy()
                    for j in range(n_objectives):
                        perturbation = np.random.uniform(-0.3, 0.3)
                        variant[j] *= (1 + perturbation)
                    variants.append(variant)
            
            variants = np.array(variants)
            logging.info(f"成功生成{len(variants)}个变体，形状：{variants.shape}")
            
            # Normalize objective values for comparison
            normalized_objectives = np.zeros_like(variants)
            for i in range(n_objectives):
                min_val, max_val = variants[:, i].min(), variants[:, i].max()
                if min_val == max_val:
                    normalized_objectives[:, i] = 0.5
                else:
                    normalized_objectives[:, i] = (variants[:, i] - min_val) / (max_val - min_val)
                    if i < len(objective_types) and objective_types[i] == -1:
                        normalized_objectives[:, i] = 1 - normalized_objectives[:, i]
            
            solution_types = ['Current' if i == 0 else f'Variant {i}' for i in range(len(variants))]
            df = pd.DataFrame(normalized_objectives, columns=objective_labels)
            df['Solution'] = solution_types
            
            # 1. 平行坐标图分析
            logging.info("\n正在生成平行坐标图分析...")
            fig_parallel = plt.figure(figsize=(12, 8))
            ax_parallel = fig_parallel.add_subplot(111)
            parallel_coordinates(df, 'Solution', ax=ax_parallel, colormap=plt.cm.tab10)
            
            if self.use_english_labels:
                parallel_title = f'{n_objectives}-Dimensional Multi-Objective Solution - Parallel Coordinates'
            else:
                parallel_title = f'{n_objectives}-维多目标优化解决方案 - 平行坐标图'
                
            ax_parallel.set_title(parallel_title, fontsize=16)
            ax_parallel.grid(True)
            ax_parallel.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
            plt.tight_layout()
            
            # 确保solutions目录存在
            try:
                # 获取当前工作目录
                current_dir = os.getcwd()
                solutions_dir = os.path.join(current_dir, 'solutions')
                logging.info(f"当前工作目录: {current_dir}")
                logging.info(f"尝试创建或确认目录: {solutions_dir}")
                
                os.makedirs(solutions_dir, exist_ok=True)
                
                # 保存为UI期望的文件名，使用绝对路径
                parallel_path = os.path.join(solutions_dir, 'parallel_coordinates.png')
                logging.info(f"尝试保存平行坐标图到: {parallel_path}")
                plt.savefig(parallel_path, bbox_inches='tight')
                
                if os.path.exists(parallel_path):
                    file_size = os.path.getsize(parallel_path)
                    logging.info(f"平行坐标图已成功保存，文件大小: {file_size} 字节")
                else:
                    logging.error(f"保存失败！文件不存在: {parallel_path}")
            except Exception as e:
                logging.error(f"保存平行坐标图时出错: {str(e)}")
                import traceback
                logging.error(traceback.format_exc())
            
            # 分析平行坐标图结果
            correlations = {}
            tradeoffs = []
            performances = {}
            
            for i in range(n_objectives):
                for j in range(i+1, n_objectives):
                    corr = np.corrcoef(normalized_objectives[:, i], normalized_objectives[:, j])[0, 1]
                    corr_key = f"{objective_names[i]}-{objective_names[j]}"
                    correlations[corr_key] = round(corr, 2)
                    if abs(corr) > 0.7:
                        if self.use_english_labels:
                            relation = "positive" if corr > 0 else "negative"
                            tradeoffs.append(f"Strong {relation} correlation between {objective_names[i]} and {objective_names[j]} (r={round(corr, 2)})")
                        else:
                            relation = "正相关" if corr > 0 else "负相关"
                            tradeoffs.append(f"{objective_names[i]}和{objective_names[j]}之间存在强{relation}(r={round(corr, 2)})")
            
            for i, obj_name in enumerate(objective_names):
                if i < normalized_objectives.shape[1]:
                    current_val = normalized_objectives[0, i]
                    if self.use_english_labels:
                        performance = "excellent" if current_val > 0.8 else "good" if current_val > 0.6 else "average" if current_val > 0.4 else "poor"
                    else:
                        performance = "优秀" if current_val > 0.8 else "良好" if current_val > 0.6 else "一般" if current_val > 0.4 else "较差"
                    performances[obj_name] = performance
            
            if self.use_english_labels:
                parallel_analysis = {
                    "Objective Correlations": correlations,
                    "Identified Tradeoffs": tradeoffs,
                    "Current Solution Performance": performances
                }
            else:
                parallel_analysis = {
                    "目标相关性": correlations,
                    "发现的权衡关系": tradeoffs,
                    "当前解表现": performances
                }
            
            logging.info("平行坐标图分析结果:" if not self.use_english_labels else "Parallel Coordinates Analysis Results:")
            for key, value in parallel_analysis.items():
                logging.info(f"- {key}: {value}")
            
            plt.close(fig_parallel)
            
            # 2. 雷达图分析
            logging.info("\n正在生成雷达图分析...")
            fig_radar = plt.figure(figsize=(10, 10))
            ax_radar = fig_radar.add_subplot(111, projection='polar')
            
            current_solution = normalized_objectives[0]
            
            # 确保权重数组与目标数量匹配
            weights = np.ones(n_objectives)  # 初始化全1数组
            for i in range(n_objectives):
                if i < len(objective_types):
                    weights[i] = 1 if objective_types[i] == 1 else -1
                else:
                    weights[i] = 1  # 默认最大化
            
            # 计算变体分数
            if len(normalized_objectives) > 1:
                variant_scores = np.sum(normalized_objectives[1:] * weights, axis=1)
                # 修复索引检查，确保best_variant_idx是整数而不是数组
                best_variant_idx = np.argmax(variant_scores) + 1 if len(variant_scores) > 0 else 0
                
                # 获取最佳变体，如果没有变体就用当前解
                if best_variant_idx > 0 and best_variant_idx < len(normalized_objectives):
                    best_variant = normalized_objectives[best_variant_idx]
                else:
                    best_variant = current_solution.copy()
            else:
                # 如果只有当前解，创建一个略微修改的版本作为"最佳"变体
                best_variant = current_solution.copy() * 1.1  # 稍微提高一点
            
            # 确保角度数组与目标数量匹配
            angles = np.linspace(0, 2*np.pi, n_objectives, endpoint=False).tolist()
            angles += angles[:1]  # 闭合图形
            
            # 处理数据数组
            if len(current_solution) < n_objectives:
                logging.warning(f"当前解维度({len(current_solution)})小于目标数量({n_objectives})，填充零值")
                padding = np.zeros(n_objectives - len(current_solution))
                current_solution = np.concatenate([current_solution, padding])
            
            if len(best_variant) < n_objectives:
                logging.warning(f"最佳变体维度({len(best_variant)})小于目标数量({n_objectives})，填充零值")
                padding = np.zeros(n_objectives - len(best_variant))
                best_variant = np.concatenate([best_variant, padding])
            
            # 确保数据数组与角度数组长度匹配
            current_data = current_solution[:n_objectives].tolist()
            current_data += current_data[:1]  # 闭合图形
            
            # 确保最佳变体数据数组与角度数组长度匹配
            best_data = best_variant[:n_objectives].tolist()
            best_data += best_data[:1]  # 闭合图形
            
            # 绘制雷达图
            ax_radar.plot(angles, current_data, 'r-', linewidth=2, label='当前解' if not self.use_english_labels else 'Current Solution')
            ax_radar.plot(angles, best_data, 'b--', linewidth=2, label='最佳变体' if not self.use_english_labels else 'Best Variant')
            ax_radar.fill(angles, current_data, 'r', alpha=0.1)
            ax_radar.fill(angles, best_data, 'b', alpha=0.1)
            
            # 设置刻度和标签
            ax_radar.set_xticks(angles[:-1])
            # 确保标签数量与角度数量匹配
            ax_radar.set_xticklabels([objective_labels[i] if i < len(objective_labels) else 
                                   (f"目标{i+1}" if not self.use_english_labels else f"Objective {i+1}") 
                                   for i in range(n_objectives)])
            
            if self.use_english_labels:
                radar_title = f'{n_objectives}-Dimensional Solution Radar Chart'
            else:
                radar_title = f'{n_objectives}-维解决方案雷达图'
                
            ax_radar.set_title(radar_title, fontsize=16)
            ax_radar.grid(True)
            ax_radar.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
            plt.tight_layout()
            
            # 确保solutions目录存在
            try:
                # 使用之前定义的绝对路径
                radar_path = os.path.join(solutions_dir, 'radar_chart.png')
                logging.info(f"尝试保存雷达图到: {radar_path}")
                plt.savefig(radar_path, bbox_inches='tight')
                
                if os.path.exists(radar_path):
                    file_size = os.path.getsize(radar_path)
                    logging.info(f"雷达图已成功保存，文件大小: {file_size} 字节")
                else:
                    logging.error(f"保存失败！文件不存在: {radar_path}")
            except Exception as e:
                logging.error(f"保存雷达图时出错: {str(e)}")
                import traceback
                logging.error(traceback.format_exc())
            
            plt.close(fig_radar)
            
            # 3. 热力图分析（如果目标数量足够多）
            if n_objectives >= 4:
                logging.info("\n正在生成热力图分析...")
                fig_heatmap = plt.figure(figsize=(10, 8))
                
                # 创建相关系数矩阵
                corr_matrix = np.corrcoef(normalized_objectives.T)
                
                # 创建热图
                ax_heatmap = fig_heatmap.add_subplot(111)
                im = ax_heatmap.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
                
                # 添加颜色条
                cbar = fig_heatmap.colorbar(im)
                cbar.set_label('Correlation Coefficient')
                
                # 设置刻度和标签
                ax_heatmap.set_xticks(np.arange(n_objectives))
                ax_heatmap.set_yticks(np.arange(n_objectives))
                short_names = [name.split('\n')[0] for name in objective_labels]
                ax_heatmap.set_xticklabels(short_names, rotation=45, ha='right')
                ax_heatmap.set_yticklabels(short_names)
                
                # 显示相关系数值
                for i in range(n_objectives):
                    for j in range(n_objectives):
                        text = ax_heatmap.text(j, i, f'{corr_matrix[i, j]:.2f}',
                                          ha="center", va="center", color="black" if abs(corr_matrix[i, j]) < 0.7 else "white")
                
                if self.use_english_labels:
                    heatmap_title = 'Objective Correlation Heatmap'
                else:
                    heatmap_title = '目标相关性热力图'
                    
                ax_heatmap.set_title(heatmap_title, fontsize=16)
                plt.tight_layout()
                
                # 确保solutions目录存在
                try:
                    # 使用之前定义的绝对路径
                    heatmap_path = os.path.join(solutions_dir, 'heatmap.png')
                    logging.info(f"尝试保存热力图到: {heatmap_path}")
                    plt.savefig(heatmap_path, bbox_inches='tight')
                    
                    if os.path.exists(heatmap_path):
                        file_size = os.path.getsize(heatmap_path)
                        logging.info(f"热力图已成功保存，文件大小: {file_size} 字节")
                    else:
                        logging.error(f"保存失败！文件不存在: {heatmap_path}")
                except Exception as e:
                    logging.error(f"保存热力图时出错: {str(e)}")
                    import traceback
                    logging.error(traceback.format_exc())
                
                plt.close(fig_heatmap)
            
            # 分析和记录结果
            logging.info("\n高维解决方案分析完成")
            logging.info(f"已生成平行坐标图、雷达图和热力图(如果目标数量≥4)")
            logging.info("所有图表已保存到solutions/目录")
            
            # 创建一个汇总图像，便于UI显示
            try:
                # 先创建散点矩阵图，UI中期望这个图表
                if n_objectives >= 3:
                    logging.info("\n正在生成散点矩阵图...")
                    # 创建DataFrame用于散点矩阵图
                    scatter_data = {}
                    for i, name in enumerate(objective_names[:n_objectives]):
                        if i < variants.shape[1]:
                            scatter_data[name] = variants[:, i]
                    
                    if scatter_data:
                        scatter_df = pd.DataFrame(scatter_data)
                        # 创建散点矩阵图
                        scatter_fig = plt.figure(figsize=(12, 10))
                        ax = pd.plotting.scatter_matrix(scatter_df, alpha=0.8, 
                                                      figsize=(12, 10), diagonal='kde')
                        
                        # 调整散点矩阵图的样式
                        for i in range(len(scatter_df.columns)):
                            for j in range(len(scatter_df.columns)):
                                if i != j:
                                    # 为非对角线图添加趋势线
                                    ax[i, j].grid(True)
                                    
                                    # 增加点的大小和颜色区分
                                    ax[i, j].scatter(scatter_df.iloc[:, j], scatter_df.iloc[:, i], 
                                                   c='blue', alpha=0.6, s=50)
                                    
                                    # 标记当前解
                                    if len(variants) > 0:
                                        ax[i, j].scatter([variants[0, j]], [variants[0, i]], 
                                                      c='red', s=100, marker='*', 
                                                      label='Current Solution')
                        
                        plt.tight_layout()
                        plt.suptitle('Objective Values Scatter Matrix' if self.use_english_labels else '目标值散点矩阵图', fontsize=16, y=1.02)
                        
                        # 保存散点矩阵图
                        try:
                            scatter_path = os.path.join(solutions_dir, 'scatter_matrix.png')
                            logging.info(f"尝试保存散点矩阵图到: {scatter_path}")
                            plt.savefig(scatter_path, bbox_inches='tight')
                            
                            if os.path.exists(scatter_path):
                                file_size = os.path.getsize(scatter_path)
                                logging.info(f"散点矩阵图已成功保存，文件大小: {file_size} 字节")
                            else:
                                logging.error(f"保存失败！文件不存在: {scatter_path}")
                        except Exception as e:
                            logging.error(f"保存散点矩阵图时出错: {str(e)}")
                            import traceback
                            logging.error(traceback.format_exc())
                            
                        plt.close(scatter_fig)
                
                # 创建一个空白的汇总图像
                fig_summary = plt.figure(figsize=(12, 10))
                plt.axis('off')  # 关闭坐标轴
                
                # 添加标题
                title = f'{n_objectives}-Dimensional Multi-Objective Solution'
                if not self.use_english_labels:
                    title = f'{n_objectives}-维多目标优化解决方案'
                plt.title(title, fontsize=16)
                
                # 添加描述文本
                plt.text(0.5, 0.5, 
                        f"Visualization images available in solutions directory:\n" +
                        f"- Parallel Coordinates: solutions/parallel_coordinates.png\n" +
                        f"- Radar Chart: solutions/radar_chart.png" +
                        (f"\n- Heatmap: solutions/heatmap.png" if n_objectives >= 4 else "") +
                        f"\n- Scatter Matrix: solutions/scatter_matrix.png",
                        ha='center', va='center', fontsize=14)
                
                # 保存汇总图像
                try:
                    summary_path = os.path.join(solutions_dir, 'high_dimensional_solution.png')
                    logging.info(f"尝试保存汇总图到: {summary_path}")
                    plt.savefig(summary_path, bbox_inches='tight')
                    
                    if os.path.exists(summary_path):
                        file_size = os.path.getsize(summary_path)
                        logging.info(f"汇总图已成功保存，文件大小: {file_size} 字节")
                    else:
                        logging.error(f"保存失败！文件不存在: {summary_path}")
                except Exception as e:
                    logging.error(f"保存汇总图时出错: {str(e)}")
                    import traceback
                    logging.error(traceback.format_exc())
                
                plt.close(fig_summary)
            except Exception as e:
                logging.error(f"创建汇总图时出错: {e}")
            
            # 返回分析结果
            if self.use_english_labels:
                return {
                    "correlations": correlations,
                    "tradeoffs": tradeoffs,
                    "performances": performances
                }
            else:
                return {
                    "相关性": correlations,
                    "权衡关系": tradeoffs,
                    "表现": performances
                }
            
        except Exception as e:
            logging.error(f"生成可视化时出错: {str(e)}")
            import traceback
            error_details = traceback.format_exc()
            logging.error(error_details)
            
            # 输出到标准错误以便在控制台看到
            print(f"\n*** 可视化错误 ***")
            print(f"错误消息: {str(e)}")
            print(f"错误详情: \n{error_details}")
            print(f"*** 错误结束 ***\n")
            
            # 创建一个错误信息图像，确保UI能显示一些内容
            try:
                # 创建存放目录
                os.makedirs('solutions', exist_ok=True)
                
                # 创建一个简单的错误图像
                fig = plt.figure(figsize=(10, 8))
                plt.text(0.5, 0.5, f"可视化生成失败\n错误: {str(e)[:200]}...",
                         ha='center', va='center', fontsize=14, color='red')
                plt.axis('off')
                plt.savefig('solutions/visualization_error.png')
                plt.close(fig)
                logging.info("已创建错误信息图像: solutions/visualization_error.png")
            except Exception as e2:
                logging.error(f"创建错误信息图像时出错: {str(e2)}")
            
            return None

    def visualize_training_loss(self, training_losses, save_path=None, title="训练损失趋势"):
        """
        可视化训练损失趋势
        
        Args:
            training_losses: 列表或字典，包含训练过程中的损失值
            save_path: 保存图像的路径，如果为None则显示图像
            title: 图表标题
        """
        # 确保可视化目录存在
        os.makedirs('solutions', exist_ok=True)
        
        plt.figure(figsize=(10, 6))
        
        epochs = None
        loss_values = None
        
        # 处理不同类型的输入
        if isinstance(training_losses, dict):
            # 如果是字典，提取epochs和loss值
            if 'epoch' in training_losses and 'loss' in training_losses:
                epochs = training_losses['epoch']
                loss_values = training_losses['loss']
            else:
                # 如果字典中没有这些键，尝试转换为列表
                epochs = list(range(1, len(training_losses) + 1))
                loss_values = list(training_losses.values())
        elif isinstance(training_losses, list):
            # 如果是列表，直接使用
            epochs = list(range(1, len(training_losses) + 1))
            loss_values = training_losses
        else:
            raise ValueError("training_losses应为列表或字典")
        
        # 绘制曲线
        plt.plot(epochs, loss_values, 'b-', linewidth=2, marker='o', markersize=4)
        
        # 添加平滑曲线（移动平均）
        if len(loss_values) > 5:
            window_size = min(5, len(loss_values) // 5)
            if window_size > 0:
                smoothed_values = []
                for i in range(len(loss_values)):
                    start = max(0, i - window_size // 2)
                    end = min(len(loss_values), i + window_size // 2 + 1)
                    smoothed_values.append(sum(loss_values[start:end]) / (end - start))
                plt.plot(epochs, smoothed_values, 'r-', linewidth=2, alpha=0.7, label='平滑曲线')
        
        # 设置标题和标签
        if self.use_english_labels:
            plt.title("Training Loss Trend" if title == "训练损失趋势" else title, fontsize=14)
            plt.xlabel("Epochs", fontsize=12)
            plt.ylabel("Loss", fontsize=12)
        else:
            plt.title(title, fontsize=14)
            plt.xlabel("迭代次数", fontsize=12)
            plt.ylabel("损失值", fontsize=12)
        
        # 添加网格
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # 添加标注，标记最小损失点
        if loss_values:
            min_loss = min(loss_values)
            min_epoch = epochs[loss_values.index(min_loss)]
            plt.annotate(f'最小值: {min_loss:.4f}' if not self.use_english_labels else f'Min: {min_loss:.4f}', 
                        xy=(min_epoch, min_loss),
                        xytext=(min_epoch + len(epochs)*0.05, min_loss + (max(loss_values) - min(loss_values))*0.1),
                        arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                        fontsize=10)
        
        # 添加图例
        if len(loss_values) > 5:
            plt.legend(['损失值', '平滑曲线'] if not self.use_english_labels else ['Loss', 'Smoothed'])
        
        # 设置Y轴范围，给最小值留出一些空间
        if loss_values:
            y_min = max(0, min(loss_values) - (max(loss_values) - min(loss_values))*0.1)
            y_max = max(loss_values) + (max(loss_values) - min(loss_values))*0.2
            plt.ylim(y_min, y_max)
        
        # 保存或显示图像
        if save_path:
            plt.tight_layout()
            plt.savefig(save_path, dpi=150)
            plt.close()
            return save_path
        else:
            plt.tight_layout()
            plt.show()
            return None

def visualize_instance(instance, problem_type, title=''):
    vis_dir = 'solutions'
    os.makedirs(vis_dir, exist_ok=True)
    
    plt.figure(figsize=(12, 8))
    
    if problem_type == 'knapsack':
        items = instance['items']
        metrics = instance['metrics']
        greedy_solution = instance['greedy_solution']
        
        # 绘制物品分布
        weights = items[:, 1]
        values = items[:, 0]
        
        # 使用不同颜色区分贪心解选择的物品
        selected = greedy_solution == 1
        plt.scatter(weights[selected], values[selected], c='red', s=100, label='Selected (Greedy)')
        plt.scatter(weights[~selected], values[~selected], c='blue', s=100, label='Not Selected')
        
        # 添加物品编号
        for i, (v, w) in enumerate(zip(values, weights)):
            plt.annotate(f'Item {i}', (w, v), xytext=(5, 5), textcoords='offset points')
        
        # 添加指标信息
        plt.title(f'Knapsack Instance: {title}\n' + 
                 f'Selected Items: {int(np.sum(greedy_solution))}/{len(greedy_solution)} ' +
                 f'({metrics["selected_ratio"]*100:.1f}%)\n' +
                 f'Total Value: {metrics["total_value"]:.1f}, ' +
                 f'Weight: {metrics["total_weight"]:.1f}/{instance["capacity"]:.1f}')
        
        plt.xlim(0, 10)
        plt.ylim(0, 20)
        plt.xlabel('Weight')
        plt.ylabel('Value')
        plt.legend()
        
    elif problem_type == 'tsp':
        coordinates = instance['coordinates']
        tour = instance['nearest_neighbor_tour']
        metrics = instance['metrics']
        
        # 绘制城市点
        plt.scatter(coordinates[:, 0], coordinates[:, 1], c='blue', s=100, label='Cities')
        
        # 绘制最近邻路径
        for i in range(len(tour)-1):
            start = coordinates[tour[i]]
            end = coordinates[tour[i+1]]
            plt.plot([start[0], end[0]], [start[1], end[1]], 'r--', alpha=0.6)
        
        # 添加城市编号
        for i, (x, y) in enumerate(coordinates):
            plt.annotate(f'City {i}', (x, y), xytext=(5, 5), textcoords='offset points')
        
        plt.title(f'TSP Instance: {title}\nTour Length: {metrics["tour_length"]:.2f}')
        plt.xlabel('X coordinate')
        plt.ylabel('Y coordinate')
        plt.legend()
    
    plt.grid(True)
    save_path = os.path.join(vis_dir, f'visualization_{problem_type}_{title}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def visualize_solution(instance, solution, problem_type, title=''):
    if problem_type == 'tsp':
        plt.figure(figsize=(10, 10))
        coordinates = instance
        
        # 绘制城市点
        plt.scatter(coordinates[:, 0], coordinates[:, 1], c='blue', s=100)
        
        # 绘制路径
        for i in range(len(solution)-1):
            start = coordinates[solution[i]]
            end = coordinates[solution[i+1]]
            plt.plot([start[0], end[0]], [start[1], end[1]], 'r-')
            
        plt.title(f'TSP Solution: {title}')
        plt.grid(True)
        plt.savefig(f'solution_{problem_type}_{title}.png')
        plt.close()

def main():
    """测试可视化功能"""
    visualizer = ProblemVisualizer()
    
    # 可视化训练集样本
    print("Visualizing training samples...")
    visualizer.visualize_dataset_samples(None, 'tsp')
    visualizer.visualize_dataset_samples(None, 'knapsack')
    
    # 可视化验证集样本
    print("Visualizing validation samples...")
    visualizer.visualize_dataset_samples(None, 'tsp', split='val')
    visualizer.visualize_dataset_samples(None, 'knapsack', split='val')

if __name__ == '__main__':
    main()
