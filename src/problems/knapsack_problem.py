import numpy as np
from typing import Dict, List, Tuple
from src.problems.base_problem import BaseProblem

class KnapsackProblem(BaseProblem):
    def __init__(self, config: Dict):
        """初始化多目标背包问题
        
        Args:
            config: 配置字典，包含问题参数
        """
        super().__init__(config)
        self.num_categories = config['problem']['num_categories']
        self.items_per_category = config['problem']['items_per_category']
        self.num_objectives = config['problem']['num_objectives']
        
        # 从配置中获取目标名称，如果没有则使用默认值
        self.objective_names = config.get('problem', {}).get('objective_names')
        if not self.objective_names:
            self.objective_names = config.get('model', {}).get('objective_names', 
                ['objective_1'] + [f'objective_{i+1}' for i in range(1, self.num_objectives)])
        
        self.objective_names = list(self.objective_names)

        # 确保目标名称数量匹配目标数量
        while len(self.objective_names) < self.num_objectives:
            self.objective_names.append(f"objective_{len(self.objective_names) + 1}")
        
        # 限制目标名称不超过目标数量
        if len(self.objective_names) > self.num_objectives:
            self.objective_names = self.objective_names[:self.num_objectives]
        
        if len(set(self.objective_names)) != self.num_objectives:
            raise ValueError("目标名称必须唯一")

        # 生成物品数据
        self.items = self._generate_items()
        
        self._update_capacities()

    def _update_capacities(self):
        limits = self.config['problem'].get('objective_constraints')
        if limits is None:
            totals = self.items.sum(axis=(0, 1))
            limits = dict(zip(self.objective_names[1:], totals[1:] * 0.5))

        unknown_names = limits.keys() - set(self.objective_names)
        if unknown_names:
            raise ValueError(f"未知的容量约束目标: {sorted(unknown_names)}")
        for name, limit in limits.items():
            if not np.isfinite(limit) or limit < 0:
                raise ValueError(f"目标 {name} 的容量必须是有限的非负数")

        self.capacities = np.array(
            [limits.get(name, np.inf) for name in self.objective_names]
        )
        for name, capacity in zip(self.objective_names, self.capacities):
            setattr(self, f"{name}_capacity", capacity)
        names_to_capacities = {
            name.casefold(): capacity for name, capacity in zip(self.objective_names, self.capacities)
        }
        for index, name in enumerate(('weight', 'volume', 'cost'), start=1):
            default = self.capacities[index] if index < self.num_objectives else np.inf
            setattr(self, f"{name}_capacity", names_to_capacities.get(name, default))

    def generate_instance(self, size=None):
        """生成问题实例
        
        Args:
            size: 问题规模（如果为None，使用配置中的值）
            
        Returns:
            Dict: 包含问题实例信息的字典
        """
        # 生成新的物品数据
        self.items = self._generate_items()
        
        self._update_capacities()

        # 生成一个随机初始解
        initial_solution = self.get_random_solution()
        
        return {
            'items': self.items,
            'weight_capacity': self.weight_capacity,
            'volume_capacity': self.volume_capacity,
            'cost_capacity': self.cost_capacity,
            'initial_solution': initial_solution,
            'metrics': self.compute_objectives(initial_solution)
        }
        
    def generate_instance_from_data(self, items_data):
        """从给定的物品数据生成问题实例
        
        Args:
            items_data: numpy数组，包含物品的属性数据
            
        Returns:
            Dict: 包含问题实例信息的字典，如果数据无效则返回None
        """
        try:
            # 确保数据形状正确
            if not isinstance(items_data, np.ndarray):
                items_data = np.array(items_data)
            
            if items_data.shape != self.items.shape:
                return None
                
            # 更新物品数据
            self.items = items_data
            
            self._update_capacities()

            # 生成一个随机初始解
            initial_solution = self.get_random_solution()
            
            # 计算目标值
            objectives = self.compute_objectives(initial_solution)
            
            return {
                'items': self.items,
                'weight_capacity': self.weight_capacity,
                'volume_capacity': self.volume_capacity,
                'cost_capacity': self.cost_capacity,
                'initial_solution': initial_solution,
                'metrics': objectives
            }
        except Exception as e:
            print(f"Error generating instance from data: {e}")
            return None
        
    def _generate_items(self) -> np.ndarray:
        """生成物品数据
        Returns:
            np.ndarray: 形状为 [num_categories, items_per_category, num_objectives] 的数组
                       每个物品有多个属性，第一个是价值，其余是约束
        """
        # 创建适应目标数量的数组
        items = np.zeros((self.num_categories, self.items_per_category, self.num_objectives))
        
        # 为每个目标定义基准值范围
        base_ranges = []
        for i in range(self.num_objectives):
            if i == 0:  # 价值（第一个目标）
                base_ranges.append((50, 200))
            else:  # 其他目标
                # 使用递减的范围，确保后续目标的值不会过大
                base_min = max(5, 50 / (i + 1))
                base_max = max(20, 200 / (i + 1))
                base_ranges.append((base_min, base_max))
        
        # 为每个类别设置不同的基准值范围
        for i in range(self.num_categories):
            # 为每个类别和每个目标设置基准值
            base_values = [np.random.uniform(low, high) for low, high in base_ranges]
            
            for j in range(self.items_per_category):
                # 在基准值的基础上添加随机波动（±20%）
                objective_values = [base * np.random.uniform(0.8, 1.2) for base in base_values]
                
                # 添加一些极端值（约10%的概率）
                if np.random.random() < 0.1:
                    # 生成极高或极低的值
                    multiplier = np.random.choice([0.5, 1.5])  # 50%更低或50%更高
                    objective_values = [val * multiplier for val in objective_values]
                
                items[i, j] = objective_values
        
        return items
        
    def get_initial_state(self) -> Dict:
        """获取初始状态
        Returns:
            Dict: 包含初始状态信息的字典
        """
        return {
            'items': self.items.copy(),
            'current_category': 0,
            'selected': np.zeros(self.num_categories, dtype=np.int32)
        }
        
    def get_next_state(self, state: Dict, action: int) -> Dict:
        """获取执行动作后的下一个状态
        
        Args:
            state: 当前状态
            action: 选择的动作（物品索引）
            
        Returns:
            Dict: 下一个状态
        """
        next_state = {
            'items': state['items'].copy(),
            'current_category': state['current_category'] + 1,
            'selected': state['selected'].copy()
        }
        
        next_state['selected'][state['current_category']] = action
        return next_state
        
    def compute_objectives(self, selected: np.ndarray) -> np.ndarray:
        """计算目标函数值
        
        Args:
            selected: 选择的物品索引数组
            
        Returns:
            np.ndarray: 所有目标函数的值
        """
        # 初始化目标函数值数组
        objectives = np.zeros(self.num_objectives)
        
        for category in range(self.num_categories):
            item_idx = selected[category]
            item = self.items[category, item_idx]
            
            # 累加每个目标的值
            for i in range(self.num_objectives):
                objectives[i] += item[i]
            
        return objectives
        
    def is_feasible(self, selected: np.ndarray) -> bool:
        """判断选择是否满足约束条件
        
        Args:
            selected: 选择的物品索引数组
            
        Returns:
            bool: 是否可行
        """
        selected = np.asarray(selected)
        if (selected.shape != (self.num_categories,)
                or not np.issubdtype(selected.dtype, np.integer)
                or np.any(selected < 0)
                or np.any(selected >= self.items_per_category)):
            return False
        return bool(np.all(self.compute_objectives(selected) <= self.capacities))

    def get_random_solution(self) -> np.ndarray:
        """生成随机解决方案
        
        Returns:
            np.ndarray: 随机选择的物品索引数组
        """
        return np.random.randint(0, self.items_per_category, size=self.num_categories)
        
    def get_solution_space_size(self) -> int:
        """获取解空间大小
        
        Returns:
            int: 解空间大小
        """
        return self.items_per_category ** self.num_categories 