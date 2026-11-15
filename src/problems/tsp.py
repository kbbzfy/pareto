import numpy as np
from typing import List, Dict, Any
from src.problems.base_problem import BaseProblem
import math

class TSPProblem(BaseProblem):
    def __init__(self, config: Dict = None):
        """初始化TSP问题类
        
        Args:
            config: 配置字典，可选
        """
        # 如果没有提供配置，创建一个空配置
        if config is None:
            config = {'problem': {'type': 'tsp', 'size': 20}}
            
        super().__init__(config)
        self.name = 'tsp'
        # 从配置获取大小，如果没有使用默认值
        self.dimension = config.get('problem', {}).get('size', 20)
        self.coordinates = None
        self.distance_matrix = None
        
        # 从配置获取目标数量
        self.num_objectives = config.get('problem', {}).get('n_objectives', 3)
        
        # 从配置获取目标名称
        self.objective_names = config.get('problem', {}).get('objective_names', 
                                         ["总距离", "时间窗口", "平衡性"])
        
        # 从配置获取目标类型（1为最大化，-1为最小化）
        self.objective_types = config.get('problem', {}).get('objective_types', 
                                         [-1, -1, -1])  # 默认全部最小化
        
    def generate_instance(self, size=None):
        """生成TSP问题实例"""
        if size is not None:
            self.dimension = size
        
        # 生成更有意义的城市分布
        # 使用聚类方式生成城市，更接近实际情况
        num_clusters = min(4, self.dimension // 5)  # 城市群的数量
        cluster_centers = np.random.uniform(0.2, 0.8, size=(num_clusters, 2))
        
        coordinates = []
        for _ in range(self.dimension):
            # 随机选择一个聚类中心
            center = cluster_centers[np.random.randint(num_clusters)]
            # 在聚类中心周围生成城市坐标
            point = center + np.random.normal(0, 0.1, 2)
            # 确保坐标在[0,1]范围内
            point = np.clip(point, 0, 1)
            coordinates.append(point)
            
        self.coordinates = np.array(coordinates)
        
        # 计算距离矩阵
        self.distance_matrix = self._compute_distance_matrix()
        
        # 生成一个贪心解（最近邻算法）作为参考
        nearest_neighbor_tour = self._get_nearest_neighbor_solution()
        
        return {
            'coordinates': self.coordinates,
            'distance_matrix': self.distance_matrix,
            'nearest_neighbor_tour': nearest_neighbor_tour,
            'metrics': {
                'tour_length': self._compute_tour_length(nearest_neighbor_tour)
            }
        }
        
    def _compute_distance_matrix(self):
        """计算城市间距离矩阵"""
        n = self.dimension
        matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i+1, n):
                dist = np.linalg.norm(self.coordinates[i] - self.coordinates[j])
                matrix[i][j] = dist
                matrix[j][i] = dist
        return matrix
        
    def _get_nearest_neighbor_solution(self):
        """使用最近邻算法生成一个解"""
        n = self.dimension
        unvisited = set(range(1, n))
        tour = [0]  # 从城市0开始
        
        while unvisited:
            last = tour[-1]
            next_city = min(unvisited, 
                          key=lambda x: self.distance_matrix[last][x])
            tour.append(next_city)
            unvisited.remove(next_city)
            
        tour.append(0)  # 回到起点
        return np.array(tour)
        
    def _compute_tour_length(self, tour):
        """计算路径长度"""
        return sum(self.distance_matrix[tour[i]][tour[i+1]] 
                  for i in range(len(tour)-1))
        
    def compute_objectives(self, solution):
        """计算多个目标函数值"""
        # 初始化目标数组
        objectives = np.zeros(self.num_objectives)
        
        # 目标1：总距离
        total_distance = sum(self.distance_matrix[solution[i]][solution[i+1]] 
                           for i in range(len(solution)-1))
        objectives[0] = total_distance
        
        # 如果只有一个目标，直接返回
        if self.num_objectives == 1:
            return objectives
        
        # 目标2：时间窗口违反程度
        if self.num_objectives > 1:
            current_time = 0
            time_violations = 0
            for i in range(len(solution)-1):
                current_city = solution[i]
                next_city = solution[i+1]
                # 假设每单位距离需要1单位时间
                travel_time = self.distance_matrix[current_city][next_city]
                current_time += travel_time
                
                # 模拟时间窗口约束 [earliest, latest]
                earliest = current_city * 10  # 简单模拟时间窗口
                latest = earliest + 20
                
                if current_time < earliest:
                    time_violations += earliest - current_time
                    current_time = earliest
                elif current_time > latest:
                    time_violations += current_time - latest
            
            objectives[1] = time_violations
        
        # 目标3：路径平衡性（使用相邻段距离的变异系数）
        if self.num_objectives > 2:
            segments = [self.distance_matrix[solution[i]][solution[i+1]] 
                       for i in range(len(solution)-1)]
            mean_dist = np.mean(segments)
            if mean_dist > 0:
                balance = np.std(segments) / mean_dist  # 使用变异系数
            else:
                balance = 0
            
            objectives[2] = balance
        
        # 目标4+：附加目标（如果需要）
        if self.num_objectives > 3:
            # 计算最长路段长度（最大化效率）
            max_segment = max(self.distance_matrix[solution[i]][solution[i+1]] 
                             for i in range(len(solution)-1))
            objectives[3] = max_segment
            
        # 目标5+：根据需要添加更多目标
        # 这里我们使用随机值作为示例，实际应用中应根据问题具体定义
        for i in range(4, self.num_objectives):
            # 使用基于总距离的派生目标
            objectives[i] = total_distance * (0.8 + 0.4 * np.random.random())
        
        return objectives
        
    def is_feasible(self, solution):
        """检查解是否可行"""
        if len(solution) != self.dimension + 1:  # +1 因为要回到起点
            return False
        # 检查是否访问了所有城市恰好一次
        visited = set(solution[:-1])  # 不包括最后回到起点
        return len(visited) == self.dimension and solution[0] == solution[-1]
        
    def get_initial_state(self):
        """获取初始状态"""
        # 确保已经生成了坐标
        if self.coordinates is None:
            self.generate_instance()
            
        return {
            'coordinates': self.coordinates,
            'visited': np.zeros(self.dimension, dtype=np.int32),
            'current_pos': 0  # 从城市0开始
        }
        
    def get_next_state(self, state, action):
        """获取下一个状态"""
        new_state = {
            'coordinates': state['coordinates'],
            'visited': state['visited'].copy(),
            'current_pos': action
        }
        new_state['visited'][action] = 1
        return new_state
        
    def get_random_solution(self):
        """生成随机解决方案"""
        # 生成一个随机排列
        cities = np.arange(1, self.dimension)
        np.random.shuffle(cities)
        tour = np.concatenate(([0], cities, [0]))  # 从城市0开始，最后回到0
        return tour
        
    def get_solution_space_size(self):
        """获取解空间大小"""
        # TSP的解空间大小是(n-1)!（第一个城市固定为0）
        return math.factorial(self.dimension - 1)
