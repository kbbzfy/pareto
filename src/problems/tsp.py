import numpy as np
from typing import List, Dict, Any
from src.problems.base_problem import BaseProblem
import math

TSP_OBJECTIVE_NAMES = ("总距离", "时间窗口迟到量", "路段长度变异系数", "最长路段")


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
        
        problem_config = config.get('problem', {})
        self.num_objectives = problem_config.get('num_objectives', problem_config.get('n_objectives', 3))
        if not 1 <= self.num_objectives <= len(TSP_OBJECTIVE_NAMES):
            raise ValueError("TSP 目前支持 1 至 4 个有明确计算定义的目标")

        names = list(problem_config.get('objective_names') or ())[:self.num_objectives]
        aliases = ({}, {"时间窗口": TSP_OBJECTIVE_NAMES[1]},
                   {"平衡性": TSP_OBJECTIVE_NAMES[2]},
                   {"成本": TSP_OBJECTIVE_NAMES[3], "cost": TSP_OBJECTIVE_NAMES[3],
                    "最大段长": TSP_OBJECTIVE_NAMES[3]})
        self.objective_names = [
            aliases[i].get(name.casefold(), name) for i, name in enumerate(names)
        ] + list(TSP_OBJECTIVE_NAMES[len(names):self.num_objectives])
        if len(set(self.objective_names)) != self.num_objectives:
            raise ValueError("TSP 目标名称必须唯一")
        directions = list(problem_config.get('objective_types') or ())[:self.num_objectives]
        self.objective_types = directions + [-1] * (self.num_objectives - len(directions))
        if any(direction not in (-1, 1) for direction in self.objective_types):
            raise ValueError("TSP 目标方向必须为 1（最大化）或 -1（最小化）")

    def generate_instance(self, size=None):
        """生成TSP问题实例"""
        if size is not None:
            self.dimension = size
        
        if self.dimension < 2:
            raise ValueError("TSP 至少需要两个城市")

        # 生成更有意义的城市分布
        # 使用聚类方式生成城市，更接近实际情况
        num_clusters = max(1, min(4, self.dimension // 5))  # 城市群的数量
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
        if not self.is_feasible(solution):
            raise ValueError("TSP 目标值必须基于完整闭合路线计算")

        segments = self.distance_matrix[solution[:-1], solution[1:]]
        objectives = [float(segments.sum())]
        if self.num_objectives >= 2:
            # 示例窗口由城市 ID 定义，到达过早可以等待，回到出发城市无需再次服务。
            arrival = 0.0
            lateness = 0.0
            for city, travel_time in zip(solution[1:-1], segments):
                earliest = city * 10
                arrival = max(arrival + travel_time, earliest)
                lateness += max(0.0, arrival - (earliest + 20))
            objectives.append(lateness)
        if self.num_objectives >= 3:
            mean_length = segments.mean()
            objectives.append(segments.std() / mean_length if mean_length else 0.0)
        if self.num_objectives >= 4:
            objectives.append(segments.max())
        return np.asarray(objectives)

    def is_feasible(self, solution):
        """检查解是否可行"""
        if len(solution) != self.dimension + 1:  # +1 因为要回到起点
            return False
        # 检查是否访问了所有城市恰好一次
        visited = set(solution[:-1])  # 不包括最后回到起点
        return visited == set(range(self.dimension)) and solution[0] == solution[-1]
        
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
