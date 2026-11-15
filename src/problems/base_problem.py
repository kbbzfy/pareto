from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any
import numpy as np

class BaseProblem(ABC):
    def __init__(self, config: Dict):
        """初始化基类
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.name = None
        self.dimension = None
    
    @abstractmethod
    def generate_instance(self, size):
        """生成问题实例"""
        pass
    
    @abstractmethod
    def compute_objectives(self, solution: Any) -> np.ndarray:
        """计算目标函数值"""
        pass
    
    @abstractmethod
    def is_feasible(self, solution: Any) -> bool:
        """检查解决方案是否可行"""
        pass
    
    @abstractmethod
    def get_initial_state(self) -> Dict:
        """获取初始状态"""
        pass
    
    @abstractmethod
    def get_next_state(self, state: Dict, action: Any) -> Dict:
        """获取下一个状态"""
        pass
    
    @abstractmethod
    def get_random_solution(self) -> Any:
        """生成随机解决方案"""
        pass
    
    @abstractmethod
    def get_solution_space_size(self) -> int:
        """获取解空间大小"""
        pass 