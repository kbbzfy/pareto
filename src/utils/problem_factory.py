from src.problems.knapsack_problem import KnapsackProblem
from src.problems.tsp import TSPProblem
import yaml
import os

class ProblemFactory:
    _instance = None  # 单例实例
    _config = None    # 配置信息
    
    @classmethod
    def initialize(cls, config_path=None, config_dict=None):
        """初始化工厂配置
        
        Args:
            config_path: 配置文件路径
            config_dict: 配置字典对象（优先于文件）
        """
        if config_dict:
            cls._config = config_dict
        elif config_path and os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                cls._config = yaml.safe_load(f)
        else:
            # 尝试默认配置路径
            default_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'configs', 'knapsack_config.yaml')
            if os.path.exists(default_path):
                with open(default_path, 'r', encoding='utf-8') as f:
                    cls._config = yaml.safe_load(f)
            else:
                # 使用空配置
                cls._config = {'problem': {}, 'model': {}, 'data': {}}
                print("警告：未找到配置文件，使用默认空配置")
    
    @classmethod
    def get_config(cls):
        """获取当前配置"""
        if cls._config is None:
            cls.initialize()
        return cls._config
            
    @classmethod
    def create_problem(cls, problem_type, config=None):
        """创建问题实例
        
        Args:
            problem_type: 问题类型 ('knapsack' 或 'tsp')
            config: 可选的配置参数，优先级高于默认配置
            
        Returns:
            BaseProblem: 问题实例
        """
        # 确保有配置
        if cls._config is None and config is None:
            cls.initialize()
            
        # 使用传入的配置或默认配置
        problem_config = config if config is not None else cls._config
        
        if problem_type == 'knapsack':
            return KnapsackProblem(problem_config)
        elif problem_type == 'tsp':
            tsp_problem = TSPProblem(problem_config)
            return tsp_problem
        else:
            raise ValueError(f"未知问题类型: {problem_type}")

# 创建统一的工厂接口
problem_factory = ProblemFactory 