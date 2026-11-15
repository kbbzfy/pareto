import unittest
import torch
import numpy as np
from src.models.fast_solver import FastSolver

class TestFastSolver(unittest.TestCase):
    def setUp(self):
        """初始化测试环境"""
        self.input_dim = 128
        self.hidden_dim = 256
        self.model = FastSolver(self.input_dim, self.hidden_dim)
        self.model.problem_type = 'tsp'  # 设置问题类型
        
    def test_model_initialization(self):
        """测试模型初始化"""
        self.assertEqual(self.model.encoder[0].in_features, self.input_dim)
        self.assertEqual(self.model.encoder[0].out_features, self.hidden_dim)
        self.assertEqual(self.model.decoder[-1].out_features, self.input_dim)
        
    def test_forward_pass(self):
        """测试前向传播"""
        batch_size = 32
        x = torch.randn(batch_size, self.input_dim)
        solution, objectives = self.model(x)
        
        # 检查输出维度
        self.assertEqual(solution.shape, (batch_size, self.input_dim))
        self.assertEqual(objectives.shape, (batch_size, 2))  # 两个目标值
        
    def test_tsp_solution_processing(self):
        """测试TSP问题的解决方案处理"""
        batch_size = 4
        x = torch.randn(batch_size, self.input_dim)
        solution, objectives = self.model.predict(x)
        
        # 检查TSP解决方案是否为有效排列
        for i in range(batch_size):
            sol = solution[i].cpu().numpy()
            self.assertEqual(len(np.unique(sol)), len(sol))  # 检查是否包含所有城市
            self.assertTrue(np.all(sol >= 0))  # 检查索引非负
            self.assertTrue(np.all(sol < self.input_dim))  # 检查索引在有效范围内
            
    def test_objective_calculation(self):
        """测试目标函数计算"""
        # 创建示例TSP问题数据
        problem_data = {
            'cities': [
                {'x': 0, 'y': 0, 'time_window': (0, 10)},
                {'x': 3, 'y': 4, 'time_window': (5, 15)},
                {'x': 6, 'y': 8, 'time_window': (10, 20)}
            ]
        }
        
        # 创建示例解决方案
        solution = torch.tensor([0, 1, 2])
        objectives = self.model._calculate_tsp_objectives(solution, problem_data)
        
        # 检查目标值
        self.assertEqual(objectives.shape, (2,))  # 距离和时间惩罚
        self.assertGreaterEqual(objectives[0].item(), 0)  # 距离非负
        self.assertGreaterEqual(objectives[1].item(), 0)  # 时间惩罚非负
        
    def test_knapsack_solution_processing(self):
        """测试背包问题的解决方案处理"""
        self.model.problem_type = 'knapsack'
        batch_size = 4
        x = torch.randn(batch_size, self.input_dim)
        solution, _ = self.model.predict(x)
        
        # 检查解决方案是否为二进制值
        self.assertTrue(torch.all((solution == 0) | (solution == 1)))
        
    def test_data_augmentation_compatibility(self):
        """测试与数据增强的兼容性"""
        batch_size = 8
        x = torch.randn(batch_size, self.input_dim)
        
        # 测试模型对增强数据的处理能力
        augmented_x = x + torch.randn_like(x) * 0.1  # 添加噪声
        solution1, obj1 = self.model(x)
        solution2, obj2 = self.model(augmented_x)
        
        # 检查输出维度一致性
        self.assertEqual(solution1.shape, solution2.shape)
        self.assertEqual(obj1.shape, obj2.shape)

if __name__ == '__main__':
    unittest.main()
