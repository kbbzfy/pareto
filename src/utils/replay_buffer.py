import random
from collections import namedtuple, deque
from typing import List, Tuple

Experience = namedtuple('Experience', 
    ['state', 'action', 'reward', 'next_state', 'done'])

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        
    def push(self, experience: Experience):
        """添加经验"""
        self.buffer.append(experience)
        
    def sample(self, batch_size: int) -> List[Experience]:
        """随机采样经验批次"""
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))
    
    def __len__(self) -> int:
        return len(self.buffer) 