import gurobipy as gp
from gurobipy import GRB
import numpy as np
import random
import copy  # For deep copying
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader



class ReplayMemory(Dataset):  # Extend Dataset for DataLoader compatibility
    def __init__(self, capacity):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.capacity = capacity
        self.memory = []  # Stores experiences as tuples (state, action, reward, next_state, done)

    def push(self, event):
        """Add a new event to the replay buffer."""
        self.memory.append(event)
        if len(self.memory) > self.capacity:
            del self.memory[0]

    def __len__(self):
        """Return the current size of the replay buffer."""
        return len(self.memory)

    def __getitem__(self, idx):
        """Return the experience at the given index."""
        state, action, reward, next_state, done = self.memory[idx]
        state = torch.tensor(state, dtype=torch.float32, device=self.device)
        action = torch.tensor(action, dtype=torch.long, device=self.device).unsqueeze(0)
        reward = torch.tensor(reward, dtype=torch.float32, device=self.device).unsqueeze(0)
        next_state = torch.tensor(next_state, dtype=torch.float32, device=self.device)
        done = torch.tensor(done, dtype=torch.float32, device=self.device).unsqueeze(0)
        return state, action, reward, next_state, done

    def sample(self, batch_size):
        """Sample a batch of experiences."""
        indices = random.sample(range(len(self.memory)), batch_size)
        return [self[i] for i in indices]

class ReplayMemoryDataset(torch.utils.data.Dataset):
    def __init__(self, memory):
        self.memory = memory.memory  # Access the internal memory list

    def __len__(self):
        return len(self.memory)

    def __getitem__(self, idx):
        state, action, reward, next_state, done = self.memory[idx]
        return (torch.FloatTensor(state),
                torch.LongTensor([action]),
                torch.FloatTensor([reward]),
                torch.FloatTensor(next_state),
                torch.FloatTensor([done]))

