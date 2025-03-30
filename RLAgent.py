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
# Class for Reinforcement Learning Agent (Q-learning)

class RLAgent:

    def __init__(self, num_actions, selection_method, alpha=0.2, gamma=0.9, epsilon=0.1):
        self.q_table = {}  # Q-values for state-action pairs
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.num_actions = num_actions
        self.selection_method = selection_method  # 'e-greedy' or 'softmax'

    def get_q_value(self, state, action):
        # Initialize unseen state-action pairs with small random values (noise)
        if (state, action) not in self.q_table:
            self.q_table[(state, action)] = np.random.uniform(-0.1, 0.1)
        return self.q_table[(state, action)]

    def update_q_value(self, state, action, reward, next_state):
        # Update Q-value based on the reward and next state's best action
        best_next_action = max([self.get_q_value(next_state, a) for a in range(self.num_actions)])
        current_q_value = self.get_q_value(state, action)
        new_q_value = current_q_value + self.alpha * (reward + self.gamma * best_next_action - current_q_value)
        self.q_table[(state, action)] = new_q_value

    def select_action(self, state):
        if self.selection_method == 'e-greedy':
            # Epsilon-greedy action selection
            if random.random() < self.epsilon:
                action = random.randint(0, self.num_actions - 1)  # Explore
                print("Action (Exploration - Epsilon-Greedy):", action)
                return action
            else:
                q_values = [self.get_q_value(state, a) for a in range(self.num_actions)]
                action = np.argmax(q_values)  # Exploit
                print("Action (Exploitation - Epsilon-Greedy):", action)
                return action
        elif self.selection_method == 'softmax':
            # Softmax action selection
            q_values = np.array([self.get_q_value(state, a) for a in range(self.num_actions)])
            exp_q_values = np.exp(q_values - np.max(q_values))  # For numerical stability
            probabilities = exp_q_values / np.sum(exp_q_values)
            print("Probabilities (Softmax):", probabilities)
            action = np.random.choice(range(self.num_actions), p=probabilities)
            print("Action (Softmax):", action)
            return action

