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

from replay_memory import ReplayMemory, ReplayMemoryDataset

# Neural Network for Deep Q-Learning
class QNetwork(nn.Module):

    def __init__(self, state_size, action_size, hidden_size=64):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class DQLAgent:
    
    def __init__(self, num_actions, state_size, selection_method,
                 alpha=0.001, gamma=0.9, epsilon=0.1, buffer_size=10000,
                 batch_size=64, target_update_freq=1000, tau=0.001):
        self.num_actions = num_actions
        self.state_size = state_size
        self.selection_method = selection_method
        self.gamma = gamma
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.tau = tau  # Soft update rate

        # Neural networks
        self.policy_net = QNetwork(state_size, num_actions)
        self.target_net = QNetwork(state_size, num_actions)
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=alpha)
        self.loss_fn = nn.MSELoss()

        # Replay Memory and DataLoader
        self.memory = ReplayMemory(buffer_size)
        self.data_loader = None
        self.steps_done = 0

        # Initialize DataLoader if replay memory is prefilled
        self._initialize_data_loader()

    def store_experience(self, state, action, reward, next_state, done):
        """Store an experience in the replay buffer."""
        self.memory.push((state, action, reward, next_state, done))

        # Reinitialize DataLoader periodically
        if len(self.memory) % 100 == 0 and len(self.memory) > 0:
            self._initialize_data_loader()

    def _initialize_data_loader(self):
        """Initialize or reinitialize the DataLoader for the replay memory."""
        if len(self.memory) > 0:  # Ensure replay memory has at least one experience
            dataset = ReplayMemoryDataset(self.memory)
            self.data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        else:
            self.data_loader = None  # No DataLoader if memory is empty

    def train(self):
        """Train the agent using a sampled batch of experiences."""
        if self.data_loader is None:  # Skip training if DataLoader is not ready
            return None

        total_loss = 0
        for batch in self.data_loader:
            states, actions, rewards, next_states, dones = batch

            # Current Q values
            current_q = self.policy_net(states).gather(1, actions)

            # Compute target Q values
            with torch.no_grad():
                max_next_q = self.target_net(next_states).max(1)[0].unsqueeze(1)
                target_q = rewards + self.gamma * max_next_q * (1 - dones)

            # Compute loss
            loss = self.loss_fn(current_q, target_q)
            total_loss += loss.item()

            # Optimize the model
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # Perform soft update of the target network
        self.soft_update(self.policy_net, self.target_net)

        # Return average loss for the batch
        return total_loss / len(self.data_loader)

    def soft_update(self, local_model, target_model):
        """Soft update model parameters."""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

    def select_action(self, state):
        """Select an action based on epsilon-greedy or softmax strategy."""
        if self.selection_method == 'e-greedy':
            # Epsilon-greedy strategy
            if random.random() < self.epsilon:
                # Exploration: choose a random action
                action = random.randint(0, self.num_actions - 1)
                print("Action (Exploration - Epsilon-Greedy):", action)
                return action
            else:
                # Exploitation: choose the action with the highest Q-value
                q_values = self.get_q_values(state)
                action = np.argmax(q_values)
                print("Action (Exploitation - Epsilon-Greedy):", action)
                return action
        elif self.selection_method == 'softmax':
            # Softmax strategy
            q_values = self.get_q_values(state)
            exp_q_values = np.exp(q_values - np.max(q_values))  # For numerical stability
            probabilities = exp_q_values / np.sum(exp_q_values)
            print("Probabilities (Softmax):", probabilities)
            action = np.random.choice(range(self.num_actions), p=probabilities)
            print("Action (Softmax):", action)
            return action
        
    def get_q_values(self, state):
        """Retrieve Q-values for a given state from the policy network."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(next(self.policy_net.parameters()).device)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor).squeeze(0).cpu().numpy()
        return q_values

    def update_q_value(self, state, action, reward, next_state, done):
        """Update Q-values by storing the experience and training the network."""
        # Store the experience in the replay memory
        self.store_experience(state, action, reward, next_state, done)
        
        # Train the network
        self.train()
