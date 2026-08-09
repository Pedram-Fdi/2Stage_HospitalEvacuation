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
from replay_memory import ReplayMemory

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

    def __init__(self,
                 num_actions,
                 state_size,
                 selection_method,
                 alpha=1e-3,
                 gamma=0.9,
                 epsilon_start=1.0,       # new
                 epsilon_final=0.1,       # new
                 max_steps=10000,         # new
                 decay_fraction=0.1,      # first 10%
                 buffer_size=5000,
                 batch_size=64,
                 target_update_freq=1000,
                 tau=0.001,
                 learning_starts=32):
        self.num_actions = num_actions
        self.state_size = state_size
        self.selection_method = selection_method
        self.gamma = gamma
        self.epsilon_start       = epsilon_start
        self.epsilon_final       = epsilon_final
        self.max_steps           = max_steps
        self.decay_steps         = int(decay_fraction * max_steps)
        self.steps_done          = 0
        self.epsilon             = epsilon_start
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.tau = tau  # Soft update rate
        # Minimum number of stored transitions before the first gradient step.
        self.learning_starts = max(1, min(learning_starts, batch_size))

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Neural networks
        self.policy_net = QNetwork(state_size, num_actions).to(self.device)
        self.target_net = QNetwork(state_size, num_actions).to(self.device)
        # The target network must start as a copy of the policy network, otherwise the
        # bootstrapped targets are pure noise during the first updates.
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=alpha)
        self.loss_fn = nn.SmoothL1Loss()

        # Replay Memory
        self.memory = ReplayMemory(buffer_size)
        self.memory.device = self.device
        self.steps_done = 0
        self.train_steps = 0

    def store_experience(self, state, action, reward, next_state, done):
        """Store an experience in the replay buffer."""
        self.memory.push((state, action, reward, next_state, done))

    def sample_batch(self):
        """Sample one mini-batch of experiences and stack it into batched tensors."""
        batch = self.memory.sample(min(self.batch_size, len(self.memory)))
        states = torch.stack([experience[0] for experience in batch])
        actions = torch.stack([experience[1] for experience in batch])
        rewards = torch.stack([experience[2] for experience in batch])
        next_states = torch.stack([experience[3] for experience in batch])
        dones = torch.stack([experience[4] for experience in batch])
        return states, actions, rewards, next_states, dones

    def train(self):
        """Train the agent on a single mini-batch sampled from the replay buffer."""
        if len(self.memory) < self.learning_starts:
            return None

        states, actions, rewards, next_states, dones = self.sample_batch()

        # Current Q values
        current_q = self.policy_net(states).gather(1, actions)

        # Compute target Q values
        with torch.no_grad():
            max_next_q = self.target_net(next_states).max(1)[0].unsqueeze(1)
            target_q = rewards + self.gamma * max_next_q * (1 - dones)

        loss = self.loss_fn(current_q, target_q)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
        self.optimizer.step()

        self.train_steps += 1
        self.update_target_network()

        return loss.item()

    def update_target_network(self):
        """Refresh the target network, either periodically (hard) or every step (soft)."""
        if self.target_update_freq and self.target_update_freq > 0:
            if self.train_steps % self.target_update_freq == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
        else:
            self.soft_update(self.policy_net, self.target_net)

    def soft_update(self, local_model, target_model):
        """Soft update model parameters."""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

    def select_action(self, state):
        """Select an action based on epsilon-greedy or softmax strategy."""
        # 1) bump your step counter
        self.steps_done += 1

        # 2) linearly decay ε over the first `decay_steps` calls
        if self.steps_done <= self.decay_steps:
            frac = self.steps_done / self.decay_steps
            self.epsilon = self.epsilon_start - frac * (self.epsilon_start - self.epsilon_final)
        else:
            # beyond the first 10%, freeze at ε_final
            self.epsilon = self.epsilon_final

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
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor).squeeze(0).cpu().numpy()
        return q_values

    def update_q_value(self, state, action, reward, next_state, done):
        """Update Q-values by storing the experience and training the network."""
        # Store the experience in the replay memory
        self.store_experience(state, action, reward, next_state, done)
        
        # Train the network
        self.train()