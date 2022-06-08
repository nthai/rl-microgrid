from collections import deque
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Memory():
    def __init__(self, capacity=1000) -> None:
        self.capacity = capacity
        self.memory = deque(maxlen=self.capacity)

    def sample(self, sample_size=64):
        data = None
        try:
            data = random.sample(self.memory, sample_size)
        except ValueError: # happens if sample_size > len(data)
            data = random.sample(self.memory, len(self.memory))
        
        prev_states = []
        prev_masks = []
        actions = []
        qfactors = []
        rewards = []
        next_states = []
        dones = []

        for pstate, pmask, act, qfact, rew, nstate, done in data:
            prev_states.append(pstate)
            prev_masks.append(pmask)
            actions.append(act)
            qfactors.append(qfact)
            rewards.append(rew)
            next_states.append(nstate)
            dones.append(done)
        
        return (torch.tensor(np.array(prev_states), dtype=torch.float32, device=device),
                torch.tensor(np.array(prev_masks), dtype=torch.float32, device=device),
                torch.tensor(np.array(actions), dtype=torch.int64, device=device),
                torch.tensor(np.array(qfactors), dtype=torch.float32, device=device),
                torch.tensor(np.array(rewards), dtype=torch.float32, device=device),
                torch.tensor(np.array(next_states), dtype=torch.float32, device=device),
                torch.tensor(np.array(dones), dtype=torch.float32, device=device))

    def store(self, transition):
        self.memory.append(transition)

class DQNAgent(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()

        self.dim_state = config['state_size']
        self.num_actions = config['action_size']
        self.hidden_nodes = config['hidden_nodes']
        self.learn_rate = config['learning_rate']
        self.epsilon = config['epsilon']
        self.gamma = config['gamma']

        self.memory = Memory()

        self.input_layer = nn.Linear(self.dim_state, self.hidden_nodes)
        self.output_layer = nn.Linear(self.hidden_nodes, self.num_actions)

        nn.init.xavier_uniform_(self.input_layer.weight)
        nn.init.xavier_uniform_(self.output_layer.weight)

        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=self.learn_rate)

        self.to(device=device)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.input_layer(inp))
        x = self.output_layer(x)
        return x

    def select_action(self, state: list, mask: list) -> int:
        state = torch.tensor(state, dtype=torch.float32, device = device)
        # TODO: shouldn't ignore mask
        # mask = torch.tensor(mask, dtyep=torch.float32, device=device)

        qfactors = self.forward(state)
        action = 0
        if torch.rand([1]) < self.epsilon:
            action = torch.randint(0, self.num_actions, [1]).item()
        else:
            action = qfactors.argmax().item()
        return action, qfactors[action].item()

    def store(self, prev_state, prev_mask, action, qfactor, reward,
              next_state, done):
        self.memory.store((prev_state, prev_mask, action, qfactor, reward,
                           next_state, done))
    
    def train(self):
        pstates, _, actions, _, rewards, nstates, dones \
            = self.memory.sample()

        prevq = self(pstates)
        prevq = prevq.gather(1, actions.unsqueeze(1))
        maxq = self(nstates).max(1).values.unsqueeze(1)
        dones = (1 - dones).unsqueeze(1)

        target = rewards + self.gamma * maxq * dones
        loss = self.criterion(prevq, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
