import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PPOAgent(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.dim_state = config['state_size']
        self.num_actions = config['action_size']
        self.hidden_nodes = config["hidden_nodes"]
        self.learn_rate = config["learning_rate"]
        self.epochs = config["epochs"] # should be 3...30
        self.batch_size = config["batch_size"] # should be ~32
        self.alphaR = config["alphaR"] # should be ~ 0.1
        self.eps_clips = config["eps_clips"] # should be 0.1 or 0.2
        self.gae_lambda = config["gae_lambda"] # should be ~ 0.9

        self.input_layer = nn.Linear(self.dim_state, self.hidden_nodes)
        self.policy_layer = nn.Linear(self.hidden_nodes, self.num_actions)
        self.value_layer = nn.Linear(self.hidden_nodes, 1)

        nn.init.xavier_uniform_(self.input_layer.weight)
        nn.init.xavier_uniform_(self.policy_layer.weight)
        nn.init.xavier_uniform_(self.value_layer.weight)

        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=self.learn_rate)

        self._reset_history()
        self.avg_reward = 0
        self.to(device=device)

    def _reset_history(self):
        self.prev_states = []
        self.prev_masks = []
        self.actions = []
        self.probs = []
        self.rewards = []
        self.next_states = []

    def _policy(self, state_tensor, mask_tensor):
        dim = 1 if state_tensor.dim() == 2 else 0
        vartensor = F.relu(self.input_layer(state_tensor))
        mask_logits = torch.clamp(torch.log(mask_tensor),
                                  min=torch.finfo(torch.float32).min)
        vartensor = self.policy_layer(vartensor) + mask_logits
        probs = F.softmax(vartensor, dim=dim)
        return probs

    def _value(self, state_tensor):
        vartensor = F.relu(self.input_layer(state_tensor))
        value = self.value_layer(vartensor)
        return value

    def select_action(self, state, mask):
        state = torch.tensor(state, dtype=torch.float32, device=device)
        mask = torch.tensor(mask, dtype=torch.float32, device=device)

        probs = self._policy(state, mask)
        cat = Categorical(probs)
        action = cat.sample().item()
        prob = probs[action].item()
        return action, prob

    def store(self, prev_state, prev_mask, action, prob,
              reward, next_state, done):
        self.prev_states.append(prev_state)
        self.prev_masks.append(prev_mask)
        self.actions.append(action)
        self.probs.append(prob)
        self.rewards.append(reward)
        self.next_states.append(next_state)

    def _get_history_tensors(self):
        pstates = torch.tensor(np.array(self.prev_states), dtype=torch.float32, device=device)
        pmasks = torch.tensor(np.array(self.prev_masks), dtype=torch.float32, device=device)
        actions = torch.tensor(np.array(self.actions), dtype=torch.long, device=device)
        probs = torch.tensor(np.array(self.probs), dtype=torch.float32, device=device)
        rewards = torch.tensor(np.array(self.rewards), dtype=torch.float32, device=device)
        nstates = torch.tensor(np.array(self.next_states), dtype=torch.float32, device=device)

        return pstates, pmasks, actions, probs, rewards, nstates

    def train(self):
        if len(self.prev_states) < self.batch_size:
            return

        pstates, pmasks, actions, probs, rewards, nstates = self._get_history_tensors()

        self.avg_reward = ((1 - self.alphaR) * self.avg_reward +
                           self.alphaR * torch.mean(rewards).item())

        for _ in range(self.epochs):
            pvalues = self._value(pstates).squeeze()
            nvalues = self._value(nstates).squeeze()
            # td_target = rewards - self.avg_reward + nvalues
            td_target = rewards.view(-1) - self.avg_reward + nvalues.view(-1)

            delta = td_target - pvalues # this is the TD-error
            delta = delta.cpu()
            delta = delta.detach().numpy()

            advantages = []
            advantage = 0
            for tderr in delta[::-1]:
                advantage = self.gae_lambda * advantage + tderr
                advantages.append(advantage)
            advantages = np.array(list(reversed(advantages)))
            advantages = torch.tensor(advantages, dtype=torch.float32, device=device)

            pdistr = self._policy(pstates, pmasks)
            pprobs = pdistr.gather(1, actions.unsqueeze(1)).squeeze()
            ratio = torch.exp(torch.log(pprobs) - torch.log(probs))

            default_surr = ratio * advantages
            clipped_surr = torch.clamp(ratio, 1-self.eps_clips, 1+self.eps_clips) * advantages
            loss = (-torch.min(default_surr, clipped_surr) +
                    self.criterion(pvalues, td_target.detach()))

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        self._reset_history()
    
    def reset(self):
        pass