import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

class Critic(nn.Module):
    def __init__(self, state_size, action_size,hidden_size=256, seed=1):
        super().__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)

    def forward(self, state):
        x = F.leaky_relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class Policy(nn.Module): #Actor
    def __init__(self, state_size, action_size, seed=1, hidden_size=256):
        super().__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        self.softmax = nn.Softmax(dim=-1)

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)

    def forward(self, state):
        x = F.leaky_relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        action_probabilities = self.softmax(self.fc3(x))
        return action_probabilities

    def evaluate(self, state, epsilon=1e-7):
        action_probabilities = self.forward(state)
        dist = Categorical(action_probabilities)
        action = dist.sample().to(state.device)

        z = action_probabilities == 0.0
        z = z.float()*epsilon
        log_action_probs = torch.log(action_probabilities+z)
        return action.detach().cpu(), action_probabilities, log_action_probs

    def get_action(self, state):
        dist = Categorical(self.forward(state))
        action = dist.sample().to(state.device)
        return action.detach().cpu()

    def greedy_action(self,state, epsilon=1e-7):
        action_probabilities = self.forward(state)
        dist = Categorical(action_probabilities)
        greedy_actions = torch.argmax(action_probabilities, dim=1, keepdim=True)
        z = action_probabilities == 0.0
        z = z.float()*epsilon
        log_action_probs = torch.log(action_probabilities+z)
        return greedy_actions.detach().cpu(), action_probabilities, log_action_probs

