from numpy.random import beta
import torch as t
import numpy as np
import torch.nn as nn
from torch.nn.modules import linear
from torch.nn.modules.activation import Softmax
import torch.optim as optim
import time

class Actor(nn.Module):
    def __init__(self, n_features, n_actions, lr=0.001) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 128),
            nn.ReLU(),
            nn.Linear(128, len(n_actions)),
            nn.Softmax(),
        )

        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)
        self.action_prob_buffer = None

    def learn(self, td):
        log_probability = t.log(self.action_prob_buffer)
        exp_v = log_probability*td.detach()
        loss = -exp_v
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return exp_v

    def choose_action(self, s):
        s_unsqueeze = t.FloatTensor(s).unsqueeze(dim=0)
        prob_weights = self.net(s_unsqueeze)
        action = np.random.choice(
            range(prob_weights.detach().numpy().shape[1]), p=prob_weights.squeeze(dim=0).detach().numpy())
        self.action_prob_buffer = prob_weights[0][action]
        return action

    def save(self):
        t.save(self.net.state_dict(),
               "./models/AC/models/{}-{}-{} {}-{}-{}.pth".format(time.localtime()[0],
                                              time.localtime()[1],
                                              time.localtime()[2],
                                              time.localtime()[3],
                                              time.localtime()[4],
                                              time.localtime()[5], ))

    def load(self, path: str):
        model = t.load(path)
        self.net.load_state_dict(model)





class Critic(nn.Module):
    def __init__(self, n_features, lr=0.01, gamma=0.9) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 30),
            nn.ReLU(),
            nn.Linear(30, 1)
        )
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)
        self.GAMMA = gamma

    def learn(self, s, r, s_):
       
        s = t.FloatTensor(s)
        s_ = t.FloatTensor(s_)
        with t.no_grad():
            v_ = self.net(s_)
        td_error = t.mean((r+self.GAMMA*v_)-self.net(s))
        loss = td_error.square()
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return td_error