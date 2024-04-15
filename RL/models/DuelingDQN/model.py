import torch as t
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import time

class SumTree(object):
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.tree = np.zeros(2*capacity-1)
        self.data = np.zeros(capacity, dtype=object)
        self.data_pointer = 0

    def add(self, p, data):
        tree_idx = self.data_pointer+self.capacity-1  
        self.data[self.data_pointer] = data  
        self.update(tree_idx, p)
        self.data_pointer = self.data_pointer+1
        if self.data_pointer >= self.capacity:
            self.data_pointer = 0
  

    def update(self, tree_idx: int, p):
        change = p-self.tree[tree_idx]  
        self.tree[tree_idx] = p
      
        while tree_idx != 0:
            tree_idx = (tree_idx-1)//2
            self.tree[tree_idx] += change

    def get_leaf(self, v):
        parent_idx = 0
        while True:
            cl_idx = 2*parent_idx+1
            cr_idx = cl_idx+1
            if cl_idx >= len(self.tree):
                leaf_idx = parent_idx
                break
            # 
            else:
                if v <= self.tree[cl_idx]:
                    parent_idx = cl_idx
                else:
                    v -= self.tree[cl_idx]
                    parent_idx = cr_idx
        data_idx = leaf_idx-self.capacity+1 
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]
  

    @property
    def total_p(self):
        return self.tree[0]


class MEMORY_BUFFER_PER(object):
    epsilon = 0.01
    alpha = 0.6
    beta = 0.4
    beta_increment_per_sampling = 0.001
    abs_err_upper = 1.  

    def __init__(self, capacity: int) -> None:
        super().__init__()
        self.Sumtree = SumTree(capacity=capacity)

    def store(self, transition):
        max_p = np.max(self.Sumtree.tree[-self.Sumtree.capacity])  
        if max_p == 0:
            max_p = self.abs_err_upper
        self.Sumtree.add(max_p, transition)

    def sample(self, n):
        b_idx, b_memory, ISWeights = np.empty((n), dtype=np.int32), np.empty(
            (n, self.Sumtree.data[0].size)), np.empty((n, 1))
     
        pri_seg = self.Sumtree.total_p/n
        self.beta = np.min(
            [1., self.beta+self.beta_increment_per_sampling])  
        min_prob = np.min(
            self.Sumtree.tree[-self.Sumtree.capacity:])/self.Sumtree.total_p
        for i in range(n):
            a, b = pri_seg*i, pri_seg*(i+1)
            v = np.random.uniform(a, b)
            idx, p, data = self.Sumtree.get_leaf(v)
            prob = p/self.Sumtree.total_p
            ISWeights[i, 0] = np.power(prob/min_prob, -self.beta)
            b_idx[i], b_memory[i, :] = idx, data
        return b_idx, b_memory, ISWeights

    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.epsilon  
        clipped_errors = np.minimum(abs_errors, self.abs_err_upper)
        ps = np.power(clipped_errors, self.alpha)
        for ti, p in zip(tree_idx, ps):
            self.Sumtree.update(ti, p)


class _DQN(nn.Module):
    def __init__(self, n_states: int, n_actions: int, hidden_layers: int):
        super().__init__()
        self.input_size = n_states
        self.output_size = n_actions
        self.hidden_layers = hidden_layers

        self.feature_layer = nn.Linear(self.input_size, self.hidden_layers)
        self.value_layer = nn.Linear(self.hidden_layers, 1)  
        self.advantage_layer = nn.Linear(
            self.hidden_layers, self.output_size)  
    def forward(self, x: t.Tensor) -> t.Tensor:
        feature1 = F.relu(self.feature_layer(x))
        feature2 = F.relu(self.feature_layer(x))
        value = self.value_layer(feature1)
        advantage = self.advantage_layer(feature2)
        return value+advantage-advantage.mean(dim=1, keepdim=True)


class DuelingDQN(nn.Module):
    def __init__(self,
                 n_states: int,
                 n_actions: int,
                 hidden_layers: int,
                 lr=0.001,
                 memory_size=100000,
                 target_replace_iter=100,
                 batch_size=32,
                 reward_decay=0.9,
                 e_greedy=0.9) -> None:
        super().__init__()
        self.n_actions = n_actions
        self.n_states = n_states
        self.train_net = _DQN(n_states, len(n_actions), hidden_layers)
        self.target_net = _DQN(n_states, len(n_actions), hidden_layers)
        self.target_net.load_state_dict(self.train_net.state_dict())
        self.optimizer = optim.RMSprop(
            self.train_net.parameters(), lr=lr, eps=0.001, alpha=0.95)
        self.memory = MEMORY_BUFFER_PER(memory_size)
        self.learn_step_counter = 0
        self.epsilon = e_greedy
        self.target_replace_iter = target_replace_iter
        self.batch_size = batch_size
        self.gamma = reward_decay
        # for storing memory
        self.memory_counter = 0

    def choose_action(self, x):
        x = t.FloatTensor(x).unsqueeze(dim=0)
        if np.random.uniform() < self.epsilon:
            actions_value = self.target_net(x)
            action = t.argmax(actions_value).data.numpy()
        else:
            action = np.random.randint(0, len(self.n_actions))
        return action

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        self.memory.store(transition)
        self.memory_counter += 1

    def learn(self):
        # target parameter update
        if self.learn_step_counter % self.target_replace_iter == 0:
            self.target_net.load_state_dict(self.train_net.state_dict())
        self.learn_step_counter += 1

        tree_idx, batch_memory, ISWeights = self.memory.sample(self.batch_size)
        b_s = t.FloatTensor(batch_memory[:, :self.n_states])
        b_a = t.LongTensor(
            batch_memory[:, self.n_states:self.n_states+1].astype(int))
        b_r = t.FloatTensor(batch_memory[:, self.n_states+1:self.n_states+2])
        b_s_ = t.FloatTensor(batch_memory[:, -self.n_states:])
        q_eval = self.train_net(b_s).gather(1, b_a)
        q_next = self.target_net(b_s_).detach()



        q_target = b_r+self.gamma*q_next.max(1)[0].unsqueeze(1)
        loss = (q_eval-q_target).pow(2)*t.FloatTensor(ISWeights)
        prios = loss.data.numpy()
        loss = loss.mean()
        self.memory.batch_update(tree_idx, prios)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


    def save(self):
        t.save(self.train_net.state_dict(),
               "./models/DuelingDQN/models/{}-{}-{} {}-{}-{}.pth".format(time.localtime()[0],
                                              time.localtime()[1],
                                              time.localtime()[2],
                                              time.localtime()[3],
                                              time.localtime()[4],
                                              time.localtime()[5], ))

    def load(self, path: str):
        model = t.load(path)
        self.train_net.load_state_dict(model)
        self.target_net.load_state_dict(model)
