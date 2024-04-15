
import torch                                    
import torch.nn as nn                           
import torch.nn.functional as F                
import numpy as np                              
import gym                                    
import time




class Net(nn.Module):
    def __init__(self,n_states,n_actions):
      
        super(Net, self).__init__()

        self.fc1 = nn.Linear(n_states, 50)
        self.fc1.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(50, len(n_actions))
        self.out.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        actions_value = self.out(x)
        return actions_value 


class DQN(object):
    def __init__(self,n_states,n_actions,lr,epsilon,gamma,target_replace_iter,memory_capacity,batch_size):
        self.eval_net, self.target_net = Net(n_states,n_actions), Net(n_states,n_actions)
        self.learn_step_counter = 0
        self.memory_counter = 0
        self.n_states=n_states
        self.n_actions=n_actions
        self.lr=lr
        self.epsilon=epsilon
        self.gamma=gamma
        self.target_replace_iter=target_replace_iter
        self.memory_capacity=memory_capacity
        self.batch_size=batch_size
        self.memory = np.zeros((self.memory_capacity, self.n_states * 2 + 2))
        self.optimizer = torch.optim.Adam(
            self.eval_net.parameters(), lr=self.lr)    
        self.loss_func = nn.MSELoss()

    def choose_action(self, x):
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        if np.random.uniform() < self.epsilon:
            actions_value = self.eval_net.forward(x)
            action = torch.max(actions_value, 1)[1].data.numpy()
            action = action[0]
        else:                                                                  
            action = np.random.randint(0, len(self.n_actions))
        return action

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        index = self.memory_counter % self.memory_capacity
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        if self.learn_step_counter % self.target_replace_iter == 0:                  
            self.target_net.load_state_dict(
                self.eval_net.state_dict())         
        self.learn_step_counter += 1                                            

 
        sample_index = np.random.choice(self.memory_capacity,self.batch_size)
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :self.n_states])
        b_a = torch.LongTensor(b_memory[:, self.n_states:self.n_states+1].astype(int))
        b_r = torch.FloatTensor(b_memory[:, self.n_states+1:self.n_states+2])
        b_s_ = torch.FloatTensor(b_memory[:, -self.n_states:])

        q_eval = self.eval_net(b_s).gather(1, b_a)
        q_next = self.target_net(b_s_).detach()
        q_target = b_r + self.gamma * q_next.max(1)[0].view(self.batch_size, 1)
        loss = self.loss_func(q_eval, q_target)
        self.optimizer.zero_grad()                                      
        loss.backward()
        self.optimizer.step()                                           
        return loss

    def save(self):

        torch.save(self.eval_net.state_dict(),
               "./models/DQN/models/{}-{}-{} {}-{}-{}.pth".format(time.localtime()[0],
                                              time.localtime()[1],
                                              time.localtime()[2],
                                              time.localtime()[3],
                                              time.localtime()[4],
                                              time.localtime()[5], ))

    def load(self, path: str):
        model = torch.load(path)
        self.eval_net.load_state_dict(model)
        self.target_net.load_state_dict(model)

