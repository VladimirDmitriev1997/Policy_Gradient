###
# REINFORCE agent for arbitrary environment
# CV with net for regression of a
###
import math
from scipy.integrate import quad
from scipy import special
import sys 
import os
import torch  
import gym
import numpy as np  
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch.nn.utils as utils

pi = Variable(torch.FloatTensor([math.pi])).cuda()

class REINFORCE_Agent(nn.Module):
    def __init__(self, num_states, sampler_parameters_dim, hidden_size, learning_rate=3e-4):
        super(REINFORCE_Agent, self).__init__()

        self.linear1 = nn.Linear(num_states, hidden_size)
        self.linear2 = nn.Linear(hidden_size, 1)
        self.linear2_ = nn.Linear(hidden_size, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        
    
    

    
    
    
    def forward(self, inputs):
        x = inputs
        x = F.relu(self.linear1(x))
        mu = self.linear2(x)
        sigma_sq = self.linear2_(x)
        sigma_sq = F.softplus(sigma_sq)
        

        return torch.transpose(torch.stack([mu, sigma_sq]),0,1)

        return output

    
    def get_action(self, state):
        ### Take state in numpy
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.forward(Variable(state))
        
        highest_prob_action, rands = self.distribution_model(probs)
        
        highest_prob_action = torch.tensor(rands, dtype = torch.float32) * torch.sqrt(probs[:,1]**2) +  probs[:,0]
        log_prob = torch.log(torch.exp(-((highest_prob_action-probs[:,0])**2)/(2*probs[:,1]**2))*torch.sqrt(2*np.pi*probs[:,1]**2))
        return highest_prob_action, log_prob, rands, probs, None
    
    def distribution_model(self, probs):
        #This function set distribution of decisions
        rands = np.random.normal(size = probs.shape[0])
        highest_prob_action = torch.tensor(rands, dtype = torch.float32) * torch.sqrt(probs[:,1]**2) +  probs[:,0]

        return highest_prob_action, rands

    def update_policy(self, rewards, log_probs, state_dim, action_dim, trajectory_len, gamma, states=None, actions=None, params=None, rands=None, cv_constructor=None):
        
        discounted_rewards = []
        l = []
        
        if cv_constructor != None:
            if cv_constructor.status == 'learning':
                rewards = cv_constructor.learn_regression(1, actions, states, rewards)
                return 0
            if cv_constructor.status == 'work':
                rewards = cv_constructor.get_cv_correction(states, actions, rewards)
        
        for t in range(len(rewards)):
            Gt = 0 
            pw = 0
            for r in rewards[t:]:
                Gt = Gt + gamma**pw * r
                pw = pw + 1
            discounted_rewards.append(Gt)
        
        discounted_rewards = torch.tensor(discounted_rewards)
        #discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9) # normalize discounted rewards

        policy_gradient = []
        for log_prob, Gt in zip(log_probs, discounted_rewards):
            policy_gradient.append(-log_prob * Gt)
    
        self.optimizer.zero_grad()
        policy_gradient = torch.stack(policy_gradient).mean() # sum()
        policy_gradient.backward()
        utils.clip_grad_norm(self.parameters(), 40)
        self.optimizer.step()
        #return rewards

        
    def parser(self, extra_):
        return 0


class A2C_Agent(nn.Module):
    def __init__(self, num_states, sampler_parameters_dim, hidden_size, learning_rate=3e-4):
        super(A2C_Agent, self).__init__()



        

        self.values = []
        self.entropies = []



        self.linear1 = nn.Linear(num_states, hidden_size).cuda()
        self.linear2 = nn.Linear(hidden_size, 1).cuda()
        self.linear2_ = nn.Linear(hidden_size, 1).cuda()
        self.linear3 = nn.Linear(hidden_size, hidden_size).cuda()
        self.linear4_v = nn.Linear(num_states, hidden_size).cuda()
        self.linear5 = nn.Linear(hidden_size, hidden_size).cuda()
        self.linear_v5 = nn.Linear(hidden_size, 1).cuda()
        

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, inputs):
        x = inputs.cuda()
        y = inputs.cuda()
        x = F.relu(self.linear1(x))
        mu = self.linear2(x)
        sigma_sq= self.linear2_(x)
        sigma_sq = F.softplus(sigma_sq)
        
        y = F.relu(self.linear4_v(y))
        value= self.linear_v5(y)
        return value, torch.transpose(torch.stack([mu, sigma_sq]),0,1)
    

    def get_action(self, state):
        ### Take state in numpy
        state = torch.from_numpy(state).float().unsqueeze(0)
        value, probs = self.forward(Variable(state).cuda())
        
        mu = probs[:,0]
        sigma_sq = probs[:,1]
        highest_prob_action, rands = self.distribution_model(probs)
        self.entropies.append(-0.5*((sigma_sq+2*pi.expand_as(sigma_sq)).log()+1))
        self.values.append(value)
        highest_prob_action = torch.tensor(rands, dtype = torch.float32).cuda() * torch.sqrt(probs[:,1]**2) +  probs[:,0]
        log_prob = torch.log(torch.exp(-((highest_prob_action-probs[:,0])**2)/(2*probs[:,1]**2))*torch.sqrt(2*np.pi*probs[:,1]**2))
        return highest_prob_action, log_prob, rands, probs, None
    
    def distribution_model(self, probs):
        #This function set distribution of decisions
        rands = np.random.normal(size = probs.shape[0])
        highest_prob_action = torch.tensor(rands, dtype = torch.float32).cuda() * torch.sqrt(probs[:,1]**2) +  probs[:,0]

        return highest_prob_action, rands

    def update_policy(self, rewards, log_probs, state_dim, action_dim, trajectory_len, gamma, states=None, actions=None, params=None, rands=None, cv_constructor=None):
        
        p_loss = 0
        v_loss = 0
        deltas = []
        if cv_constructor != None:
            if cv_constructor.status == 'learning':
                rewards = cv_constructor.learn_regression(1, actions, states, rewards)
                return 0
            if cv_constructor.status == 'work':
                rewards = cv_constructor.get_cv_correction(states, actions, rewards)
        for i in range(len(rewards)):
            if i!=len(rewards)-1: 
                deltas.append((gamma**i)*(rewards[i] + gamma*self.values[i+1].detach() - self.values[i].detach()).reshape(-1))
            else:
                deltas.append((gamma**i)*(rewards[i] - self.values[i].detach()).reshape(-1))
        
        
        
        

     
        for i in range(len(rewards)):
            p_loss = p_loss - self.values[i]*(Variable(torch.tensor(deltas[i], dtype = torch.float32)).cuda())
            v_loss = v_loss - (log_probs[i]*(Variable(torch.tensor(deltas[i], dtype = torch.float32))).cuda()) - (0.0001*self.entropies[i].cuda()).sum()
    
        loss = (v_loss + p_loss)/ len(rewards)
    
        
        self.optimizer.zero_grad()
        loss.backward()
        utils.clip_grad_norm(self.parameters(), 40)
        self.optimizer.step()
        self.entropies = []
        self.values = []

        
    def parser(self, extra_):

        return 0

class MDP():
    def __init__(self, env, agent, trajectory_len, gamma, state_dim, action_dim, CV=None):
        super(MDP, self).__init__()

        self.env = env
        self.agent = agent
        self.CV = CV
        self.states_history = []
        self.actions_history = []
        self.trajectory_len = trajectory_len # maximal trajectory length
        self.t = 0 # current step
        self.done = False # has env reach the final step
        self.log_probs = []
        self.rewards = []
        self.params = []
        self.rands = []
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.state = torch.Tensor([env.reset()]).cpu().detach().numpy().reshape(-1)
        
        
    def step(self):
        if (self.done) or (self.t>=self.trajectory_len):
            return 0
        action, log_prob, rand, param, extra_  = self.agent.get_action(self.state) #entropy, rand, param, value
        action = action.cpu().detach().numpy()
        self.actions_history.append(action)
        if self.CV != None:
            self.agent.parser(extra_)
            self.CV.get_H(self.t,rand)
        next_state, reward, done, _ = self.env.step(action.reshape((1,-1))[0])
        self.state = next_state
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        state = torch.Tensor([next_state])

        self.params.append(param)
        self.states_history.append(state.cpu().detach().numpy().reshape(-1))
        self.rands.append(rand)
        
        self.t=self.t+1
        self.done = done
        return 1
    
    def finalize_trajectory(self):

        self.states_history = np.array(self.states_history)
        self.params = np.array(self.params)
        self.actions_history = np.array(self.actions_history)
        self.rands = np.array(self.rands)

        
        self.agent.update_policy(self.rewards, self.log_probs, self.state_dim, self.action_dim, self.t, self.gamma, self.states_history, self.actions_history, self.params, self.rands, self.CV)# entropies,values,epoches_regr, Nets, trajectory_len,H,state_dim, action_dim,K, flag, 0.9999)
        if CV == None:
            return np.sum(self.rewards), np.mean(self.rewards)
        else:
            return np.sum(self.rewards), np.mean(self.rewards)#, np.sum(c_rewards), np.mean(c_rewards)


def herm(k, l, xi):
  return special.hermitenorm(k)(xi)/np.sqrt(special.gamma(k+1))

class Approx_net(nn.Module):
    def __init__(self, input_dim):
        super(Approx_net, self).__init__()
        self.linear1 = nn.Linear(input_dim, 128)
        torch.nn.init.normal_(self.linear1.weight,mean = 0.0, std = 0.1) #0.01
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(128, 128)
        torch.nn.init.normal_(self.linear2.weight,mean = 0.0, std = 0.1)
        self.relu2 = nn.ReLU()
        self.linear21 = nn.Linear(128, 128)
        torch.nn.init.normal_(self.linear21.weight,mean = 0.0, std = 0.1)
        self.relu21 = nn.ReLU()
        self.linear22 = nn.Linear(128, 128)
        torch.nn.init.normal_(self.linear22.weight,mean = 0.0, std = 0.1)
        self.relu22 = nn.ReLU()
        self.linear3 = nn.Linear(128, 1)
        torch.nn.init.normal_(self.linear3.weight,mean = 0.0, std = 1)
        

    def forward(self, x):
        x = self.linear22(self.relu21(self.linear21(self.relu2(self.linear2(self.relu1(self.linear1(x)))))))
        x = self.linear3(self.relu22(x))
  
        return x



class CV():
    def __init__(self, lag, K, burn_in, burn_off, polynomial, trajectory_len, state_dim, action_dim, status, C_1, C_2, lr = 0.001):
        super(CV, self).__init__()
        
        self.polynomial = polynomial
        self.K = K
        self.lag = lag
        self.burn_in = burn_in
        self.burn_off = burn_off
        self.lr = lr
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_trajectory_len = trajectory_len
        

        self.c_2 = C_2
        self.c_1 = C_1
        self.status = status

        
        self.H = np.zeros((self.max_trajectory_len, self.K))

        self.trajectory_len = 0
        self.Nets = []
        self.archive = None
    
    def init_regression(self, states_num, actions_num):
        for i in range(self.lag):
            Nets_ = []
            for j in range(self.K):
                net = Approx_net(states_num+actions_num).cuda()
                Nets_.append([net, nn.MSELoss(), optim.Adam(net.parameters(), lr=self.lr)])
            self.Nets.append(Nets_)
    
    def construct_Q(self, actions, states, y):
        ### get actions, states and target value to rearrange them in convenient way
        actions = actions.reshape((-1,self.action_dim))
        states = states.reshape((-1,self.state_dim))
        y = np.array(y).reshape((-1,1))
        q = np.hstack((actions,states,y))
        q = torch.from_numpy(q)
        Q = torch.tensor(q,dtype = torch.float32, requires_grad=True).cuda()
        return Q
    
    
  
        
    def get_H(self, t, rand):
        for k in range(1, self.K+1):
            self.H[t,k-1] = self.polynomial(k,t,rand)
        self.trajectory_len = self.trajectory_len + 1
    
    
        

    def a_net(self, x, q, k, Nets):
        # propagation of single state-action pair to get regression
        x_ = x
        output = Nets[q][k][0](x_)
        return output
    
    
    
    
    def learn_regression(self, n_epoches, actions, states, y):
        
        Q = self.construct_Q(actions, states, y)
        for epoch in range(n_epoches):
            for t in range(self.trajectory_len):
                c = 0
                for q in range(max(t-self.lag+1,0),t+1):
                    for k in range(1,self.K+1):
                        x = torch.tensor(Q[t-q,:self.action_dim+ self.state_dim].detach().clone(),dtype = torch.float32, requires_grad=True).detach().cpu()
                        x = torch.tensor(x,dtype = torch.float32, requires_grad=True).reshape((1,-1)).cuda()
                        b_2 =  self.a_net(x, t-q, k-1, self.Nets)*self.H[q,k-1]
                        if c == 0:
                            c = 1
                            CV_ = b_2
                        else:
                            CV_ = torch.cat((b_2, CV_))
                CV1 = torch.sum(CV_, dim = 0, keepdim = True)
                
                if t == 0:
                    Deltas_shifted = Q[t,-1].detach().clone() - CV1#torch.tensor((Q[t,0]-CV1),dtype = torch.float32, requires_grad=True).reshape((1,-1)).cuda()
                else:
                    Deltas_shifted = torch.cat((Q[t,-1].detach().clone() - CV1, Deltas_shifted)).cuda()
   

          
            loss = self.Loss(Deltas_shifted, Q[:,-1].detach().clone())
            loss.backward()
            #Losses.append([C_1.numpy(),C_2.numpy()])
            for i in range(self.lag):
                for j in range(self.K):
                    self.Nets[i][j][2].step()
                    self.Nets[i][j][2].zero_grad()
    
    
    
    def get_cv_correction(self, states, actions, y):
        n_epoches = 1
        Q = self.construct_Q(actions, states, y)
        for epoch in range(n_epoches):
            for t in range(self.trajectory_len):
                c = 0
                for q in range(max(t-self.lag+1,0),t+1):
                    for k in range(1,self.K+1):
                        x = torch.tensor(Q[t-q,:self.action_dim+ self.state_dim].detach().clone(),dtype = torch.float32, requires_grad=True).detach().cpu()
                        x = torch.tensor(x,dtype = torch.float32, requires_grad=True).reshape((1,-1)).cuda()
                        b_2 =  self.a_net(x, t-q, k-1, self.Nets)*self.H[q,k-1]
                        if c == 0:
                            c = 1
                            CV_ = b_2
                        else:
                            CV_ = torch.cat((b_2, CV_))
                CV1 = torch.sum(CV_, dim = 0, keepdim = True)
                if t <= self.burn_in:
                    if t == 0:
                        Deltas_shifted = Q[t,-1].detach().clone().reshape((1,1))
                    if t > 0:
                        Deltas_shifted = torch.cat((Q[t,-1].detach().clone().reshape((1,1)), Deltas_shifted)).cuda()
                if t > self.burn_in and t < self.trajectory_len - self.burn_off:
                    Deltas_shifted = torch.cat((Q[t,-1].detach().clone() - CV1.detach(), Deltas_shifted)).cuda()
                if t >= self.trajectory_len - self.burn_off:
                    Deltas_shifted = torch.cat((Q[t,-1].detach().clone().reshape((1,1)), Deltas_shifted)).cuda()
        self.archive = Deltas_shifted.reshape((-1,1))
        return  Deltas_shifted.reshape((-1,1))
    
    def clean_cv(self, status):
        
        self.status = status

        
        self.H = np.zeros((self.max_trajectory_len, self.K))

        self.trajectory_len = 0
        
        self.archive = None
    
    
    
    def Loss(self, R, ref):
        # Loss for control variate
        c_1 = self.c_1
        c_2 = self.c_2
        cv = (R - ref)
        L =  c_1*torch.std(R) + c_2*torch.sqrt(torch.mean(torch.mul(cv,cv)))
        return L