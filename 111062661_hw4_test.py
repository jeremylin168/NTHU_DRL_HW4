import gym
import numpy as np
import torch


class Agent(object):
    """Agent that acts randomly."""
    def __init__(self):
        self.action_space = gym.spaces.Box(low=0, high=1, shape=(22,), dtype=np.float32)

    def act(self, observation):
        return self.action_space.sample()
# Environment
from osim.env import L2M2019Env


# pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

#
import numpy as np
import random

class Agent_train():
    def __init__(self):

        self.device = torch.device("cpu")

        self.epslion_m = 0.999

        # basic parameters
        self.action_num = 22

        self.learning_rate = 3e-4
        self.gamma = 0.99
        self.tau = 0.01

        #self.alpha = 1/self.action_num
        #self.alpha = torch.autograd.Variable(torch.tensor(1.0, dtype=torch.float32), requires_grad=True).to(self.device)
        #self.alpha.parameters().requires_grad = True
        #self.h = -self.action_num # - dim(action)
        #self.alpha_optimizer = torch.optim.Adam([self.alpha], lr=self.learning_rate)

        #self.alpha = torch.exp(self.log_alpha)

        #self.alpha_ = torch.clamp(self.alpha, min=0)
        #self.alpha_1 = torch.clamp(self.alpha, min=0) 

        self.batch_size = 256

        self.skip_frame = 2
        self.skip_frame_counter = 2
        self.action_buffer = 0

        self.update_iter = 1
        self.update_counter = 0


    

        # Networks
        self.actor = Actor(learning_rate=self.learning_rate).to(self.device)
        #self.critic_1 = Critic(learning_rate=1e-3).to(self.device)
        #self.critic_2 = Critic(learning_rate=1e-3).to(self.device)

        #self.target_critic_1 = Critic(learning_rate=self.learning_rate).to(self.device)
        #self.target_critic_2 = Critic(learning_rate=self.learning_rate).to(self.device)

    

        #self.update_target_network(tau=1)

        # memory
        #self.memory_counter = 0
        #self.memory_limit = 64 * 64 * 8
        #self.state_size = 339

        #self.memory = np.zeros((self.memory_limit, self.state_size * 2 * 2 + self.action_num + 2)) # two state frame & 22 action & 1 reward & 1 done
        
        #state buffer
        self.state_buffer = np.zeros((2, 339))
        self.train_state_buffer = np.zeros((2, 339))
        self.next_state_buffer = np.zeros((2, 339))


        self.load_weight()

    def show_sample(self):
        #n_state = self.process_obs(observation)
        x = torch.tensor(np.reshape(self.state_buffer.ravel(), (1, self.state_size*2)), dtype=torch.float32).to(self.device)
        out_action, log_probs, mu, sigma = self.actor.forward(x, reparameterize=False)
        print(out_action)
        print(log_probs)
        print(mu)
        print(sigma)
        print(self.alpha)
        #print(self.log_alpha)

    def act(self, observation, rand_out = False, eps=0):

        self.skip_frame_counter += 1
        if self.skip_frame_counter >= self.skip_frame:
            # store state to state buffer

            n_state = self.process_obs(observation)
            self.state_buffer[1:, :] = self.state_buffer[:-1, :]
            self.state_buffer[0, :] = n_state

            # get action from Actor
            x = torch.tensor(np.reshape(self.state_buffer.ravel(), (1, self.state_size*2)), dtype=torch.float32).to(self.device)
            out_action, log_probs, mu, sigma = self.actor.forward(x, reparameterize=False)
            #print(out_action)
            #print(mu)
            #print(sigma)

            # epsilon greedy
            if rand_out and np.random.uniform() < self.epslion_m ** eps :
                out_action = np.random.uniform(-1, 1, self.action_num)
                self.action_buffer = out_action
            else:

                # store actions to action buffer
                self.action_buffer = out_action.cpu().detach().numpy()[0]
            

            #print(self.action_buffer)
            self.new_action = (self.action_buffer + 1)/2

            action = self.new_action
            self.skip_frame_counter = 0

        else:
            action = self.new_action

        return action

    def process_obs(self, observation):
        arr = np.zeros((339))
        arr[:242] = [a for c in observation['v_tgt_field'] for b in c for a in b]
        num = 242
        for i in observation.keys():
            if i is not 'v_tgt_field':
                for j in observation[i].keys():
                    if type(observation[i][j]) is dict:
                        for k in observation[i][j].keys():
                            arr[num] = observation[i][j][k]
                            num += 1
                    elif type(observation[i][j]) is list:
                        for k in observation[i][j]:
                            arr[num] = k
                            num += 1
                    else:
                        arr[num] = observation[i][j]
                        num += 1
        return arr
        
    """
    def store_memory(self, observation, action, reward, next_observation, done):

        obs  = self.process_obs(observation)
        next_obs = self.process_obs(next_observation)

        self.train_state_buffer[1:, :] = self.train_state_buffer[:-1, :]
        self.train_state_buffer[0, :] = obs

        self.next_state_buffer[1:, :] = self.next_state_buffer[:-1, :]
        self.next_state_buffer[0, :] = next_obs

        #self.train_state_buffer = np.zeros((2, 339))
        #self.next_state_buffer = np.zeros((2, 339))
        
        r_action = action* 2 -1


        new_record = np.hstack((self.train_state_buffer.ravel(), r_action, reward, self.next_state_buffer.ravel(), done))

        index = self.memory_counter % self.memory_limit
        self.memory[index, :] = new_record
        #print(np.reshape(self.memory[index, :self.state_size], (1, 4, 96, 96)).astype(int))

        self.memory_counter +=1
        
        if self.memory_counter < self.memory_limit:
            for i in range(7):
                index = self.memory_counter % self.memory_limit
                self.memory[index, :] = new_record
                #print(np.reshape(self.memory[index, :self.state_size], (1, 4, 96, 96)).astype(int))
                self.memory_counter +=1
        
        if self.memory_counter == self.memory_limit:
            print("train start")
        if self.memory_counter >= self.memory_limit * 2:
            self.memory_counter = self.memory_limit
            


    def update_network(self):
        if self.memory_counter < self.memory_limit:
            return
        
        # depatch memory
        sample_index = np.random.choice(self.memory_limit, self.batch_size)
        sampled_memory = self.memory[sample_index, :]

        states = torch.FloatTensor(np.reshape(sampled_memory[:, :self.state_size*2], (self.batch_size, self.state_size*2))).to(self.device)
        actions = torch.FloatTensor((sampled_memory[:, self.state_size*2:self.state_size*2 + self.action_num])).to(self.device)
        rewards = torch.FloatTensor(sampled_memory[:, self.state_size*2 + self.action_num:self.state_size*2 + self.action_num + 1]).to(self.device)
        next_states = torch.FloatTensor(np.reshape(sampled_memory[:, self.state_size*2 + self.action_num + 1:self.state_size*4 + self.action_num + 1], (self.batch_size, self.state_size*2))).to(self.device)
        dones = torch.FloatTensor(sampled_memory[:, -1]).to(self.device)

        #print(rewards)
        # update critic network
        actions_, log_probs_, mu, sigma = self.actor.forward(next_states, reparameterize=False)

        t_q1 = self.target_critic_1.forward(next_states, actions_)
        t_q2 = self.target_critic_2.forward(next_states, actions_)

        q1 = self.critic_1.forward(states, actions)
        q2 = self.critic_2.forward(states, actions)

        t_q_min = torch.min(t_q1, t_q2).view(-1)

        b_eq = rewards.view(-1) + self.gamma * (1-dones) * ( t_q_min - self.alpha * log_probs_.view(-1))

        critic_1_loss = F.mse_loss(q1.view(-1), b_eq.detach())
        critic_2_loss = F.mse_loss(q2.view(-1), b_eq.detach())
        #critic_loss = torch.add(critic_1_loss, critic_2_loss)
        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()
        critic_1_loss.backward()
        critic_2_loss.backward()
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()



        # update actor network
        repar_actions_, repar_log_probs, mu, sigma = self.actor.forward(states, reparameterize=True)

        q1 = self.critic_1.forward(states, repar_actions_)
        q2 = self.critic_2.forward(states, repar_actions_)

        q_min = torch.min(q1, q2).view(-1)
        actor_loss = -torch.mean(q_min - self.alpha * repar_log_probs.view(-1)) 
        #print(actor_loss).detach()


        for param in self.critic_1.parameters():
            param.requires_grad = False
        for param in self.critic_2.parameters():
            param.requires_grad = False

        self.actor.optimizer.zero_grad()
        actor_loss.backward()
        self.actor.optimizer.step()

        for param in self.critic_1.parameters():
            param.requires_grad = True
        for param in self.critic_2.parameters():
            param.requires_grad = True


        #actions_a, log_probs_a, mu, sigma = self.actor.forward(states, reparameterize=False)
        alpha_loss = torch.mean(-self.alpha*(repar_log_probs + self.h).detach())
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        #self.alpha = torch.exp(self.log_alpha)



        #self.alpha_1 = torch.clamp(self.alpha, min=0) 

        

        #self.alpha = self.alpha - self.learning_rate*torch.mean(-self.alpha*log_probs_a.detach() - self.alpha* self.h)

        #print(self.alpha)

        self.update_counter += 1
        if self.update_counter >= self.update_iter:
            self.update_target_network(self.tau)
            self.update_counter = 0

    def update_target_network(self, tau):
        for param, target_param in zip(self.critic_1.parameters(), self.target_critic_1.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

        for param, target_param in zip(self.critic_2.parameters(), self.target_critic_2.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

        
        critic_1_dict = dict(self.critic_1.named_parameters())
        target_critic_1_dict = dict(self.target_critic_1.named_parameters())
        for item in critic_1_dict:
            critic_1_dict[item] = tau * critic_1_dict[item].clone() + (1 - tau) * target_critic_1_dict[item].clone()

        critic_2_dict = dict(self.critic_2.named_parameters())
        target_critic_2_dict = dict(self.target_critic_2.named_parameters())
        for item in critic_2_dict:
            critic_2_dict[item] = tau * critic_2_dict[item].clone() + (1 - tau) * target_critic_2_dict[item].clone()

        #self.target_actor.load_state_dict(actor_dict)
        self.target_critic_1.load_state_dict(critic_1_dict)
        self.target_critic_2.load_state_dict(critic_2_dict)
        


    def save_weight(self, path = "./models_3/test_model_weight", ep="none"):
        with open(f"{path}_ep{ep}_alpha.txt", "w") as f:
            f.write(f"{self.alpha}")
        self.actor.save_weight(path=f"{path}_ep{ep}_act")
        self.critic_1.save_weight(path=f"{path}_ep{ep}_c1")
        self.critic_2.save_weight(path=f"{path}_ep{ep}_c2")
    """
    def load_weight(self, path="./models_3/test_model_weight", ep="none"):
        #with open(f"{path}_ep{ep}_alpha.txt", "r") as f:
            #self.alpha = torch.autograd.Variable(torch.tensor(float(f.read()), dtype=torch.float32), requires_grad=True).to(self.device)
            #self.alpha = torch.autograd.Variable(torch.tensor(float(f.read())+0.5, dtype=torch.float32), requires_grad=True).to(self.device) #
            #self.alpha_optimizer = torch.optim.Adam([self.alpha], lr=self.learning_rate)
            #self.alpha.parameters().requires_grad = True
            #self.alpha = torch.exp(self.log_alpha)
            #self.alpha_1 = torch.clamp(self.alpha, min=0) 
        #print(self.alpha)
        self.actor.load_weight(path=f"111062892_hw4_data")
        #self.critic_1.load_weight(path=f"111062892_hw4_data")
        #self.critic_2.load_weight(path=f"111062892_hw4_data")


class Actor(nn.Module):
    def __init__(self, learning_rate=1e-4):
        super(Actor, self).__init__()

        self.L1 = nn.Linear(339*2, 512)
        self.L2 = nn.Linear(512, 512)
        self.mu = nn.Linear(512, 22)
        self.sigma = nn.Linear(512, 22)

        # optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, input, reparameterize=True):
        x = F.relu(self.L1(input))
        x = F.relu(self.L2(x))
        mu = self.mu(x)
        #sigma = F.sigmoid(self.sigma(x)) * 0.05
        sigma = torch.clamp(self.sigma(x), min=-10, max=2)

        #print(mu)
        #print(sigma)

        probs = Normal(mu, torch.exp(sigma))
        if reparameterize:
            # reparameterization
            actions = probs.rsample()
        else:
            actions = probs.sample()
            #actions = mu
        action = torch.tanh(actions)
        log_probs = probs.log_prob(actions)
        log_probs -= torch.log(1 - action.pow(2)+ 1e-6)
        log_probs = log_probs.sum(1, keepdim=True)

        return action, log_probs, mu, sigma


    def save_weight(self, path):
        torch.save(self.state_dict(), f"{path}")

    def load_weight(self, path):
        self.load_state_dict(torch.load(f"{path}"))

class Critic(nn.Module):
    def __init__(self, learning_rate=3e-4):
        super(Critic, self).__init__()
        # input 4 * 96 * 96
        self.L1 = nn.Linear(339*2 + 22, 512)
        self.L2 = nn.Linear(512, 256)
        self.q = nn.Linear(256, 1)

        # optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, input_obs, input_act):
        x = F.relu(self.L1(torch.cat((input_obs, input_act), 1)))
        x = F.relu(self.L2(x))
        q = self.q(x)

        return q
    def save_weight(self, path):
        torch.save(self.state_dict(), f"{path}")

    def load_weight(self, path):
        self.load_state_dict(torch.load(f"{path}"))
"""
if __name__ == "__main__":
    env = L2M2019Env(visualize=False, difficulty=2)
    observation = env.reset()


    episode_num = 50000

    agent = Agent_train()
    agent.act(observation)
    agent.save_weight()
    #agent.load_weight(ep="4399_") # 1149  2949_

    ave_reward = np.zeros(10)

    for eps in range(episode_num):
        # reset env
        obs = env.reset()
        #agent.show_sample()
        #env.render()
        done = False

        total_reward = 0
        frame_counter = 0
        sam = True

        while not done:
            r_reward = 0
            agent.skip_frame_counter = 2
            action = agent.act(obs, rand_out=False, eps=eps+4000)

            # skip 2 frame
            for i in range(agent.skip_frame):
                
                next_obs, reward, done, info = env.step(action)
                if sam:
                    agent.show_sample()
                    sam = False

                #env.render()

                total_reward += reward
                r_reward += reward
                
                if done:
                    break
            if done:
                agent.store_memory(obs, action, r_reward, next_obs, 1)
            else:
                agent.store_memory(obs, action, r_reward, next_obs, 0)
            

            frame_counter += 1
            if frame_counter >= 1:
                agent.update_network()
                frame_counter = 0
            
            #time_counter += 1
            #if time_counter > 2000 and rewards < 1500:
            #    done = True

            obs = next_obs

        if (eps+1) % 50 == 0:
            agent.save_weight(ep=eps)
        ave_reward[1:] = ave_reward[:-1]
        ave_reward[0] = total_reward
        print(f"Episode: {eps}, rewards: {total_reward}, ave reward: {np.mean(ave_reward)}")

    agent.save_weight()
    """