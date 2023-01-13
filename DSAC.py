import torch
import torch.nn as nn
import torch.optim as optim
import torch.functional as F
from torch.nn.utils import clip_grad_norm_
from networks import Policy, Critic
import copy
from replay_buffers import ReplayBuffer, PrioritizedReplay
import os


class DSACAgent(nn.Module):
    def __init__(self, state_size, action_size, device, params) -> None:
        super().__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.device = device
        self.gama = params.gamma
        self.tau = params.tau
        self.beta = params.beta
        hidden_size = params.hidden_size
        learning_rate = params.learning_rate
        self.clip_grad_param = params.clip_grad_param

        self.target_entropy = - action_size

        self.log_alpha = torch.tensor([0.0], requires_grad=True)
        self.alpha = self.log_alpha.exp().detach()
        self.alpha_optimizer = optim.Adam(params=[self.log_alpha], lr=learning_rate)


        self.actor_local = Policy(state_size, action_size, hidden_size).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=learning_rate)

        self.critic1 = Critic(state_size, action_size, hidden_size, 1)
        self.critic2 = Critic(state_size, action_size, hidden_size, 2)
        
        assert self.critic1.parameters() != self.critic2.parameters()

        self.critic1_target = Critic(state_size, action_size, hidden_size).to(device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=learning_rate)
        
        self.critic2_target = Critic(state_size, action_size, hidden_size).to(device)
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        self.critic2_optimizer = optim.Adam(self.critic1.parameters(), lr=learning_rate)

        memory_type = {"Replay":ReplayBuffer, "PER":PrioritizedReplay}
        self.memory = memory_type[params.memory_type](params, self.device)

    
    def beta_scheduling(self):
        self.i += 1
        beta = min(self.beta_end, self.beta_start+ self.i * self.beta_step) 
        return beta 

    def collect_random(self, env, num_samples=200):
        state = env.reset()
        for _ in range(num_samples):
            action = env.action_space.sample()
            next_state, reward, done, _ = env.step(action)
            self.memory.add(state, action, reward, next_state, done)
            state = next_state
            if done:
                state = env.reset()

    def add_exp(self,state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)

    def get_action(self,state):
        state = torch.from_numpy(state).float().to(self.device)

        with torch.no_grad():
            action, action_probs, log_action_probs = self.actor_local.get_action(state)
            entropy  = - torch.sum(action_probs * log_action_probs).cpu().detach()

        return action.numpy(), entropy.numpy()

    def calc_policy_loss(self, states, alpha, old_entropy):
      _, action_probs, log_pis = self.actor_local.evaluate(states)
      beta = self.beta
      q1 = self.critic1(states)   
      q2 = self.critic2(states)
      min_Q = (q1+q2)/2
      log_action_pi = torch.sum(log_pis * action_probs, dim=1)
      entropy = - log_action_pi
      entropy_penalty = 1/2 * beta * torch.pow(old_entropy-entropy, 2)

      entropy_penalty_mean = entropy_penalty.mean()
      
      actor_loss = ((action_probs * (2 * alpha * log_pis - min_Q )).sum(1)+ entropy_penalty_mean).mean()
      
      return actor_loss, log_action_pi, entropy.sum()
    
    def learn(self):
        states, actions, rewards, next_states, dones, old_entropy = self.memory.sample()
        # update actor
        current_alpha = copy.deepcopy(self.alpha)
        actor_loss, log_pis, entropy = self.calc_policy_loss(states, current_alpha.to(self.device), old_entropy)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        # alpha loss
        alpha_loss = - (self.log_alpha.exp() * (log_pis.cpu() + self.target_entropy).detach().cpu()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp().detach()
        # update critic
        with torch.no_grad():
            _, action_probs, log_pis = self.actor_local.evaluate(next_states)
            Q_target1_next = self.critic1_target(next_states)
            Q_target2_next = self.critic2_target(next_states)
            Q_n = (Q_target1_next + Q_target2_next)/2
            Q_target_next = action_probs * (Q_n - self.alpha.to(self.device) * log_pis)

            # Compute Q targets for current states (y_i)
            Q_targets = rewards + (self.gamma * (1 - dones) * Q_target_next.sum(dim=1).unsqueeze(-1))

        # compute critic loss
        q1 = self.critic1(states).gather(1,actions.long())
        q2 = self.critic2(states).gather(1, actions.long())
        
        critic1_loss = 0.5 * F.mse_loss(q1, Q_targets)
        critic2_loss = 0.5 * F.mse_loss(q2, Q_targets)

        #Update critics
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward(retain_graph=True)
        clip_grad_norm_(self.critic1.parameters(), self.clip_grad_param)
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        clip_grad_norm_(self.critic2.parameters(), self.clip_grad_param)
        self.critic2_optimizer.step()

        self.soft_update(self.critic1, self.critic1_target)
        self.soft_update(self.critic2, self.critic2_target)
        
        return actor_loss.item(), alpha_loss.item(), critic1_loss.item(), critic2_loss.item(), current_alpha

    def soft_update(self,local_model, target_model):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)

    def save(self, config, save_name, wandb, ep=None):
        
        save_dir = './trained_models/'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            return
        if not ep == None:
            torch.save(self.actor_local.state_dict() , save_dir + config.run_name + save_name + str(ep) + ".pth")
            wandb.save(save_dir + config.run_name + save_name + str(ep) + ".pth")
            return
        torch.save(self.actor_local.state_dict(), save_dir + config.run_name + save_name + ".pth")
        wandb.save(save_dir + config.run_name + save_name + ".pth")
        
