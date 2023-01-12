from collections import deque, namedtuple
import random
import torch
import numpy as np



class ReplayBuffer:
    def __init__(self, config, device) -> None:
        
        self.memory = deque(maxlen=config.buffer_size)
        self.batch_size = config.batch_size
        self.device = device
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done", "entropy"])
    
    def add(self, state, action, reward, next_state, done, entropy):
        e = self.experience(state, action, reward, next_state, done, entropy)
        self.memory.append(e)

    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.stack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.stack([e.next_state for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)
        entropy = torch.from_numpy(np.vstack([e.entropy for e in experiences if e is not None])).float().to(self.device)

    def __len__(self):
        return len(self.memory)

    @property
    def size(self):
        return len(self.memory)


class PrioritizedReplay:
    def __init__(self, config, device) -> None:
        self.memory = deque(maxlen=config.buffer_size)
        self.batch_size = config.batch_size
        self.priorities = deque(maxlen=self.buffer_size)
        self.pos = 0
        self.frame = 0
        self.device = device
        self.capacity = config.buffer_size
        self.beta_start, self.beta_frames = config.beta_annealing # [ß start, ß frames]

    def beta_annealing(self, frame_idx):
        # has to reach 1 at the end of training only
        return min(1.0, self.beta_start +  (1. - self.beta_start)* frame_idx/ self.beta_frames)

    def add(self, state, action, reward, next_state, done, entropy):
        assert state.ndim == next_state.ndim # verifies the dimentions are the same
        state = np.expand_dims(state,0) # expands dimention of state [s]-> [[s]]
        next_state = np.expand_dims(next_state, 0)

        max_prio = max(self.priorities) if self.memory else 1.0

        self.pos = (self.pos + 1) % self.capacity

        if len(self.memory) < self.capacity:
            self.memory.extend([(state, action, reward, next_state, done)])
            self.priorities.extend([max_prio])
            return

        self.memory[self.pos] = (state, action, reward, next_state, done)
        self.priorities[self.pos] = max_prio
        
    def sample(self, ck=0):
        N = len(self.buffer)

        if ck == 0:
            ck = self.pos
        if ck > N:
            ck = N

        # splits the list of priorities to match the c_k size
        if N == self.capacity:
            prios = np.array(self.priorities)
        else:
            prios = np.array(list(self.priorities)[:ck]) 

        #print(prios.shape)
        # compute the probability and normalize it
        #print("prios: ",prios)
        #print("priorities: ",len(self.priorities))
        p = prios** self.alpha
        P = p/p.sum()
        #print("P: ",P)
        if len(P.shape)>1:
            P = P.sum(axis=1)

        #print(np.array(P).shape)

        # choose c_k indices using probability p
        #print("a: ",N)
        #print("p: ", P)
        indices = np.random.choice(N, self.batch_size, p=P)
        samples = np.take(self.buffer, indices)
        
        # update beta for the annealing
        beta = self.beta_by_frame(self.frame)
        self.frame += 1

        # Importance Sampling Weights
        weights = (N * P[indices]) ** (-beta)
        weights = np.array(weights/weights.max(), dtype=np.float32)

        states, actions, rewards, next_states, dones, entropy = zip(*samples)
        #print(actions[0])
        states      = torch.FloatTensor(np.float32(np.concatenate(states))).to(self.device)
        next_states = torch.FloatTensor(np.float32(np.concatenate(next_states))).to(self.device)
        actions     =torch.FloatTensor(np.float32(actions)).to(self.device)#.unsqueeze(1)
        rewards     = torch.FloatTensor(rewards).to(self.device).unsqueeze(1) 
        dones       = torch.FloatTensor(dones).to(self.device).unsqueeze(1)
        weights    = torch.FloatTensor(weights).unsqueeze(1)
        entropy = torch.FloatTensor(entropy).unsqueeze(1)
        
        return states, actions, rewards, next_states, dones, entropy, indices, weights

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = abs(prio)

    def __len__(self):
        return len(self.memory)

    @property
    def size(self):
        return len(self.memory)