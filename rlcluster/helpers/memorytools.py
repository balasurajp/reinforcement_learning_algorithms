import math, time, random, torch
import numpy as np
from scipy.signal import lfilter
from collections import namedtuple, deque
from torch.multiprocessing import Process, Queue
from rlcluster.helpers.torchtools import device, FLOAT
eps = 1e-8

def Discounted_Cumulative_Summation(x, discount):
    """
    input: 
        x = [x0, x1, x2]
    output:
        y = [x0 + discount * x1 + discount^2 * x2,  x1 + discount * x2, x2]
    """
    return lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

class TrajectoriesBuffer:
    """
    A buffer for storing trajectories experienced by agent interacting with the environment. 
    - uses Generalized Advantage Estimation (GAE-Lambda) for calculating the advantages of state-action pairs.
    """

    def __init__(self, dim_state, dim_action, max_size, gamma=0.99, gaelambda=0.95):
        self.obs_buf = np.zeros((max_size,) + dim_state, dtype=np.float32)
        self.act_buf = np.zeros((max_size,) + dim_action, dtype=np.float32)
        self.logp_buf = np.zeros((max_size,), dtype=np.float32)
        self.rew_buf = np.zeros((max_size,), dtype=np.float32)
        self.val_buf = np.zeros((max_size,), dtype=np.float32)
        self.ret_buf = np.zeros((max_size,), dtype=np.float32)
        self.adv_buf = np.zeros((max_size,), dtype=np.float32)

        self.gamma, self.gaelambda = gamma, gaelambda
        self.ptr, self.path_start_idx, self.max_size = 0, 0, max_size

    def store(self, state, action, log_prob, reward, value):
        """
        Append one timestep of agent-environment interaction to the buffer - (s, a, ap, r, v).
        """
        assert self.ptr < self.max_size
        self.obs_buf[self.ptr] = state
        self.act_buf[self.ptr] = action
        self.logp_buf[self.ptr] = log_prob
        self.rew_buf[self.ptr] = reward
        self.val_buf[self.ptr] = value
        self.ptr += 1

    def finish_trajectory(self, last_value=0):
        """
        - At trajectory end / trajectory chop, uses rewards and value estimates from the trajectory to compute gae based advantage estimates and
        the rewards-to-go for each state, to be the targets for the value function.
        
        - The "last_value = 0" argument should be 0 if the trajectory ended and otherwise "last_value = V(s_T)", the value function estimate for the last state
        to allow bootstraping the reward-to-go calculation for accounting timesteps beyond the arbitrary episode horizon.
        """
        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_value)
        vals = np.append(self.val_buf[path_slice], last_value)
        
        # Implementing GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = Discounted_Cumulative_Summation(deltas, self.gamma * self.gaelambda)
        
        # Computes rewards-to-go --> targets for the value function
        self.ret_buf[path_slice] = Discounted_Cumulative_Summation(rews, self.gamma)[:-1]
        
        # Last New Trajectory pointer in buffer
        self.path_start_idx = self.ptr

    def get_data(self):
        """
        At the end of an epoch, call this method to get the data from the buffer, with advantages appropriately normalized, and reset the buffer.
        """
        assert self.ptr == self.max_size
        self.ptr, self.path_start_idx = 0, 0

        self.adv_buf = (self.adv_buf - self.adv_buf.mean()) / (self.adv_buf.std()+eps)
        data = dict(states=self.obs_buf, actions=self.act_buf, log_probs=self.logp_buf, returns=self.ret_buf, advantages=self.adv_buf)
        return {k: FLOAT(v).to(device) for k,v in data.items()}


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for offpolicy agents.
    """

    def __init__(self, dim_state, dim_action, size):
        self.state_buf = np.zeros((size,) + dim_state, dtype=np.float32)
        self.act_buf = np.zeros((size,) + dim_action, dtype=np.float32)
        self.rew_buf = np.zeros((size,), dtype=np.float32)
        self.nextstate_buf = np.zeros((size,) + dim_state, dtype=np.float32)
        self.done_buf = np.zeros((size,), dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, state, action, reward, next_state, done):
        """
        Append one timestep of agent-environment interaction to the buffer - (s,a,r,s,d).
        """
        self.state_buf[self.ptr] = state
        self.act_buf[self.ptr] = action
        self.rew_buf[self.ptr] = reward
        self.nextstate_buf[self.ptr] = next_state
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size):
        # assert self.size>batch_size, 'Batch size is more than number of samples in buffer'
        # idxchoices = np.arange(0, self.size, dtype=int)
        # idxs = np.random.choice(idxchoices, batch_size, replace=False) 
        idxs = np.random.randint(0, self.size, size=batch_size) 
        batch = dict(states=self.state_buf[idxs],
                     actions=self.act_buf[idxs],
                     rewards=self.rew_buf[idxs],
                     next_states=self.nextstate_buf[idxs],
                     dones=1.0-self.done_buf[idxs]) # important trick to make episode ending masks to zeros and vice versa
        return {k: FLOAT(v).to(device) for k,v in batch.items()}

