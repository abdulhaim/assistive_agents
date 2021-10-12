import torch
import numpy as np
from misc.utils import combined_shape


class ReplayBufferPhaseI(object):
    """
    A simple FIFO experience replay buffer for SAC agents.
    """
    def __init__(self, obs_dim, act_dim, size):
        self.obs_buf = np.zeros((size, obs_dim[2], obs_dim[0], obs_dim[1]), dtype=np.float32)
        self.assistant_action_buf = np.zeros(combined_shape(size, act_dim), dtype=np.float32)
        self.assistant_reward_buf = np.zeros(size, dtype=np.float32)
        self.next_obs_buf = np.zeros((size, obs_dim[2], obs_dim[0], obs_dim[1]), dtype=np.float32)
        self.assistant_next_action_buf = np.zeros(combined_shape(size, act_dim), dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, action, reward, next_obs, next_action, discount, done):
        self.obs_buf[self.ptr] = obs
        self.assistant_action_buf[self.ptr] = action
        self.assistant_reward_buf[self.ptr] = reward
        self.next_obs_buf[self.ptr] = next_obs
        self.assistant_next_action_buf[self.ptr] = next_action
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(
            obs=self.obs_buf[idxs],
            assistant_action=self.assistant_action_buf[idxs],
            assistant_reward=self.assistant_reward_buf[idxs],
            next_obs=self.next_obs_buf[idxs],
            assistant_next_action=self.assistant_next_action_buf[idxs],
            done=self.done_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in batch.items()}


class ReplayBufferPhaseII(object):
    """
    A simple FIFO experience replay buffer for SAC agents.
    """
    def __init__(self, obs_dim, act_dim, size):
        self.obs_buf = np.zeros((size, obs_dim[0], obs_dim[1], obs_dim[2]), dtype=np.float32)
        self.assistant_action_buf = np.zeros(combined_shape(size, act_dim), dtype=np.float32)
        self.human_action_buf = np.zeros(combined_shape(size, act_dim), dtype=np.float32)
        self.assistant_reward_buf = np.zeros(size, dtype=np.float32)
        self.next_obs_buf = np.zeros((size, obs_dim[0], obs_dim[1], obs_dim[2]), dtype=np.float32)
        self.assistant_next_action_buf = np.zeros(combined_shape(size, act_dim), dtype=np.float32)
        self.human_next_action_buf = np.zeros(combined_shape(size, act_dim), dtype=np.float32)
        self.discount_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, action, reward, next_obs, next_action, done):
        self.obs_buf[self.ptr] = obs
        self.action_buf[self.ptr] = action
        self.reward_buf[self.ptr] = reward
        self.next_obs_buf[self.ptr] = next_obs
        self.next_action_buf[self.ptr] = next_action
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(
            obs=self.obs_buf[idxs],
            action=self.action_buf[idxs],
            reward=self.reward_buf[idxs],
            next_obs=self.next_obs_buf[idxs],
            next_action=self.next_action_buf[idxs],
            done=self.done_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in batch.items()}

