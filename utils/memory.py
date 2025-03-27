import numpy as np
import torch


class ReplayBuffer(object):
    def __init__(
        self, lidar_state_dim, position_state_dim, action_dim, max_size, device
    ):
        self.max_size = int(max_size)
        self.ptr = 0
        self.size = 0

        self.lidar_state = np.zeros((self.max_size, lidar_state_dim))
        self.position_state = np.zeros((self.max_size, position_state_dim))
        self.action = np.zeros((self.max_size, action_dim))
        self.next_lidar_state = np.zeros((self.max_size, lidar_state_dim))
        self.next_position_state = np.zeros((self.max_size, position_state_dim))
        self.reward = np.zeros((self.max_size, 1))
        self.not_done = np.zeros((self.max_size, 1))

        self.device = torch.device(device)

    def add(
        self, lidar_state, position_state, action, next_lidar_state, next_position_state, 
              reward, done):
        self.lidar_state[self.ptr] = lidar_state
        self.position_state[self.ptr] = position_state
        self.action[self.ptr] = action
        self.next_lidar_state[self.ptr] = next_lidar_state
        self.next_position_state[self.ptr] = next_position_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size=96):
        # TODO: Clean this up. There's probably a cleaner way to seperate
        # on-policy and off-policy sampling. Clean up extra-dimension indexing
        # also
        ind = np.random.randint(0, self.size, size=int(batch_size))

        l_s = torch.FloatTensor(
            self.lidar_state[ind][:, None, :]).to(self.device)
        p_s = torch.FloatTensor(
            self.position_state[ind][:, None, :]).to(self.device)
        a = torch.FloatTensor(
            self.action[ind][:, None, :]).to(self.device)
        n_l_s = torch.FloatTensor(
            self.next_lidar_state[ind][:, None, :]).to(self.device)
        n_p_s = torch.FloatTensor(
            self.next_position_state[ind][:, None, :]).to(self.device)
        r = torch.FloatTensor(
            self.reward[ind][:, None, :]).to(self.device)
        d = torch.FloatTensor(
            self.not_done[ind][:, None, :]).to(self.device)

        return l_s, p_s, a, n_l_s, n_p_s, r, d

    def on_policy_sample(self):
        ind = np.arange(0, self.size)

        s = torch.FloatTensor(
            self.state[ind][:, None, :]).to(self.device)
        a = torch.FloatTensor(
            self.action[ind][:, None, :]).to(self.device)
        ns = torch.FloatTensor(
            self.next_state[ind][:, None, :]).to(self.device)

        # reward and dones don't need to be "batched"
        r = torch.FloatTensor(
            self.reward[ind]).to(self.device)
        d = torch.FloatTensor(
            self.not_done[ind]).to(self.device)

        return s, a, ns, r, d

    def _ff_sampling(self, ind):
        # FF only need Batch size * Input size, on_policy or not

        s = torch.FloatTensor(self.state[ind]).to(self.device)
        a = torch.FloatTensor(self.action[ind]).to(self.device)
        ns = torch.FloatTensor(self.next_state[ind]).to(self.device)
        r = torch.FloatTensor(self.reward[ind]).to(self.device)
        d = torch.FloatTensor(self.not_done[ind]).to(self.device)

        return s, a, ns, r, d

    def clear_memory(self):
        self.ptr = 0
        self.size = 0
