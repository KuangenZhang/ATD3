import numpy as np
import pandas as pd
import math
import torch
import matplotlib.pyplot as plt
from scipy.spatial import distance
from scipy import signal
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

# Code based on:
# https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py
# Expects tuples of (state, next_state, action, reward, done)
class ReplayBuffer(object):
    '''
    Change the buffer to array and delete for loop.
    '''
    def __init__(self, max_size=1e6):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0

    def get(self, idx):
        return self.storage[idx]

    def add(self, data):
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = data
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(data)

    def add_final_reward(self, final_reward, steps, delay=0):
        len_buffer = len(self.storage)
        for i in range(len_buffer - steps - delay, len_buffer - delay):
            item = list(self.storage[i])
            item[3] += final_reward
            self.storage[i] = tuple(item)

    def add_specific_reward(self, reward_vec, idx_vec):
        for i in range(len(idx_vec)):
            time_step_num = int(idx_vec[i])
            item = list(self.storage[time_step_num])
            item[3] += reward_vec[i]
            self.storage[time_step_num] = tuple(item)

    def sample_on_policy(self, batch_size, option_buffer_size):
        return self.sample_from_storage(batch_size, self.storage[-option_buffer_size:])

    def sample(self, batch_size):
        return self.sample_from_storage(batch_size, self.storage)

    @staticmethod
    def sample_from_storage(batch_size, storage):
        ind = np.random.randint(0, len(storage), size=batch_size)
        x, y, u, r, d, p = [], [], [], [], [], []
        for i in ind:
            X, Y, U, R, D, P = storage[i]
            x.append(np.array(X, copy=False))
            y.append(np.array(Y, copy=False))
            u.append(np.array(U, copy=False))
            r.append(np.array(R, copy=False))
            d.append(np.array(D, copy=False))
            p.append(np.array(P, copy=False))
        return np.array(x), np.array(y), np.array(u), np.array(r).reshape(-1, 1), \
               np.array(d).reshape(-1, 1), np.array(p).reshape(-1, 1)


# Expects tuples of (state, next_state, action, reward, done)
class ReplayBufferMat(object):
    '''
    Change the buffer to array and delete for loop.
    '''
    def __init__(self, max_size=1e6):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0
        self.data_size = 0

    def add(self, data):
        data = list(data)
        if 0 == len(self.storage):
            for item in data:
                self.storage.append(np.asarray(item).reshape((1, -1)))
        else:
            if self.storage[0].shape[0] < int(self.max_size):
                for i in range(len(data)):
                    self.storage[i] = np.r_[self.storage[i], np.asarray(data[i]).reshape((1, -1))]
            else:
                for i in range(len(data)):
                    self.storage[i][int(self.ptr)] = np.asarray(data[i]).reshape((1, -1))
                self.ptr = (self.ptr + 1) % self.max_size
        self.data_size = len(self.storage[0])

    def sample_on_policy(self, batch_size, option_buffer_size):
        return self.sample_from_storage(
            batch_size, start_idx = self.storage[0].shape[0] - option_buffer_size)

    def sample(self, batch_size):
        return self.sample_from_storage(batch_size)

    def sample_from_storage(self, batch_size, start_idx = 0):
        buffer_len = self.storage[0].shape[0]
        ind = np.random.randint(start_idx, buffer_len, size=batch_size)
        data_list = []
        # if buffer_len > 9998:
        #     print(buffer_len, ind)
        for i in range(len(self.storage)):
            # if buffer_len > 9998:
            #     print('{},shape:{}'.format(i, self.storage[i].shape))
            data_list.append(self.storage[i][ind])
        return tuple(data_list)

    def add_final_reward(self, final_reward, steps):
        self.storage[3][-steps:] += final_reward


def fifo_data(data_mat, data):
    data_mat[:-1] = data_mat[1:]
    data_mat[-1] = data
    return data_mat


def softmax(x):
    # This function is different from the Eq. 17, but it does not matter because
    # both the nominator and denominator are divided by the same value.
    # Equation 17: pi(o|s) = ext(Q^pi - max(Q^pi))/sum(ext(Q^pi - max(Q^pi))
    x_max = np.max(x, axis=-1, keepdims=True)
    e_x = np.exp(x - x_max)
    e_x_sum = np.sum(e_x, axis=-1, keepdims=True)
    out = e_x / e_x_sum
    return out


def write_table(file_name, data):
    df = pd.DataFrame(data)
    df.to_excel(file_name + '.xls', index=False)



