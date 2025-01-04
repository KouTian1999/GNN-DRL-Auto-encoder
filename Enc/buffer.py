import random
from collections import deque
import numpy as np


class ReplayBuffer():
    def __init__(self, buffer_size, batch_size):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.buffer = deque(maxlen=int(buffer_size))

    def put_sample(self, state, action, reward, next_state):
        # state [m_b, n_b]
        # action [m_b, n_b]
        # reward single scalar
        # next_state [m_b, n_b]
        self.buffer.append([state, action, reward, next_state])

    def get_batch(self):
        # state [batch_size, m_b, n_b]
        # action [batch_size, m_b, n_b]
        # reward [batch_size, 1]
        # next_state [batch_size, m_b, n_b]
        sample = random.sample(self.buffer, self.batch_size)
        states, actions, rewards, next_states = map(np.asarray, zip(*sample))
        rewards = np.expand_dims(rewards, axis=-1)
        return states, actions, rewards, next_states

    def current_size(self):
        return len(self.buffer)



