import torch as t

from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
lr = 1e-5
device = t.device("cuda:0" if t.cuda.is_available() else "cpu")
batch_size = 256
buffer_capacity = 60000
OBSERVE = 10000
ACTIONS = len(COMPLEX_MOVEMENT)
epsilon = 0.1
