from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import memory
import time
import cv2
import numpy as np
from utils import *

env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT)
D = memory.Memory(buffer_capacity)


done = True
<<<<<<< HEAD
for step in range(5000):
    if done:
        state = env.reset()
    state, reward, done, info = env.step(env.action_space.sample()) #state (240 256 3)
    time.sleep(0.01)
    env.render()
=======
state = env.reset()
x_t, reward, done, info = env.step(env.action_space.sample())
x_t = cv2.cvtColor(cv2.resize(x_t, (80, 80)), cv2.COLOR_BGR2GRAY)
ret, x_t = cv2.threshold(x_t, 1, 255, cv2.THRESH_BINARY)
s_t = np.stack((x_t, x_t, x_t, x_t), axis=0)
s_t = np.ascontiguousarray(s_t, dtype=np.float32)
env.render()
time.sleep(0.01)
while True:
    if done:
        state = env.reset()
    a_t = online_net.get_action(s_t)
    x_t1,reward, done, info = env.step(env.action_space.sample())
    x_t1 = cv2.cvtColor(cv2.resize(x_t1, (80, 80)), cv2.COLOR_BGR2GRAY)
    ret, x_t1 = cv2.threshold(x_t1, 1, 255, cv2.THRESH_BINARY)
    x_t1 = np.reshape(x_t1, (1, 80, 80))
    # s_t1 = np.append(x_t1, s_t[:,:,1:], axis = 2)
    s_t1 = np.append(x_t1, s_t[:3, :, :], axis=0)
    s_t1 = np.ascontiguousarray(s_t1, dtype=np.float32)
    D.push(s_t, s_t1, a_t, reward, not done)


>>>>>>> 0404496074adb7cf734c79bb869c7458ac672942

env.close()