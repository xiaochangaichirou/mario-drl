from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros

import random
import os
import net
from utils import *
import memory
import time
import cv2
import numpy as np
import gc

env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
env = JoypadSpace(env, COMPLEX_MOVEMENT)
done = True


D = memory.Memory(buffer_capacity)
online_net = net.DQN().to(device)
optimizer = t.optim.Adam(online_net.parameters(), lr = lr)
state = env.reset()

if os.path.isfile("saved_network/checkpointdict.npy"):
    checkpointdict = np.load("saved_network/checkpointdict.npy", allow_pickle=True).item()
    checkpoint = t.load(checkpointdict["checkpoint_path"])
    online_net.load_state_dict(checkpoint['state_dict'])
    print("Successfully loaded:", "saved_network/checkpoint.pth.tar")
    del checkpoint
    gc.collect()
else:
    print("Could not find old network weights")

action = env.action_space.sample()
x_t, reward, done, info = env.step(action)
x_t = cv2.cvtColor(cv2.resize(x_t, (80, 80)), cv2.COLOR_BGR2GRAY)
ret, x_t = cv2.threshold(x_t, 1, 255, cv2.THRESH_BINARY)
s_t = np.stack((x_t, x_t, x_t, x_t), axis=0)
s_t = np.ascontiguousarray(s_t, dtype=np.float32)
time.sleep(0.01)
turn = 0
while True:
    if done or turn % 2000 == 0:
        state = env.reset()
    time.sleep(0.01)
    s_t_tensor = t.Tensor(s_t).unsqueeze(0).to(device)
    if random.random() < epsilon:
        a_t = env.action_space.sample()
    else:
        a_t = online_net.get_action(s_t_tensor)
        a_t = a_t.item()
    x_t1,reward, done, info = env.step(a_t)
    x_t1 = cv2.cvtColor(cv2.resize(x_t1, (80, 80)), cv2.COLOR_BGR2GRAY)
    ret, x_t1 = cv2.threshold(x_t1, 1, 255, cv2.THRESH_BINARY)
    x_t1 = np.reshape(x_t1, (1, 80, 80))
    # s_t1 = np.append(x_t1, s_t[:,:,1:], axis = 2)
    s_t1 = np.append(x_t1, s_t[:3, :, :], axis=0)
    s_t1 = np.ascontiguousarray(s_t1, dtype=np.float32)
    a_t_np = np.zeros(ACTIONS)
    a_t_np[a_t] = 1
    D.push(s_t, s_t1, a_t_np, reward, not done)
    if len(D) > buffer_capacity:
        D.pop()
    if turn > OBSERVE:
        minibatch = D.sample(batch_size)
        loss = net.DQN.train_model(online_net, optimizer, minibatch)
    turn = turn + 1
    if turn % 1000 == 0:
        print(turn)
    if turn % 20000 == 0:
        times = time.gmtime()
        save_path = 'saved_network/' +  'mario-dqn' + f'{times.tm_year}-{times.tm_mon}-{times.tm_mday}-{times.tm_hour}-{times.tm_min}' + ".pth.tar"
        np.save("saved_network/checkpointdict.npy", {'checkpoint_path': save_path})
        t.save({
            'turn': turn,
            'state_dict': online_net.state_dict(),
            'optimizer': optimizer.state_dict()
        }, save_path)

env.close()