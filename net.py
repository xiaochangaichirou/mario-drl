import torch as t
from torch import nn
import torch
import torch.nn.functional as F
import numpy as np
from utils import *


class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4, padding=2)
        self.pool1 = nn.MaxPool2d(2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(1600, 2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, 256)
        self.head = nn.Linear(256, ACTIONS)
    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.head(x)

    @classmethod
    def train_model(cls, online_net, optimizer, batch):
        states = t.tensor(np.array(batch.state), dtype=torch.float32).to(device)
        next_states = t.tensor(np.array(batch.next_state),dtype=torch.float32).to(device)
        actions = t.Tensor(np.array(batch.action)).to(device)
        rewards = t.Tensor(np.array(batch.reward)).to(device)
        masks = t.Tensor(np.array(batch.mask)).to(device)
        pred = online_net(states).squeeze(1)
        next_pred = online_net(next_states).squeeze(1)

        pred = torch.sum(pred.mul(actions), dim=1)
        b = next_pred.max(1)[0]
        target = rewards + masks * 0.99 * b

        loss = F.mse_loss(pred, target.detach())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss
    def get_action(self, input):
        qvalue = self.forward(input)
        return qvalue.max(1)[1].view(1, 1)[0,0]
