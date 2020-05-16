import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pdb
import time
import argparse
import sys
from matplotlib import pyplot as plt
import matplotlib as mpl
import os
mpl.use('Agg')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Net(nn.Module):

    def __init__(self,
                lr=0.001):
        super(Net,self).__init__()
        
        self.fc_block = nn.Sequential(
                                    nn.Linear(784, 512)
                                    nn.ReLU()
                                    nn.Linear(512, 256)
                                    nn.ReLU()
                                    nn.Linear(256,64)
                                    nn.ReLU()
                                    nn.Linear(64,10)
                                    nn.ReLU()
                                    )

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.parameters(),
                                    lr = lr)

    def forward(self, x):
        out = self.fc_block(x)
        return out

def train_net(net, inputs, num_epochs):
    epoch_loss_arr = []
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        net = net.train()
        total_samples = 0
        for i, data in enumerate(trainloader):
            inputs, labels = data[0].to(device), data[1].to(device)
            total_samples + = inputs.shape[0]
            net.optimizer.zero_grad()
            out = net(inputs)
            loss = net.criterion(out, labels)
            loss.backward()
            self.optimizer.step()
            
            epoch_loss += loss.item()

        print("epoch_loss: {}".format(epoch_loss / total_samples))
        epoch_loss_arr.append(epoch_loss / total_samples)




