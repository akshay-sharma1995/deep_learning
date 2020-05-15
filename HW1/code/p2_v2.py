import torch
import torchvision
import torchvision.transforms as transforms
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
import numpy as np
mpl.use('Agg')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                transforms.RandomVerticalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]
                                )


trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                                  shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                               download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                                 shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
                   'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


class Net(nn.Module):

    def __init__(self,
                lr = 0.001,
                momentum=0.9):
        
        super(Net,self).__init__()

        self.conv_block = nn.Sequential(
                                        nn.Conv2d(3,64,3,1,1),
                                        nn.BatchNorm2d(64),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(64,64,3,1,1),
                                        nn.BatchNorm2d(64),
                                        nn.ReLU(inplace=True),
                                        nn.MaxPool2d(kernel_size=2,stride=2),
                                        nn.Conv2d(64,128,3,1,1),
                                        nn.BatchNorm2d(128),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(128,128,3,1,1),
                                        nn.BatchNorm2d(128),
                                        nn.ReLU(inplace=True),
                                        nn.MaxPool2d(kernel_size=2,stride=2),
                                        nn.Conv2d(128,256,3,1,1),
                                        nn.BatchNorm2d(256),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(256,256,3,1,1),
                                        nn.BatchNorm2d(256),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(256,256,3,1,1),
                                        nn.BatchNorm2d(256),
                                        nn.ReLU(inplace=True),
                                        nn.MaxPool2d(kernel_size=2,stride=2),
                                        nn.Conv2d(256,512,3,1,1),
                                        nn.BatchNorm2d(512),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(512,512,3,1,1),
                                        nn.BatchNorm2d(512),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(512,512,3,1,1),
                                        nn.BatchNorm2d(512),
                                        nn.ReLU(inplace=True),
                                        nn.MaxPool2d(kernel_size=2,stride=2),
                                        nn.Conv2d(512,512,3,1,1),
                                        nn.BatchNorm2d(512),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(512,512,3,1,1),
                                        nn.BatchNorm2d(512),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(512,512,3,1,1),
                                        nn.BatchNorm2d(512),
                                        nn.ReLU(inplace=True),
                                        nn.MaxPool2d(kernel_size=2,stride=2),
                                        )

        self.fc_block = nn.Sequential(
                                        nn.Linear(512,256),
                                        nn.ReLU(),
                                        nn.Linear(256,64),
                                        nn.ReLU(),
                                        nn.Linear(64,10),
                                    )

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.parameters(), 
                                    lr=lr, 
                                    momentum=momentum)

    def forward(self, x):
        
        x = self.conv_block(x)
        x = x.view(-1, 512)
        x = self.fc_block(x)
        
        return x

    def save_model(self, path):
        torch.save(self.state_dict(),path)

    def load_model(self, path):
        self.load_state_dict(torch.load(path))



def train_network(net,num_epochs,path):

    epoch_loss_arr = []
    test_accuracy_arr = []
    report_loss_iter = 2000
    for epoch in range(num_epochs):  # loop over the dataset multiple times
        
        running_loss = 0.0
        # pdb.set_trace()
        iterator = iter(trainloader)
        net = net.train()
        epoch_loss = 0.0
        for i, data in enumerate(trainloader, 0):
        # for i, data in enumerate(trial_dataloader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)
            
            # zero the parameter gradients
            net.optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(inputs)
            loss = net.criterion(outputs, labels)
            loss.backward()
            net.optimizer.step()

            # print statistics
            running_loss += loss.item()
            epoch_loss = running_loss*1.0
            # if i % 2000 == 1999:    # print every 2000 mini-batches
            if i%report_loss_iter==(report_loss_iter-1):
                print('[%d, %5d] loss: %.3f' %
                (epoch + 1, i + 1, running_loss / report_loss_iter))
                running_loss = 0.0
        epoch_loss_arr.append(epoch_loss/report_loss_iter)
        epoch_loss = 0.0
        if((epoch+1)%1==0):
            net.save_model(path)
            test_accuracy_arr.append(test_network(net))
            net = net.train()
            np.save("train_loss_arr",epoch_loss_arr)
            np.save("test_accuracy_arr",test_accuracy_arr)
    print('Finished Training')
    return epoch_loss_arr, test_accuracy_arr

def test_network(net):
    correct = 0
    total = 0
    net = net.eval()
    with torch.no_grad():
        for data in testloader:
        # for data in trial_dataloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data,1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print("Accuracy on 10000 test images: {} %%".format(100*correct/total))
    return correct/total

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--lr', dest='lr', type=float, default=1e-3, help="learning rate")
    parser.add_argument('--momentum', dest='momentum', type=float, default=0.9, help="learning rate")
    parser.add_argument('--epochs', dest='epochs',type=int, default=50,help="learning_rate")
    return parser.parse_args()

def plot_props(data,prop_name,plot_save_path):
    fig = plt.figure(figsize=(16,9))
    plt.plot(data)
    plt.ylabel(prop_name)
    plt.savefig(os.path.join(plot_save_path,prop_name+".jpg"))
    plt.close()

def main(args):
    args = parse_args()
    lr = args.lr
    num_epochs = args.epochs
    momentum = args.momentum
    path = "./cifar_net_lr_{}_momentum_{}.pth".format(lr,momentum)
    plot_path = './'
    print("device: {}".format(device))
    net = Net(lr,momentum)
    net.to(device)
    start_time = time.time()
    train_loss_arr, test_accuracy_arr = train_network(net,num_epochs,path)
    plot_props(train_loss_arr, "train_loss_lr_{}_momentum_{}".format(lr,momentum), plot_path)
    plot_props(test_accuracy_arr, "test_acc_lr_{}_momentum_{}".format(lr,momentum), plot_path)

    print("Time taken train for {} epochs: {}".format(num_epochs, time.time()-start_time))
    test_acc = test_network(net)


if __name__=="__main__":
    main(sys.argv)
