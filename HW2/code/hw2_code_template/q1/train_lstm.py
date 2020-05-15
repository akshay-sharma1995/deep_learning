import os
import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from matplotlib import pyplot as plt
from dataset import FlowDataset
from lstm import FlowLSTM
import pdb

def main():
    # check if cuda available
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # define dataset and dataloader
    train_dataset = FlowDataset(mode='train')
    test_dataset = FlowDataset(mode='test')
    train_loader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True, num_workers=4)
    test_loader = DataLoader(dataset=test_dataset, batch_size=16, shuffle=False, num_workers=4)

    # hyper-parameters
    num_epochs = 20
    lr = 0.001
    input_size = 17 # do not change input size
    hidden_size = 128
    num_layers = 2
    dropout = 0.1

    model = FlowLSTM(
        input_size=input_size, 
        hidden_size=hidden_size, 
        num_layers=num_layers, 
        dropout=dropout,
        learning_rate=lr
    ).to(device)

    # define your LSTM loss function here
    # loss_func = ?

    # define optimizer for lstm model
    # optim = Adam(model.parameters(), lr=lr)
    path = "./p1_model.ckpt"
    save_model_interval = 2 
    train_loss_arr = []
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0
        for n_batch, (in_batch, label) in enumerate(train_loader):
            num_batches += 1
            in_batch, label = in_batch.to(device), label.to(device)
            
            # train LSTM
            output_seq = model(in_batch) 


            # calculate LSTM loss
            # loss = loss_func(...)
            # loss = torch.dist(output_seq, label, 2)
            loss = torch.norm(output_seq-label, dim=2)
            loss = torch.mean(loss, -1)
            loss = torch.mean(loss, -1)

            model.optimizer.zero_grad()
            loss.backward()
            model.optimizer.step()
            
            epoch_loss += loss.item()
            # print loss while training

            if (n_batch + 1) % 200 == 0:
                print("Epoch: [{}/{}], Batch: {}, Loss: {}".format(
                    epoch, num_epochs, n_batch, loss.item()))

        train_loss_arr.append(epoch_loss / num_batches)
        if((epoch+1)%save_model_interval==0):
            save_model(model.state_dict(), path)
    plot_prop(train_loss_arr, "lstm_train_loss_vs_epoch")
    # test trained LSTM model
    l1_err, l2_err = 0, 0
    l1_loss = nn.L1Loss()
    l2_loss = nn.MSELoss()
    model.eval()
    with torch.no_grad():
        for n_batch, (in_batch, label) in enumerate(test_loader):
            in_batch, label = in_batch.to(device), label.to(device)
            pred = model.test(in_batch, label.shape[1])

            l1_err += l1_loss(pred, label).item()
            l2_err += l2_loss(pred, label).item()

    print("Test L1 error:", l1_err)
    print("Test L2 error:", l2_err)

    # visualize the prediction comparing to the ground truth
    if device is 'cpu':
        pred = pred.detach().numpy()[0,:,:]
        label = label.detach().numpy()[0,:,:]
    else:
        pred = pred.detach().cpu().numpy()[0,:,:]
        label = label.detach().cpu().numpy()[0,:,:]

    r = []
    num_points = 17
    interval = 1./num_points
    x = int(num_points/2)
    for j in range(-x,x+1):
        r.append(interval*j)

    plt.figure()
    for i in range(1, len(pred)):
        c = (i/(num_points+1), 1-i/(num_points+1), 0.5)
        plt.plot(pred[i], r, label='t = %s' %(i), c=c)
    plt.xlabel('velocity [m/s]')
    plt.ylabel('r [m]')
    plt.legend(bbox_to_anchor=(1,1),fontsize='x-small')
    plt.savefig("./pred_using_lstm.png")
    plt.show()

    plt.figure()
    for i in range(1, len(label)):
        c = (i/(num_points+1), 1-i/(num_points+1), 0.5)
        plt.plot(label[i], r, label='t = %s' %(i), c=c)
    plt.xlabel('velocity [m/s]')
    plt.ylabel('r [m]')
    plt.legend(bbox_to_anchor=(1,1),fontsize='x-small')
    plt.savefig("./ground_truth.png")
    plt.show()


def plot_prop(prop, prop_name, std=None):
    prop = np.array(prop)
    figure, ax = plt.subplots(1,1,figsize=(16,9))
    ax.plot(prop, color='orangered')
    ax.set(xlabel="epoch", ylabel=prop_name)

    if(std!=None):
        std = np.array(std)
        ax.fill_between(range(std.shape[0]), prop-std, prop+std, facecolor='peachpuff', alpha=0.7)
    plt.savefig("{}.png".format(prop_name))
    plt.close()

def save_model(state_dict, path):
    torch.save({
                'lstm_model':state_dict,
                },path)
def load_model(lstm, path):
    ckpt = torch.load(path)
    lstm.load_state_dict(ckpt['lstm_model'])
if __name__ == "__main__":
    main()

