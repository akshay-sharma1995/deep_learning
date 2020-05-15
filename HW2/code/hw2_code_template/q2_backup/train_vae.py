'''
train and test VAE model on airfoils
'''

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
import numpy as np

from dataset import AirfoilDataset
from vae import VAE
from utils import *
import pdb
from datetime import datetime
import argparse
import sys

watch = datetime.now()
device = ("cuda" if torch.cuda.is_available() else "cpu")

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--lr', dest='lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--num-epochs', dest='num_epochs', type=int, default=60, help='number of epochs')

    return parser.parse_args()

def main(args):
    args = parse_args()
    # check if cuda available
    # device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
# define dataset and dataloader
    dataset = AirfoilDataset()
    airfoil_x = dataset.get_x()
    airfoil_dim = airfoil_x.shape[0]
    airfoil_dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    # hyperparameters
    latent_dim = 16 # please do not change latent dimension
    lr = args.lr      # learning rate
    num_epochs = args.num_epochs

    # build the model
    vae = VAE(airfoil_dim=airfoil_dim, latent_dim=latent_dim, lr=lr).to(device)
    print("VAE model:\n", vae)

    # define your loss function here
    # loss = ?

    # define optimizer for discriminator and generator separately
    # train the VAE model
    epoch_loss_arr = []

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0
        for n_batch, (local_batch, __) in enumerate(airfoil_dataloader):
            num_batches += 1
            y_real = local_batch.to(device)

            # train VAE
            y_real = torch.tensor(y_real).float().to(device)
            
            y_gen, mean, logvar = vae(y_real) 
            var = torch.exp(logvar)

            # calculate customized VAE loss
            # loss = your_loss_func(...)
            
            rc_loss = torch.dist(y_real, y_gen, 2)
            prior_loss = -1.0*torch.sum((logvar + 1.0),1) + torch.sum(var,1) + torch.sum(mean**2,1)
            
            loss = rc_loss + 1.0*0.5*torch.mean(prior_loss)

            vae.optimizer.zero_grad()
            loss.backward()
            vae.optimizer.step()
            
            epoch_loss += loss.item()
            # print loss while training
            if (n_batch + 1) % 30 == 0:
                print("Epoch: [{}/{}], Batch: {}, loss: {}".format(
                    epoch, num_epochs, n_batch, loss.item()))
        epoch_loss_arr.append(epoch_loss/num_batches)
    # test trained VAE model
    num_samples = 100

    # reconstuct airfoils
    real_airfoils = dataset.get_y()[:num_samples]
    recon_airfoils, __, __ = vae(torch.from_numpy(real_airfoils).to(device))
    if 'cuda' in device:
        recon_airfoils = recon_airfoils.detach().cpu().numpy()
    else:
        recon_airfoils = recon_airfoils.detach().numpy()
    
    # randomly synthesize airfoils
    noise = torch.randn((num_samples, latent_dim)).to(device)   # create random noise 
    gen_airfoils = vae.decode(noise)
    if 'cuda' in device:
        gen_airfoils = gen_airfoils.detach().cpu().numpy()
    else:
        gen_airfoils = gen_airfoils.detach().numpy()

    # plot real/reconstructed/synthesized airfoils
    plot_airfoils(airfoil_x, real_airfoils, "vae_real_airfoils")
    plot_airfoils(airfoil_x, recon_airfoils, "vae_reconstructed_airfoils")
    plot_airfoils(airfoil_x, gen_airfoils, "vae_generated_airfoils")
    plot_prop(epoch_loss_arr, "vae_train_loss_lr_{}".format(lr))    

if __name__ == "__main__":
    main(sys.argv)
