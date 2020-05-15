'''
train and test GAN model on airfoils
'''

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
import numpy as np
import sys
import argparse
from dataset import AirfoilDataset
from gan import Discriminator, Generator
from utils import *
import os
from datetime import datetime

watch = datetime.now()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--lr-dis', dest='lr_dis', type=float, default=2e-4, help='Discriminator_learning rate')
    parser.add_argument('--lr-gen', dest='lr_gen', type=float, default=2e-4, help='Generator_learning rate')
    parser.add_argument('--dis-hidden', dest='dis_hidden', type=int, default=50, help='Discriminator_hidden_dim')
    parser.add_argument('--gen-hidden', dest='gen_hidden', type=int, default=50, help='Generator_hidden_dim')
    parser.add_argument('--num-epochs', dest='num_epochs', type=int, default=60, help='number of epochs')
    parser.add_argument('--gen-update-interval', dest='gen_update_interval', type=int, default=1, help='generator_update_interval')
    return parser.parse_args()

def main(args):
    # check if cuda available
    # device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # define dataset and dataloader
    # dir_name = "{}_{}_{}_{}_{}".format(watch.year, watch.month, watch.day, watch.hour, watch.minute)
    args = parse_args()
    log_dir = os.path.join(os.getcwd(), "log_runs")

    dataset = AirfoilDataset()
    airfoil_x = dataset.get_x()
    airfoil_dim = airfoil_x.shape[0]
    airfoil_dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    # hyperparameters
    latent_dim = 16
    lr_dis = args.lr_dis # discriminator learning rate
    lr_gen = args.lr_gen # generator learning rate
    num_epochs = args.num_epochs
    disc_hidden_dim = args.dis_hidden
    gen_hidden_dim = args.gen_hidden
    
    # build the model
    dis = Discriminator(input_dim=airfoil_dim, hidden_dim=disc_hidden_dim, lr=lr_dis).to(device)
    gen = Generator(latent_dim=latent_dim, hidden_dim=gen_hidden_dim, airfoil_dim=airfoil_dim, lr=lr_gen).to(device)
    print("Distrminator model:\n", dis)
    print("Generator model:\n", gen)

    path = "./gan_model.ckpt"

    # train the GAN model
    save_model_interval = 10
    disc_loss_arr = []
    disc_fake_prob_std_arr = []
    disc_fake_prob_arr = []
    gen_loss_arr = []
    gen_update_interval = 2
    for epoch in range(num_epochs):
        epoch_disc_loss = 0.0
        epoch_gen_loss = 0.0
        disc_fake_prob = 0.0
        disc_fake_prob_std = 0.0
        num_batches = 0
        for n_batch, (local_batch, __) in enumerate(airfoil_dataloader):
            num_batches += 1
            y_real = local_batch.to(device)
            batch_size = y_real.shape[0] 
            y_real = torch.tensor(y_real).float().to(device) # train generator
            
            z = torch.randn((batch_size, latent_dim)).to(device)
            
            # train discriminator
            with torch.no_grad():
                gen_sample = gen(z)
        
            disc_real = dis(y_real)
            disc_fake = dis(gen_sample)

            loss_dis = torch.log(disc_real) + torch.log(1.0 - disc_fake)
            loss_dis = -0.5 * loss_dis.mean()

            dis.optimizer.zero_grad()
            loss_dis.backward()
            dis.optimizer.step()
            epoch_disc_loss += loss_dis.item()

            # train generator
            gen_sample = gen(z)
            fake_prob = dis(gen_sample)
            loss_gen = -torch.log(fake_prob)
            loss_gen = loss_gen.mean()
                
            if(num_batches%gen_update_interval==0):
                gen.optimizer.zero_grad()
                loss_gen.backward()
                gen.optimizer.step()
            
            disc_fake_prob_std += torch.std(fake_prob).item()
            disc_fake_prob += fake_prob.clone().detach().mean().item()
            
            epoch_gen_loss += loss_gen.item()

            # print loss while training
            if (n_batch + 1) % 30 == 0:
                print("Epoch: [{}/{}], Batch: {}, Discriminator loss: {}, Generator loss: {}".format(
                    epoch, num_epochs, n_batch, loss_dis.item(), loss_gen.item()))
        print("entry made")
        
        if((epoch+1)%save_model_interval==0):
            saveGAN(dis.state_dict(), gen.state_dict(), path)

        disc_fake_prob_arr.append(disc_fake_prob / num_batches)
        disc_fake_prob_std_arr.append((disc_fake_prob_std / num_batches))
        disc_loss_arr.append(epoch_disc_loss / num_batches)
        gen_loss_arr.append(epoch_gen_loss / num_batches)
    
    
    # test trained GAN model
    num_samples = 100
    # create random noise 
    noise = torch.randn((num_samples, latent_dim)).to(device)
    # noise = torch.randn((num_samples, noise_dim)).to(device)
    # generate airfoils
    gen_airfoils = gen(noise)
    # if 'cuda' in device:
    if device.type=='cuda':
        gen_airfoils = gen_airfoils.detach().cpu().numpy()
    else:
        gen_airfoils = gen_airfoils.detach().numpy()
    # gen_airfoils = gen_airfoils.detach().cpu().numpy()

    # plot generated airfoils
    plot_airfoils(airfoil_x, gen_airfoils, "gan_generated_airfoils.png")
    plot_prop(disc_loss_arr, "disc_loss")
    plot_prop(gen_loss_arr, "gen_loss")
    plot_prop(disc_fake_prob_arr, "disc_fake_prob_lrd_{}_lrg_{}_epochs_{}".format(lr_dis, lr_gen, num_epochs), disc_fake_prob_std_arr)


def saveGAN(disc_state_dict, gen_state_dict, path):
    torch.save({
                'disc_state_dict':disc_state_dict,
                'gen_state_dict':gen_state_dict,
                }, path)

def loadGAN(disc, gen, path):
    chckpt = torch.load(path)
    disc.load_state_dict(chckpt['disc_state_dict'])
    gen.load_state_dict(chckpt['gen_state_dict'])


if __name__ == "__main__":
    main(sys.argv)

