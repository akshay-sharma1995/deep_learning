import torch
import torch.nn as nn
import pdb

class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Encoder, self).__init__()
        # build your model here
        # your model should output a predicted mean and a predicted std of the encoding
        # both should be of dim (batch_size, latent_dim)
    
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(nn.Linear(input_dim, 684),
                                nn.ReLU(),
                                nn.Linear(684, 512),
                                nn.ReLU(),
                                nn.Linear(512, 256),
                                nn.ReLU(),
                                nn.Linear(256, 2*latent_dim),
                                )



    def forward(self, x):
        # define your feedforward pass
        out = self.encoder(x)
        mean = out[:,0:self.latent_dim]
        logvar = out[:,self.latent_dim:]
        
        return mean, logvar


class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(Decoder, self).__init__()
        # build your model here
        # your output should be of dim (batch_size, output_dim)
        # you can use tanh() as the activation for the last layer
        # since y coord of airfoils range from -1 to 1
        
        self.latent_dim = latent_dim
        self.output_dim = output_dim

        self.decoder = nn.Sequential(nn.Linear(latent_dim, 684),
                                nn.ReLU(),
                                nn.Linear(684, 512),
                                nn.ReLU(),
                                nn.Linear(512, 256),
                                nn.ReLU(),
                                nn.Linear(256,output_dim),
                                nn.Tanh(),
                                )

    def forward(self, x):
        # define your feedforward pass
        out = self.decoder(x)
        
        return out

class VAE(nn.Module):
    def __init__(self, airfoil_dim, latent_dim, lr):
        super(VAE, self).__init__()
        self.enc = Encoder(airfoil_dim, latent_dim)
        self.dec = Decoder(latent_dim, airfoil_dim)
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr)
        
    def forward(self, x):
        # define your feedforward pass
        mean, logvar = self.enc(x)
        # mean, std  = dist_params[:,0:latent_dim], encoder_vec[:,latent_dim:]
        std = torch.exp(0.5*logvar)
       
        # reparamaterization
        tau = torch.randn_like(std).to(mean.device)
        # tau = torch.normal(mean=torch.zeros(1,1), std=torch.ones(1,1)).to(mean.device).requires_grad_(False)
        sample_z = mean + tau*std
        
        gen_out = self.dec(sample_z)
        return gen_out, mean, logvar

    def decode(self, z):
        # given random noise z, generate airfoils
        return self.dec(z)

    def save_model(self, path):
        torch.save(self.state_dict(),path)

    def load_model(self, path):
        self.load_state_dict(torch.load(path))
