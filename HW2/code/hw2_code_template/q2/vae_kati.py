import torch
import torch.nn as nn
import torch.distributions
from torch.nn import functional as F
import pdb


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Encoder, self).__init__()
        # build your model here
        # your model should output a predicted mean and a predicted std of the encoding
        # both should be of dim (batch_size, latent_dim)

        self.input_dim  = input_dim
        self.latent_dim = latent_dim



        self.l1   = nn.Linear(self.input_dim , 684)
        self.l2   = nn.Linear(684, 512)
        self.l3 = nn.Linear(512,256)
        self.l4 = nn.Linear(256,64)
        self.M    = nn.Linear(64, self.latent_dim)
        self.logvar  = nn.Linear(64, self.latent_dim)
    
    def forward(self, x):
        # define your feedforward pass

        l1     = F.relu( self.l1(x) )
        l2     = F.relu( self.l2(l1) )
        l3 = F.relu(self.l3(l2))
        l4 = F.relu(self.l4(l3))
        M      = self.M(l4)
        logvar = self.logvar(l4)

        return M, logvar


    def getSample(self, M, logvar):
        std = torch.exp(logvar/2)
        return M + std * torch.randn_like(std).to(device=DEVICE).float()


class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(Decoder, self).__init__()
        # build your model here
        # your output should be of dim (batch_size, output_dim)
        # you can use tanh() as the activation for the last layer
        # since y coord of airfoils range from -1 to 1

        self.latent_dim = latent_dim
        self.output_dim = output_dim

        self.l1   = nn.Linear(self.latent_dim, 684)
        self.l2   = nn.Linear(684, 512)
        self.l3  = nn.Linear(512, 256)
        self.l4 = nn.Linear(256,self.output_dim)
    
    def forward(self, x):

        # define your feedforward pass

        l1 = F.relu( self.l1(x) )
        l2 = F.relu( self.l2(l1) )
        l3 = F.relu( self.l3(l2) )
        l4 = self.l4(l3)
        out= torch.tanh(l4)
        return out


class VAE(nn.Module):
    def __init__(self, airfoil_dim, latent_dim):
        super(VAE, self).__init__()
        self.enc = Encoder(airfoil_dim, latent_dim)
        self.dec = Decoder(latent_dim, airfoil_dim)
    
    def forward(self, x):
        # define your feedforward pass
        M, logvar = self.enc(x)
        z = self.enc.getSample(M, logvar)
        return self.decode(z), M, logvar

    def decode(self, z):

        # given random noise z, generate airfoils
        return self.dec(z)

    def computeLoss(self,M,logvar,genData, x):

        # reconLoss = F.binary_cross_entropy(genData, x, reduction= 'sum')
        # pdb.set_trace()

        reconLoss = torch.sum((genData - x)**2,1)
        reconLoss = torch.mean(reconLoss)
        KLD = -torch.sum(1 + logvar - M*M - torch.exp(logvar)) / logvar.shape[0]

        return reconLoss +  0.5 * KLD

