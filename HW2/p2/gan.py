import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim, lr):
        super(Discriminator, self).__init__()
        # build your model here
        # your output should be of dim (batch_size, 1)
        # since discriminator is a binary classifier
    
        self.input_dim = input_dim 
        self.lr = lr

        self.disc = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                nn.ReLU(),
                                nn.Linear(hidden_dim, 512),
                                nn.ReLU(),
                                nn.Linear(512, 256),
                                nn.ReLU(),
                                nn.Linear(256,1),
                                nn.Sigmoid(),
                                )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, x):
        # define your feedforward pass
        return self.disc(x)

    def save_model(self, path):
        torch.save(self.state_dict(),path)

    def load_model(self, path):
        self.load_state_dict(torch.load(path))


class Generator(nn.Module):
    def __init__(self, latent_dim, hidden_dim, airfoil_dim, lr):
        super(Generator, self).__init__()
        # build your model here
        # your output should be of dim (batch_size, airfoil_dim)
        # you can use tanh() as the activation for the last layer
        # since y coord of airfoils range from -1 to 1
        
        self.latent_dim = latent_dim
        self.airfoil_dim = airfoil_dim
        self.learning_rate = lr
        
        self.gen = nn.Sequential(nn.Linear(latent_dim, hidden_dim),
                                nn.ReLU(),
                                nn.Linear(hidden_dim, 512),
                                nn.ReLU(),
                                nn.Linear(512, 256),
                                nn.ReLU(),
                                nn.Linear(256,airfoil_dim),
                                nn.Tanh(),
                                )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, x):
        # define your feedforward pass

        return self.gen(x)

    def save_model(self, path):
        torch.save(self.state_dict(),path)

    def load_model(self, path):
        self.load_state_dict(torch.load(path))

