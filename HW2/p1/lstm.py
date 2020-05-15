import torch
import torch.nn as nn
from torch.autograd import Variable
import pdb

class FlowLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, learning_rate):
        super(FlowLSTM, self).__init__()
        # build your model here
        # your input should be of dim (batch_size, seq_len, input_size)
        # your output should be of dim (batch_size, seq_len, input_size) as well
        # since you are predicting velocity of next step given previous one
        
        # feel free to add functions in the class if needed
        self.h = None
        self.c = None
        self.hidden_size = hidden_size

        self.recurrent_unit = nn.LSTMCell(input_size, hidden_size)
        self.linear = nn.Linear(hidden_size, input_size)

        self.optimizer = torch.optim.Adam(self.parameters(), learning_rate)

    def initialize_h_c(self, batch_size, device):
        self.h = torch.zeros(batch_size, self.hidden_size).to(device)
        self.c = torch.zeros(batch_size, self.hidden_size).to(device)

    # forward pass through LSTM layer
    def forward(self, x):
        '''
        input: x of dim (batch_size, 19, 17)
        '''
        # define your feedforward pass
        self.initialize_h_c(x.shape[0], x.device)
        output_seq = []
        for i in range(x.shape[1]):
            self.h, self.c = self.recurrent_unit(x[:,i,:], (self.h, self.c))
            output_seq.append(self.linear(self.h))
        
        return torch.stack(output_seq, 1)


    # forward pass through LSTM layer for testing
    def test(self, x, seq_len):
        '''
        input: x of dim (batch_size, 17)
        '''
        self.initialize_h_c(x.shape[0], x.device)
            
        output_seq = []
        # output_seq.append(x*1.0)
        # define your feedforward pass
        for i in range(seq_len):
            self.h, self.c = self.recurrent_unit(x, (self.h, self.c))
            output_seq.append(self.linear(self.h))
            x = output_seq[-1]*1.0
        return torch.stack(output_seq, 1)






