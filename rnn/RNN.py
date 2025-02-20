import torch
import torch.nn as nn


# class RNN(nn.Module):
#     def __init__(self, input_size, hidden_size=32, num_layers=1, output_size=1):
#         super(RNN, self).__init__()
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
#         self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        
#         self.fc = nn.Linear(hidden_size, output_size) # fc layer to map hidden to output
    
#     def forward(self, x):
#         # init hidden state
#         h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
#         out, _ = self.rnn(x, h0)
#         out = self.fc(out[:, -1, :])
#         return out


# class GRU(nn.Module):
#     def __init__(self, input_size, hidden_size, num_layers, output_size=1):
#         super(GRU, self).__init__()
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
#         self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
#         self.fc = nn.Linear(hidden_size, output_size)  # Fully connected layer to map hidden to output
    
#     def forward(self, x):
#         h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
#         out, _ = self.gru(x, h0)
#         out = self.fc(out[:, -1, :])
#         return out


# variable size hidden layer GRU
class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, dropout, output_size=1):
        super(GRU, self).__init__()
        self.num_layers = len(hidden_size)
        self.hidden_sizes = hidden_size
        
        self.gru_layers = nn.ModuleList()
        for i in range(self.num_layers):
                in_size = input_size if i == 0 else hidden_size[i - 1]
                self.gru_layers.append(nn.GRU(in_size, hidden_size[i], num_layers=1, batch_first=True))

        # self.dropout = nn.Dropout(dropout) if self.num_layers > 1 else None
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size[-1], output_size)
    
    def forward(self, x):
        h = [torch.zeros(1, x.size(0), hidden_size).to(x.device) for hidden_size in self.hidden_sizes]
        
        out = x
        for i, gru_layer in enumerate(self.gru_layers):
            out, h[i] = gru_layer(out, h[i])
            if i == 0 and self.dropout is not None:
                out = self.dropout(out)

        out = self.fc(out[:, -1, :])
        return out
