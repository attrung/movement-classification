import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class LSTMClassifier(nn.Module):

    def __init__(self, input_dim, hidden_dim, target_size):
        super(LSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(input_dim, hidden_dim)
        self.linear = nn.Linear(hidden_dim, target_size)

    def forward(self, data):
        hidden = (torch.randn(1, 1, self.hidden_dim), torch.randn(1, 1, self.hidden_dim))
        for i in data:
            out, hidden = self.lstm(i.view(1, 1, -1), hidden)
        pred = self.linear(out)
        pred = F.log_softmax(pred, dim=1)
        return pred