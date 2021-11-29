import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class LSTMClassifier(nn.Module):

    def __init__(self, input_dim, hidden_dim, target_size, device = "cpu"):
        super(LSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.device = device

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, 64, 3),
            nn.MaxPool1d(2),
            nn.Dropout(p=0.05, inplace = False),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, 5),
            nn.MaxPool1d(2),
            nn.Dropout(p=0.05, inplace = False),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 5),
            nn.MaxPool1d(2),
            nn.ReLU(inplace = True),
            nn.Conv1d(256, 256, 7),
            nn.MaxPool1d(3),
            nn.ReLU(inplace = True),
        )

        self.lstm = nn.LSTM(256, hidden_dim)
        self.linear = nn.Linear(hidden_dim, target_size)
        self.softmax = nn.Softmax(dim = 1)

    def forward(self, data):
        hidden = (
            torch.randn(1, 1, self.hidden_dim).to(self.device), 
            torch.randn(1, 1, self.hidden_dim).to(self.device),
        )
        data = data.view(1, data.size(2), data.size(1))
        out = self.conv(data)
        out = out.view(out.size(-1), 1, 256)
        out, hidden = self.lstm(out, hidden)
        pred = self.linear(out[-1])
        return pred