import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dataloader import MovementDataset
from model import LSTMClassifier

torch.manual_seed(1)

train_data = MovementDataset("train")
valid_data = MovementDataset("valid")

model = LSTMClassifier(19, 64, train_data.label_size)
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

for epoch in range(1):
    for data, target in train_data:
        model.zero_grad()
        output = model(data)
        loss = loss_function(output, target)
        loss.backward()
        optimizer.step()
        print(loss)