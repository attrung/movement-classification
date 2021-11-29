import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dataloader import MovementDataset
from model import LSTMClassifier

torch.manual_seed(1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_data = MovementDataset("train")
valid_data = MovementDataset("valid")

train_loader = torch.utils.data.DataLoader(train_data, batch_size=1, shuffle=True, num_workers= 1)
valid_data = torch.utils.data.DataLoader(valid_data, batch_size=1, shuffle=True, num_workers= 1)

model = LSTMClassifier(19, 128, train_data.label_size, device)
model = model.to(device)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.03)

total_loss = 0
idx = 0
acc = 0
for step in range(100000):
    model.train()
    for data, target in train_loader:
        optimizer.zero_grad()
        model.zero_grad()
        data = data.to(device)
        target = target.to(device)
        output = model(data)
        _, predicted = torch.max(output.data, 1)
        if predicted == target:
            acc += 1
        loss = loss_function(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss
        idx += 1
        if idx == 1000:
            print(total_loss/1000)
            print(acc / 1000)
            total_loss = 0
            idx = 0
            acc = 0
        
    
