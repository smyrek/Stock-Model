import torch
import torch.nn as nn
import torch.optim as optim

from models import LSTMModel
from dataset import StockDataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'

input_dim = 5
hidden_dim = 50
layer_dim = 1
output_dim = 3

num_epochs = 10

dset = StockDataset("./data")

train_loader = torch.utils.data.DataLoader(dataset=dset, shuffle=False)

model = LSTMModel(input_dim, hidden_dim, layer_dim, output_dim).to(device)

criterion = nn.L1Loss().to(device)

learning_late = 0.01

optimizer = optim.Adam(model.parameters(), lr=learning_late)

for epoch in range(num_epochs):
    total_loss = 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()

        outputs = model(x).to(device)

        loss = criterion(outputs, y.view(-1, 3))

        loss.backward()

        optimizer.step()

        total_loss = loss

    print(outputs[-3:])
    print(y.view(-1, 3)[-3:])
    print(f"Epoch {epoch} loss = {total_loss/len(train_loader)}")
