import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim

import os
import numpy as np
import pandas as pd


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(LSTMModel, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim

        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):

        h0 = torch.zeros(self.layer_dim, x.size(
            0), self.hidden_dim).requires_grad_().to(self.device)
        c0 = torch.zeros(self.layer_dim, x.size(
            0), self.hidden_dim).requires_grad_().to(self.device)

        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        output = torch.zeros(50, 3)

        for i in range(50):
            output[i] = self.fc(out[0][i].view(1, 1, 50))

        # out = self.fc(out[:, -1:])

        return output


if __name__ == "__main__":

    test = LSTMModel(5, 50, 1, 3)
    print(test.forward(torch.zeros(1, 50, 5)).shape)
