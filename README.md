# First-Neural-Network
I took a tutorial to learn building a neural network in PyTorch

Here's the code:

import torch
from torch import nn
import random

def mystery(a,b):
  return torch.tensor(a+3*b)
  
model = nn.Sequential(nn.Linear(2,1))
model

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

for i in range(1000):

  a = random.random()
  b = random.random()
  desiredOutput = mystery(a,b)

  output = model(torch.tensor([a,b]))
  loss = criterion(output.squeeze(), desiredOutput)

  if (i % 100)==0:
    print (f"Loss: {loss.item()}")

  optimizer.zero_grad()
  loss.backward()
  optimizer.step()

