import torch
import torch.nn as nn
import torch.nn.functional as F

class Tudui(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        y = x + 1
        return y

tudui = Tudui()
x = torch.tensor(1.0)
output = tudui(x)
print(output)

input = torch.tensor([[1, 2, 0, 3, 1],
                      [0, 1, 2, 3, 1],
                      [1, 2, 1, 0, 0],
                      [5, 2, 3, 1, 1],
                      [2, 1, 0, 1, 1]])

kernel = torch.tensor([[1, 2, 1],
                       [0, 1, 0],
                       [2, 1, 0]])

print(input.shape)
print(kernel.shape)
input = torch.reshape(input, (1,1,5,5))
kernel = torch.reshape(kernel, (1,1,3,3))
print(input.shape)
print(kernel.shape)

output1 = F.conv2d(input, kernel, stride=1)
print(output1)

output2 = F.conv2d(input, kernel, stride=2)
print(output2)

output3 = F.conv2d(input, kernel, stride=1, padding=1)
print(output3)