import torch
from torch import nn


class BTCPredModelV1(torch.nn.Module):

    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.block_1 = nn.Sequential(
            nn.Conv1d(in_channels=input_shape, 
                      out_channels=hidden_units, 
                      kernel_size=3, 
                      stride=1,
                      padding=1),

            nn.Conv1d(in_channels=hidden_units, 
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),

        )
        self.block_2 = nn.Sequential(
            nn.Conv1d(in_channels=hidden_units, out_channels=hidden_units//2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=hidden_units//2, out_channels=hidden_units//2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
           
            nn.Linear(in_features=(hidden_units // 2) * (input_shape // 2), 
                      out_features=output_shape)
        )
    
    def forward(self, x: torch.Tensor):
        #print(f"Before Forward block1 ---> {x.shape}")
        x = self.block_1(x)
        #print(f"After Forward block1 ---> {x.shape}")
        x = self.block_2(x)
        #print(f"After Forward block2 ---> {x.shape}")
        x = self.classifier(x)
        #print(f"After Forward classification ---> {x.shape}")
        return x


model = BTCPredModelV1(14,64,3)

