import torch
import torch.nn as nn

# Define the CNN-Transformer Encoder model
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()

        # Conv Block
        self.conv = nn.Sequential(
                            nn.Conv2d(1, 3, 1, 1),

                    )

        # Fully connected layers for classification
        self.fc_layers = nn.Sequential(
            nn.Linear(15 * 2, 10),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(10, 6)
        )
        self.soft = nn.Softmax()

    def forward(self, x):
        x = self.soft(self.fc_layers(x))
        return xclass MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()

        # Fully connected layers for classification
        self.fc_layers = nn.Sequential(
            nn.Linear(15 * 2, 10),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(10, 6)
        )
        self.soft = nn.Softmax()

    def forward(self, x):
        x = self.soft(self.fc_layers(x))
        return x