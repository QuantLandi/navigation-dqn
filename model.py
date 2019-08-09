import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Deep Q-Network function approximator"""

    def __init__(self, state_size, action_size, seed, layer1_units=64, layer2_units=64):
        """Initializes parameters and defines model architecture.
        
        Parameters
        ----------
            state_size : int
                Dimension of each state
            action_size : int
                Dimension of each action
            seed : int
                Random seed
            layer1_units : int
                Number of nodes in first hidden layer
            layer2_units : int
                Number of nodes in second hidden layer
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        # define model architecture
        self.layer1_logits = nn.Linear(state_size, layer1_units)
        self.layer2_logits = nn.Linear(layer1_units, layer2_units)
        self.output_layer_logits = nn.Linear(layer2_units, action_size)
        

    def forward(self, state):
        """Defines activation functions for each layer and returns model output"""
        layer1_activation = F.relu(self.layer1_logits(state))
        layer2_activation = F.relu(self.layer2_logits(layer1_activation))
        output_layer_logits = self.output_layer_logits(layer2_activation)
        return output_layer_logits
