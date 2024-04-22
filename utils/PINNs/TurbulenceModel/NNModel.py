from torch import nn
import torch.nn.functional as F


class NNModel(nn.Module):
    """
    Defines a simple feedforward neural network model.

    Parameters:
        input_dim (int): The number of input features to the network.
        output_dim (int): The number of output predictions the network makes.
        hidden_dim (int): The dimensionality of the hidden layers. Defaults to 64.
        hidden_depth (int): The number of hidden layers in the network. Defaults to 8.
        activation (str): The type of activation function to use in the hidden layers. Supported functions include 'tanh', 'relu', 'sigmoid', 'leaky_relu', 'elu', and 'selu'.
    """

    def __init__(
        self, input_dim, output_dim, hidden_dim=64, hidden_depth=8, activation="tanh"
    ):
        super(NNModel, self).__init__()
        # Initialize the first layer with input_dim to hidden_dim
        layers = [nn.Linear(input_dim, hidden_dim)]

        # Dynamically create the activation layer based on the provided argument
        activation_layer = self.get_activation_layer(activation)

        # Add pairs of activation and linear layers based on hidden_depth
        for _ in range(hidden_depth - 1):
            layers.append(activation_layer)
            layers.append(nn.Linear(hidden_dim, hidden_dim))

        # Append final activation and output layer
        layers.append(activation_layer)
        layers.append(nn.Linear(hidden_dim, output_dim))

        # Compose the model from the list of layers
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass of the network.

        Parameters:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: The network's output tensor.
        """
        return self.model(x)

    def get_activation_layer(self, activation):
        """
        Returns an instance of the specified activation function.

        Parameters:
            activation (str): The name of the activation function.

        Returns:
            nn.Module: An instance of the torch.nn activation function.
        """
        if activation == "tanh":
            return nn.Tanh()
        elif activation == "relu":
            return nn.ReLU()
        elif activation == "sigmoid":
            return nn.Sigmoid()
        elif activation == "leaky_relu":
            return nn.LeakyReLU()
        elif activation == "elu":
            return nn.ELU()
        elif activation == "selu":
            return nn.SELU()
        else:
            # Raise an error if the specified activation function is not supported
            raise ValueError(f"Unsupported activation: {activation}")
