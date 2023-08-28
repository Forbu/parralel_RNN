"""
In this section, we will implement the layer parallel RNN.

This is a simple RNN-like layer that can be used as a building block for more complex models.
Basicly the idea is that we have a MLP combine with a CUMSUM layer.
"""

import torch
import torch.nn as nn

import torch.nn.functional as F

class LayerParallelRNN(nn.Module):
    """
    This is a simple RNN-like layer that can be used as a building block for more complex models.

    There is basicly 3 steps in this layer:
    1. We have a MLP layer to get the hidden state.
    2. A learnable attenuation factor to control the contribution of the previous hidden state.
    3. A cumsum layer to accumulate the hidden state.

    Optional parameters:
    - A learnable position embedding can be added to the input.

    """

    def __init__(self, dim_input, dim_hidden, dim_output, dropout=0.0):
        super().__init__()
        self.dim_input = dim_input
        self.dim_hidden = dim_hidden
        self.dim_output = dim_output
        self.dropout = dropout

        self.mlp = nn.Sequential(
            nn.Linear(dim_input, dim_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_hidden, dim_output)
        )   # MLP layer

        self.attenuation = nn.Parameter(
            torch.ones(1, dim_output))   # Attenuation factor

    def forward(self, input_temporal, hidden=None):
        """
        Args:
            input_temporal: input tensor with shape [batch_size, seq_len, dim_input]
            hidden: hidden state with shape [batch_size, dim_hidden]

        Returns:
            output: output tensor with shape [batch_size, seq_len, dim_output]
            hidden: hidden state with shape [batch_size, dim_hidden]
        """
        batch_size, seq_len, dim_input = input_temporal.shape

        # MLP layer
        output = self.mlp(input_temporal)

        # Attenuation factor modification 
        # attenuation is currently a vector of size [1, dim_hidden]
        # we need to expand it to [1, 1, dim_hidden]
        # so that we can multiply it with the output
        attenuation = F.sigmoid(self.attenuation.unsqueeze(0))

        output = output #* attenuation

        # Cumsum layer
        output = torch.cumsum(output, dim=1)

        # we can add the previous hidden state to the output
        if hidden is not None:
            # check the dimension of the hidden state
            assert hidden.shape == (batch_size, self.dim_hidden)

            output = output + hidden.unsqueeze(1)

        return output
