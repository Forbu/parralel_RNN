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

    def __init__(self, dim_input, dim_hidden, dim_output, dropout=0.0, decay=None, seq_len=None):
        super().__init__()
        self.dim_input = dim_input
        self.dim_hidden = dim_hidden
        self.dim_output = dim_output
        self.dropout = dropout

        if decay is not None:
            self.decay_activation = True

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

        if decay is not None:
            # we register the decay factor as register buffer
            self.register_buffer('decay', decay)

            # we also register a decay matrix D
            self.register_buffer('D', compute_decay_factor(seq_len, decay))

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
        if self.decay_activation:
            output = decay_output(output, self.D)

        # Cumsum layer
        output = torch.cumsum(output, dim=1)

        # we can add the previous hidden state to the output
        if hidden is not None:
            # check the dimension of the hidden state
            assert hidden.shape == (batch_size, self.dim_hidden)

            output = output + hidden.unsqueeze(1)

        return output


def compute_decay_factor(seq_len, decay):
    """
    We compute the D matrix for the decay cumsum layer.
    """
    # first compute the matrix n - m
    D = torch.arange(seq_len).unsqueeze(
        0) - torch.arange(seq_len).unsqueeze(1)

    # then compute the decay factor
    list_D = []

    for i in range(decay.shape[0]):
        list_D.append(torch.triu(decay[i] ** D).unsqueeze(-1))

    D = torch.cat(list_D, dim=-1)

    return D


def decay_output(input_temporal, D):
    """
    Function implementing the decay cumsum layer.

    Args:
        input_temporal: input tensor with shape [batch_size, seq_len, dim_input]
        decay: decay factor with shape [dim_input]

    Returns:
        output: output tensor with shape [batch_size, seq_len, dim_input]

    The idea is that we have a cumsum layer with a decay factor.
    basicly if we have a temporal input like this:
    [x1, x2, x3, x4, x5]
    the output will be:
    [x1, x1*decay + x2, x1*decay^2 + x2*decay + x3, x1*decay^3 + x2*decay^2 + x3*decay + x4, x1*decay^4 + x2*decay^3 + x3*decay^2 + x4*decay + x5]
    """
    # first compute the matrix
    # Dnm =
    #   γ ** (n − m), n ≥ m
    #   0, n < m
    assert D.shape[0] == input_temporal.shape[1]

    # add a dimension to the input : [batch_size, seq_len, 1, dim_input]
    input_temporal = input_temporal.unsqueeze(-2)

    # we repeat the input to match the dimension of the matrix
    input_temporal = input_temporal.repeat(1, 1, D.shape[0], 1)

    # we multiply the input by the matrix
    output = input_temporal * D.unsqueeze(0)

    # we sum the output
    output = torch.sum(output, dim=1)

    return output


def decay_cumsum(input_temporal, decay):
    """
    Function implementing the decay cumsum layer.

    Args:
        input_temporal: input tensor with shape [batch_size, seq_len, dim_input]
        decay: decay factor with shape [dim_input]

    Returns:
        output: output tensor with shape [batch_size, seq_len, dim_input]

    The idea is that we have a cumsum layer with a decay factor.
    basicly if we have a temporal input like this:
    [x1, x2, x3, x4, x5]
    the output will be:
    [x1, x1*decay + x2, x1*decay^2 + x2*decay + x3, x1*decay^3 + x2*decay^2 + x3*decay + x4, x1*decay^4 + x2*decay^3 + x3*decay^2 + x4*decay + x5]
    """
    # first compute the matrix
    # Dnm =
    #   γ ** (n − m), n ≥ m
    #   0, n < m
    seq_len = input_temporal.shape[1]

    # first compute the matrix n - m
    D = compute_decay_factor(seq_len, decay)

    output = decay_output(input_temporal, D)

    return output
