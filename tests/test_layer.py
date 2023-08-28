"""
Module to test the layer parallel RNN layer.
"""

import torch
import torch.nn as nn

import torch.nn.functional as F

from parallel_rnn.layer_parallel_rnn import LayerParallelRNN

def test_layer_parallel_rnn():
    """
    Test the layer parallel RNN layer.
    """
    # Create a layer parallel RNN layer
    layer_parallel_rnn = LayerParallelRNN(10, 20, 30, 0.0)

    # Create a random input
    input_temporal = torch.randn(5, 8, 10)

    # Test the forward function
    output = layer_parallel_rnn(input_temporal)

    # Check the output shape
    assert output.shape == (5, 8, 30)

    # Check the gradient
    output.mean().backward()