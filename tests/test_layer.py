"""
Module to test the layer parallel RNN layer.
"""

import torch
import torch.nn as nn

import torch.nn.functional as F

from parallel_rnn.layer_parallel_rnn import LayerParallelRNN, decay_cumsum


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


def test_decay_cumsum():
    # Define a batch of 2 sequences of length 3 with dimension 2
    input_temporal = torch.tensor([[[1., 2.], [3., 4.], [5., 6.]], [
                                  [7., 8.], [9., 10.], [11., 12.]]])

    print(input_temporal.shape)


    # Define decay factor
    decay = torch.tensor([0.5, 0.5])

    # Call the function
    output = decay_cumsum(input_temporal, decay)

    # Define the expected output
    expected_output = torch.tensor([[[1.0000,  2.0000],
                                     [3.5000,  5.0000],
                                     [6.7500,  8.500]],
                                    [[7.0000,  8.0000],
                                     [12.5000, 14.000],
                                     [17.2500, 19.00]]])
    
    print(input_temporal)

    print(output)

    # Check if the output matches the expected output
    assert torch.allclose(
        output, expected_output), "The function is not working as expected"
