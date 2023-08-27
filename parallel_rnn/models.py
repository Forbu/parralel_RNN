"""
This module is used to create a full model from the layer parallel RNN layer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from parallel_rnn.layer_parallel_rnn import LayerParallelRNN


class ParallelRNN(nn.Module):
    """
    This class is used to create a full model from the layer parallel RNN layer.
    This will include the layer parallel RNN layer and embedding layer if needed.
    """

    def __init__(self, nb_layers, dim_input, dim_output, dim_hidden, dropout=0.0, use_embedding_temporal=False, dim_embedding=0, nb_temporal=0):

        super().__init__()
        self.nb_layers = nb_layers
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.dim_hidden = dim_hidden
        self.dropout = dropout
        self.use_embedding_temporal = use_embedding_temporal
        self.dim_embedding = dim_embedding

        # Embedding layer
        if self.use_embedding_temporal:
            self.embedding = nn.Parameter(torch.randn(nb_temporal, dim_embedding))
            self.all_dim_input = dim_input + dim_embedding
        else:
            self.all_dim_input = dim_input

        # adding encoder layer to preprocess the input
        self.encoder = nn.Linear(self.all_dim_input, self.dim_hidden),

        # Layer parallel RNN layers
        self.rnn_layers = nn.ModuleList([LayerParallelRNN(
            self.dim_hidden, self.dim_hidden, self.dim_hidden, self.dropout) for _ in range(self.nb_layers)])

        # Output layer
        self.output_layer = nn.Linear(self.dim_hidden, self.dim_output)

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

        # Embedding layer
        if self.use_embedding_temporal:
            input_temporal = torch.cat(
                (input_temporal, self.embedding.expand(batch_size, seq_len, self.dim_embedding)), dim=2)
            
        # adding encoder layer to preprocess the input
        input_temporal = self.encoder(input_temporal)

        # Layer parallel RNN layers
        for i in range(self.nb_layers):
            input_temporal = self.rnn_layers[i](input_temporal, hidden)

        # Output layer
        output = self.output_layer(input_temporal)

        return output
