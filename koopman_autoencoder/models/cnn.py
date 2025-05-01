import itertools
import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, C_in, C_out, block_size=1, kernel_size=3, decoder_block=False, **conv_kwargs):
        """
        A modular convolutional block consisting of convolutional layers followed by ReLU activations.

        Parameters:
            C_in: int
                Number of input channels.
            C_out: int
                Number of output channels.
            block_size: int
                Number of convolutional layers in the block.
            kernel_size: int
                Size of the convolution kernel.
            decoder_block: bool
                If True, the block is used in a decoder, and layer configurations are adjusted accordingly.
            conv_kwargs: dict
                Additional arguments for nn.Conv2d.
        """
        super().__init__()
        if not decoder_block:
            first_layer = [nn.Conv2d(C_in, C_out, kernel_size, **conv_kwargs), nn.ReLU()]
            subsequent_layers = (block_size-1) * [
                nn.Conv2d(C_out, C_out, kernel_size, **conv_kwargs),
                nn.ReLU()
            ]
            self.stack = nn.ModuleList([*first_layer, *subsequent_layers])
        else:
            initial_layers = (block_size-1) * [
                nn.Conv2d(C_in, C_in, kernel_size, **conv_kwargs),
                nn.ReLU()
            ]
            output_layer = [nn.Conv2d(C_in, C_out, kernel_size, **conv_kwargs), nn.ReLU()]
            self.stack = nn.ModuleList([*initial_layers, *output_layer])

    def forward(self, x):
        for module in self.stack:
            x = module(x)
        return x


class BaseEncoderDecoder(nn.Module):
    def __init__(self, C, H, W, latent_dim, hiddens, block_size=1, kernel_size=3, is_encoder=True, **conv_kwargs):
        """
        Base class for both encoder and decoder blocks.

        Parameters:
            C: int
                Number of input/output channels.
            H: int
                Input height dimension.
            W: int
                Input width dimension.
            D: int
                Latent dimensionality.
            hiddens: list of int
                List of hidden dimensions for each block.
            block_size: int
                Number of convolutional layers in each block.
            kernel_size: int
                Size of the convolutional kernel.
            is_encoder: bool
                Specifies whether the block is an encoder (True) or decoder (False).
            conv_kwargs: dict
                Additional arguments for nn.Conv2d.
        """
        super().__init__()
        self.C = C
        self.H = H
        self.W = W
        self.D = latent_dim
        self.hiddens = hiddens
        self.is_encoder = is_encoder

        # Compute the output dimensions after pooling
        self.n_pools = len(hiddens)
        if H % (2**self.n_pools) != 0 or W % (2**self.n_pools) != 0:
            raise ValueError(
                f"Input dimensions (H={H}, W={W}) must be divisible by 2^{self.n_pools} due to pooling."
            )
        self.H_out = H // (2**(self.n_pools))
        self.W_out = W // (2**(self.n_pools))

        # Define the linear layer
        if is_encoder:
            self.linear = nn.Linear(hiddens[-1] * self.H_out * self.W_out, latent_dim)
        else:
            self.linear = nn.Linear(latent_dim, hiddens[-1] * self.H_out * self.W_out)

        # Build convolutional layers
        self.layers = self._build_layers(block_size, kernel_size, conv_kwargs)

    def _build_layers(self, block_size, kernel_size, conv_kwargs):
        """
        Build the layers for the encoder or decoder.

        Parameters:
            block_size: int
                Number of convolutional layers in each block.
            kernel_size: int
                Size of the convolutional kernel.
            conv_kwargs: dict
                Additional arguments for nn.Conv2d.

        Returns:
            nn.Sequential: Sequential container of layers.
        """
        layers = nn.ModuleList()
        if self.is_encoder:
            layers.append(ConvBlock(self.C, self.hiddens[0], block_size, kernel_size, decoder_block=False, **conv_kwargs))
            layers.append(nn.MaxPool2d(kernel_size=2))
            for C_n, C_np1 in itertools.pairwise(self.hiddens):
                layers.append(ConvBlock(C_n, C_np1, block_size, kernel_size, decoder_block=False, **conv_kwargs))
                layers.append(nn.MaxPool2d(kernel_size=2))
        else:
            for C_np1, C_n in itertools.pairwise(self.hiddens[::-1]):
                layers.append(nn.Upsample(scale_factor=2, mode='bilinear'))
                layers.append(ConvBlock(C_np1, C_n, block_size, kernel_size, decoder_block=True, **conv_kwargs))
            layers.append(nn.Upsample(scale_factor=2, mode='bilinear'))
            layers.append(ConvBlock(self.hiddens[0], self.C, block_size, kernel_size, decoder_block=True, **conv_kwargs))
        return nn.Sequential(*layers)

    def forward(self, x):
        if self.is_encoder:
            out = self.layers(x)
            # Flatten and apply linear layer
            out = out.view(out.size(0), -1)
            return self.linear(out)
        else:
            # Apply linear layer and unflatten
            out = self.linear(x)
            out = out.view(-1, self.hiddens[-1], self.H_out, self.W_out)
            return self.layers(out)#.view(out.size(0), -1, self.C, self.H, self.W)


class ConvEncoder(BaseEncoderDecoder):
    def __init__(self, C, H, W, latent_dim, hiddens, block_size=1, kernel_size=3, **conv_kwargs):
        """
        Encoder module for encoding input into a latent representation.

        Parameters:
            See BaseEncoderDecoder.
        """
        super().__init__(C, H, W, latent_dim, hiddens, block_size, kernel_size, is_encoder=True, **conv_kwargs)


class ConvDecoder(BaseEncoderDecoder):
    def __init__(self, C, H, W, latent_dim, hiddens, block_size=1, kernel_size=3, **conv_kwargs):
        """
        Decoder module for reconstructing input from a latent representation.

        Parameters:
            See BaseEncoderDecoder.
        """
        super().__init__(C, H, W, latent_dim, hiddens, block_size, kernel_size, is_encoder=False, **conv_kwargs)
        