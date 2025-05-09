from itertools import pairwise
from torch import nn
from models.checkpoint import checkpoint
from einops import rearrange
from torch import Tensor
from typing import List, Union, Tuple, Any


class ConvBlock(nn.Module):
    def __init__(
        self,
        C_in: int,
        C_out: int,
        block_size: int = 1,
        kernel_size: Union[int, Tuple[int, int]] = 3,
        decoder_block: bool = False,
        use_checkpoint: bool = False,  # Add this to toggle checkpointing
        **conv_kwargs: Any,
    ):
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
            use_checkpoint: bool
                If True, enables gradient checkpointing for the block.
            conv_kwargs: dict
                Additional arguments for nn.Conv2d.
        """
        super().__init__()
        self.use_checkpoint = use_checkpoint  # Store the checkpointing flag

        if not decoder_block:
            first_layer = [
                nn.Conv2d(C_in, C_out, kernel_size, **conv_kwargs),
                nn.ReLU(),
            ]
            subsequent_layers = (block_size - 1) * [
                nn.Conv2d(C_out, C_out, kernel_size, **conv_kwargs),
                nn.ReLU(),
            ]
            self.stack = nn.ModuleList([*first_layer, *subsequent_layers])
        else:
            initial_layers = (block_size - 1) * [
                nn.Conv2d(C_in, C_in, kernel_size, **conv_kwargs),
                nn.ReLU(),
            ]
            output_layer = [
                nn.Conv2d(C_in, C_out, kernel_size, **conv_kwargs),
            ]
            self.stack = nn.ModuleList([*initial_layers, *output_layer])

    def forward(self, x: Tensor):
        # Use gradient checkpointing if the flag is enabled
        if self.use_checkpoint:
            return checkpoint(
                self._forward, (x,), self.parameters(), self.use_checkpoint
            )
        else:
            return self._forward(x)

    def _forward(self, x: Tensor):
        for module in self.stack:
            x = module(x)
        return x


class BaseEncoderDecoder(nn.Module):
    def __init__(
        self,
        C: int,
        H: int,
        W: int,
        latent_dim: int,
        hiddens: List[int],
        block_size: int = 1,
        kernel_size: Union[int, Tuple[int, int]] = 3,
        is_encoder: bool = True,
        use_checkpoint: bool = False,  # Add checkpointing flag here
        **conv_kwargs,
    ):
        """
        Base class for both encoder and decoder blocks.

        Parameters:
            C: int
                Number of input/output channels.
            H: int
                Input height dimension.
            W: int
                Input width dimension.
            latent_dim: int
                Latent dimensionality.
            hiddens: list of int
                List of hidden dimensions for each block.
            block_size: int
                Number of convolutional layers in each block.
            kernel_size: int
                Size of the convolutional kernel.
            is_encoder: bool
                Specifies whether the block is an encoder (True) or decoder (False).
            use_checkpoint: bool
                If True, enables gradient checkpointing for convolutional blocks.
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
        self.use_checkpoint = use_checkpoint  # Store the flag

        # Compute the output dimensions after pooling
        self.n_pools = len(hiddens)
        if H % (2**self.n_pools) != 0 or W % (2**self.n_pools) != 0:
            raise ValueError(
                f"Input dimensions (H={H}, W={W}) must be divisible by 2^{self.n_pools} due to pooling."
            )
        self.H_out = H // (2 ** (self.n_pools))
        self.W_out = W // (2 ** (self.n_pools))

        # Define the linear layer
        if is_encoder:
            self.linear = nn.Linear(hiddens[-1] * self.H_out * self.W_out, latent_dim)
        else:
            self.linear = nn.Linear(latent_dim, hiddens[-1] * self.H_out * self.W_out)

        # Build convolutional layers
        self.layers = self._build_layers(block_size, kernel_size, conv_kwargs)

    def _build_layers(
        self,
        block_size: int,
        kernel_size: Union[int, tuple[int, int]],
        conv_kwargs: dict,
    ):
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
            layers.append(
                ConvBlock(
                    self.C,
                    self.hiddens[0],
                    block_size,
                    kernel_size,
                    decoder_block=False,
                    use_checkpoint=self.use_checkpoint,  # Pass the flag
                    **conv_kwargs,
                )
            )
            layers.append(nn.MaxPool2d(kernel_size=2))
            for C_n, C_np1 in pairwise(self.hiddens):
                layers.append(
                    ConvBlock(
                        C_n,
                        C_np1,
                        block_size,
                        kernel_size,
                        decoder_block=False,
                        use_checkpoint=self.use_checkpoint,  # Pass the flag
                        **conv_kwargs,
                    )
                )
                layers.append(nn.MaxPool2d(kernel_size=2))
        else:
            for C_np1, C_n in pairwise(self.hiddens[::-1]):
                layers.append(nn.Upsample(scale_factor=2, mode="bilinear"))
                layers.append(
                    ConvBlock(
                        C_np1,
                        C_n,
                        block_size,
                        kernel_size,
                        decoder_block=True,
                        use_checkpoint=self.use_checkpoint,  # Pass the flag
                        **conv_kwargs,
                    )
                )
            layers.append(nn.Upsample(scale_factor=2, mode="bilinear"))
            layers.append(
                ConvBlock(
                    self.hiddens[0],
                    self.C,
                    block_size,
                    kernel_size,
                    decoder_block=True,
                    use_checkpoint=self.use_checkpoint,  # Pass the flag
                    **conv_kwargs,
                )
            )
        return nn.Sequential(*layers)

    def forward(self, x: Tensor):
        if self.is_encoder:
            out = self.layers(x)
            # Flatten and apply linear layer
            out = rearrange(out, "b c h w -> b (c h w)")
            return self.linear(out)
        else:
            # Apply linear layer and unflatten
            out = self.linear(x)
            out = rearrange(
                out,
                "b (c h w) -> b c h w",
                c=self.hiddens[-1],
                h=self.H_out,
                w=self.W_out,
            )
            return self.layers(out)


class ConvEncoder(BaseEncoderDecoder):
    def __init__(
        self,
        C: int,
        H: int,
        W: int,
        latent_dim: int,
        hiddens: List[int],
        block_size: int = 1,
        kernel_size: Union[int, Tuple[int, int]] = 3,
        use_checkpoint: bool = False,
        **conv_kwargs,
    ):
        super().__init__(
            C,
            H,
            W,
            latent_dim,
            hiddens,
            block_size,
            kernel_size,
            is_encoder=True,
            use_checkpoint=use_checkpoint,  # Pass the flag
            **conv_kwargs,
        )


class ConvDecoder(BaseEncoderDecoder):
    def __init__(
        self,
        C: int,
        H: int,
        W: int,
        latent_dim: int,
        hiddens: List[int],
        block_size: int = 1,
        kernel_size: Union[int, Tuple[int, int]] = 3,
        use_checkpoint: bool = False,
        **conv_kwargs,
    ):
        super().__init__(
            C,
            H,
            W,
            latent_dim,
            hiddens,
            block_size,
            kernel_size,
            is_encoder=False,
            use_checkpoint=use_checkpoint,  # Pass the flag
            **conv_kwargs,
        )
