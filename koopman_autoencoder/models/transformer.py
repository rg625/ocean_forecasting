import torch
from torch import nn
from tensordict import TensorDict
from typing import Union, Optional, Dict
from einops import rearrange
from einops.layers.torch import Rearrange
from .autoencoder import KoopmanOutput


class ViT(nn.Module):
    """
    A pure autoregressive ViT baseline that maps an input sequence to an output sequence,
    ignoring latent space mechanics entirely.
    """

    def __init__(
        self,
        data_variables: Dict[str, int],
        input_frames: int = 2,
        height: int = 64,
        width: int = 64,
        patch_size: int = 8,
        embed_dim: int = 256,
        depth: int = 6,
        heads: int = 8,
        cond_embedding_dim: Optional[int] = 64,
        **kwargs,  # Absorbs KAE-specific args from your config
    ):
        super().__init__()
        self.data_variables = data_variables
        self.input_frames = input_frames
        self.total_input_channels = sum(self.data_variables.values())

        assert (
            height % patch_size == 0 and width % patch_size == 0
        ), "Dimensions must be divisible by patch_size"
        self.num_patches = (height // patch_size) * (width // patch_size)

        # 1. Spatiotemporal Patching (Channel Stacking)
        in_channels = self.total_input_channels * self.input_frames
        self.patch_embed = nn.Sequential(
            nn.Conv2d(
                in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
            ),
            Rearrange("b c h w -> b (h w) c"),
        )
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, embed_dim))

        # 2. Conditioning Token
        self.cond_proj = (
            nn.Linear(cond_embedding_dim, embed_dim) if cond_embedding_dim else None
        )

        # 3. Transformer backbone
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=heads,
            dim_feedforward=embed_dim * 4,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        # 4. Unpatching Head (Predicts exactly ONE future timestep)
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, patch_size * patch_size * self.total_input_channels),
            Rearrange(
                "b (h w) (p1 p2 c) -> b c (h p1) (w p2)",
                h=height // patch_size,
                p1=patch_size,
                p2=patch_size,
            ),
        )

    def _get_history_context(self, x: TensorDict) -> torch.Tensor:
        """Extracts the last `input_frames` and stacks them into the channel dimension."""
        tensors = []
        for var, channels in self.data_variables.items():
            var_data = x[var][:, -self.input_frames :]
            if channels == 1 and var_data.ndim == 4:
                var_data = var_data.unsqueeze(2)
            tensors.append(var_data)

        stacked = torch.cat(tensors, dim=2)  # [B, Frames, C, H, W]
        return rearrange(stacked, "b f c h w -> b (f c) h w")

    def forward(
        self,
        x: TensorDict,
        seq_length: Union[int, torch.Tensor],
        cond_future: Optional[torch.Tensor] = None,
    ) -> KoopmanOutput:

        seq_len_int = (
            int(seq_length.view(-1)[0].item())
            if isinstance(seq_length, torch.Tensor)
            else int(seq_length)
        )
        b = x[list(self.data_variables.keys())[0]].shape[0]
        device = x[list(self.data_variables.keys())[0]].device

        # Initialize context buffer
        current_context = self._get_history_context(x)

        # Setup conditioning
        cond_target = x.get("cond_target", cond_future)
        if cond_target is None and x.get("cond_input") is not None:
            cond_target = (
                x.get("cond_input")[:, -1].unsqueeze(1).repeat(1, seq_len_int, 1)
            )

        x_preds_list = []

        # Autoregressive Rollout
        for frame in range(seq_len_int):
            patches = self.patch_embed(current_context) + self.pos_embedding

            if self.cond_proj is not None and cond_target is not None:
                cond_tokens = self.cond_proj(cond_target[:, frame]).unsqueeze(1)
                tokens = torch.cat((cond_tokens, patches), dim=1)
            else:
                tokens = patches

            # Forward
            out_tokens = self.transformer(tokens)
            spatial_tokens = (
                out_tokens[:, 1:] if self.cond_proj is not None else out_tokens
            )

            # Predict Next Frame [B, C, H, W]
            next_frame = self.head(spatial_tokens)
            x_preds_list.append(next_frame)

            # Shift the context window for the next autoregressive step
            if frame < seq_len_int - 1:
                # Drop oldest frame's channels, append new frame's channels
                current_context = torch.cat(
                    (current_context[:, self.total_input_channels :], next_frame), dim=1
                )

        # Reconstruct the output into a TensorDict
        if seq_len_int > 0:
            stacked_preds = torch.stack(x_preds_list, dim=1)  # [B, SeqLen, C, H, W]

            pred_dict = {}
            current_idx = 0
            for var, channels in self.data_variables.items():
                var_tensor = stacked_preds[:, :, current_idx : current_idx + channels]
                pred_dict[var] = var_tensor.squeeze(2) if channels == 1 else var_tensor
                current_idx += channels

            x_preds = TensorDict(pred_dict, batch_size=[b, seq_len_int])
        else:
            x_preds = TensorDict({}, batch_size=[b, 0])

        # Create dummy tensors for the fields your loss function might expect but ViT doesn't use
        dummy_z = torch.zeros(b, seq_len_int, 1, device=device)
        dummy_recon = TensorDict(
            {
                k: torch.zeros_like(v[:, -1:])
                for k, v in x.items()
                if k in self.data_variables
            },
            batch_size=[b, 1],
        )

        return KoopmanOutput(
            x_recon=dummy_recon,  # Dummy payload
            x_preds=x_preds,  # The actual predictions
            z_preds=dummy_z,  # Dummy payload
            reynolds=None,
            disturbed_latents=None,
            dz_dt=None,
            dz_dt_disturbed=None,
        )
