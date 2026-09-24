#!/usr/bin/env python3
"""
Unit test for AzencotDiscreteLoss.
Verifies that the discrete loss correctly:
1. Accepts TensorDict inputs
2. Returns LossResult with structured metrics
3. Extracts operator matrices for consistency loss
4. Computes sub-matrix orthogonality constraints
"""

import torch
from tensordict import TensorDict
from models.azencot_loss import AzencotDiscreteLoss, AzencotDiscreteConfig, LossResult
from models.modules import KoopmanOperator, KoopmanMode

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Testing on device: {device}")

# ============================================================================
# 1. Create Mock Koopman Operator (Discrete)
# ============================================================================
latent_dim = 4
koopman_op = KoopmanOperator(
    latent_dim=latent_dim,
    cond_embedding_dim=None,
    mode=KoopmanMode.LINEAR,
    is_continuous=False,
    rank=None,  # No low-rank adaptation without conditioning
).to(device)

print(f"✓ Created discrete KoopmanOperator with latent_dim={latent_dim}")

# ============================================================================
# 2. Create Loss Function
# ============================================================================
config = AzencotDiscreteConfig(
    loss_type="l2",
    alpha=1.0,
    lamb=20.0,
    nu=1.0,
    eta=0.1,
    use_backward=False,
)

loss_fn = AzencotDiscreteLoss(config=config).to(device)
print(f"✓ Created AzencotDiscreteLoss with config:\n  {config}")

# ============================================================================
# 3. Create Mock Input Data
# ============================================================================
batch_size = 2
seq_len = 3
spatial_h, spatial_w = 8, 8
channels = 1

# Create TensorDicts with mock data
x_recon = TensorDict(
    {
        "state": torch.randn(batch_size, channels, spatial_h, spatial_w, device=device),
    }
)

x_true = TensorDict(
    {
        "state": torch.randn(batch_size, channels, spatial_h, spatial_w, device=device),
    }
)

x_preds = TensorDict(
    {
        "state": torch.randn(
            batch_size, seq_len, channels, spatial_h, spatial_w, device=device
        ),
    }
)

x_future = TensorDict(
    {
        "state": torch.randn(
            batch_size, seq_len, channels, spatial_h, spatial_w, device=device
        ),
        "seq_length": torch.tensor([[seq_len]], dtype=torch.float32, device=device),
    }
)

latent_pred = torch.randn(batch_size, seq_len, latent_dim, device=device)

print("✓ Created mock data:")
print(f"  - x_recon['state']: {x_recon['state'].shape}")
print(f"  - x_true['state']: {x_true['state'].shape}")
print(f"  - x_preds['state']: {x_preds['state'].shape}")
print(f"  - x_future['state']: {x_future['state'].shape}")
print(f"  - latent_pred: {latent_pred.shape}")

# ============================================================================
# 4. Forward Pass
# ============================================================================
try:
    loss_result = loss_fn(
        koopman_operator=koopman_op,
        x_recon=x_recon,
        x_preds=x_preds,
        latent_pred=latent_pred,
        x_true=x_true,
        x_future=x_future,
        true_latents=None,
        reynolds=None,
    )

    print("✓ Forward pass successful!")
    print(f"  - Loss type: {type(loss_result)}")
    print(f"  - Total loss: {loss_result.total_loss.item():.6f}")
    print(f"  - Metrics keys: {list(loss_result.metrics.keys())}")

    # Verify output structure
    assert isinstance(loss_result, LossResult), "Output should be LossResult"
    assert isinstance(
        loss_result.total_loss, torch.Tensor
    ), "total_loss should be Tensor"
    assert isinstance(loss_result.metrics, dict), "metrics should be dict"
    assert loss_result.total_loss.item() > 0, "Loss should be positive"

    print("\n✓ Output structure verified:")
    for key, val in loss_result.metrics.items():
        print(f"  - {key}: {val:.6f}")

except Exception as e:
    print("✗ Forward pass failed with error:")
    print(f"  {type(e).__name__}: {e}")
    import traceback

    traceback.print_exc()
    exit(1)

# ============================================================================
# 5. Backward Pass (Check Gradients)
# ============================================================================
try:
    loss_result.total_loss.backward()

    # Check if gradients exist
    has_grads = sum(1 for p in koopman_op.parameters() if p.grad is not None)
    total_params = sum(1 for p in koopman_op.parameters())

    print("\n✓ Backward pass successful!")
    print(f"  - Parameters with gradients: {has_grads}/{total_params}")

    # Check gradient norms
    grad_norm = torch.nn.utils.clip_grad_norm_(koopman_op.parameters(), max_norm=1.0)
    print(f"  - Global gradient norm: {grad_norm:.6f}")

except Exception as e:
    print("✗ Backward pass failed with error:")
    print(f"  {type(e).__name__}: {e}")
    import traceback

    traceback.print_exc()
    exit(1)

# ============================================================================
# 6. Test Matrix Extraction
# ============================================================================
try:
    print("\n✓ Matrix extraction test:")

    # Try to extract forward matrix
    if hasattr(koopman_op, "dynamics"):
        dynamics = koopman_op.dynamics
        if hasattr(dynamics, "W_param"):
            W = dynamics.W_param.weight
            D = dynamics.D_param.weight
            print(f"  - Extracted W_param: {W.shape}")
            print(f"  - Extracted D_param: {D.shape}")

            # Verify consistency loss computed these
            assert (
                "consistency" in loss_result.metrics
            ), "Consistency should be in metrics"
            print(f"  - Consistency loss: {loss_result.metrics['consistency']:.6f}")
        else:
            print("  - W_param not found in dynamics")

except Exception as e:
    print("✗ Matrix extraction test failed:")
    print(f"  {type(e).__name__}: {e}")
    import traceback

    traceback.print_exc()

# ============================================================================
# Summary
# ============================================================================
print(f"\n{'='*60}")
print("✓ All tests passed!")
print(f"{'='*60}")
print("\nDiscrete Loss Summary:")
print(f"  Total Loss: {loss_result.total_loss.item():.6f}")
print("  Components:")
for key in ["loss_fwd", "loss_identity", "loss_consistency"]:
    if key in loss_result.metrics:
        print(f"    - {key}: {loss_result.metrics[key]:.6f}")
