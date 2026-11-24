"""Check that a code tree loads the evaluated TRA checkpoint strictly and print a hash of
its eval-mode outputs, so two trees (original vs audit branch) can be compared."""

from _paths import MAIN_ROOT
import sys
import warnings

import torch
import yaml

repo = sys.argv[1]
sys.path.insert(0, repo)
warnings.simplefilter("ignore")
from tensordict import TensorDict  # noqa: E402
from models.autoencoder import KoopmanAutoencoder  # noqa: E402
from models.networks import TransformerConfig  # noqa: E402

RUN = str(
    MAIN_ROOT
    / "koopman_autoencoder/model_outputs_tra/continous_linear_128/run-20260821_031121"
)
m = yaml.safe_load(open(RUN + "/continous_linear_128_tra.yaml"))
model = KoopmanAutoencoder(
    data_variables={"v_x": 1, "v_y": 1, "p": 1, "rho": 1},
    input_frames=2,
    height=m["height"],
    width=m["width"],
    latent_dim=m["latent_dim"],
    cond_embedding_dim=m["cond_embedding_dim"],
    cond_type=m["cond_type"],
    operator_mode=m["operator_mode"],
    hidden_dims=m["hidden_dims"],
    transformer_config=TransformerConfig(**m["transformer"]),
    predict_cond=m["predict_cond"],
    disturb_std=None,
    is_continuous=True,
    operator_type="continuous",
    rank=m["rank"],
    spectral=m["spectral"],
    cond_expansion_type="Ma",
    use_attention=m["use_attention"],
    **m["conv_kwargs"],
)
sd = torch.load(
    RUN + "/checkpoints/epoch_199.pth", map_location="cpu", weights_only=False
)
model.load_state_dict(sd["model_state_dict"], strict=True)
model = model.cuda().eval()
g = torch.Generator().manual_seed(0)
x = TensorDict(
    {v: torch.randn(2, 2, 64, 128, generator=g) for v in ["v_x", "v_y", "p", "rho"]},
    batch_size=[2, 2],
)
x["cond_input"] = torch.full((2, 2), 0.64)
with torch.no_grad():
    out = model(x.cuda(), 20)
s = sum(float(out.x_preds[v].double().sum()) for v in ["v_x", "v_y", "p", "rho"])
print(
    f"strict load OK; output checksum {s:.10f}; z checksum {float(out.z_preds.double().sum()):.10f}"
)
