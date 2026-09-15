from pathlib import Path
import json
import numpy as np
import torch

MODEL_DIR = Path(__file__).resolve().parent / "model"

config = json.loads(
    (MODEL_DIR / "model_config.json").read_text(encoding="utf-8")
)

print("Architecture:", config["architecture"])

mean = np.load(MODEL_DIR / "feature_mean.npy", allow_pickle=False)
std = np.load(MODEL_DIR / "feature_std.npy", allow_pickle=False)

print("Feature mean shape:", mean.shape)
print("Feature std shape:", std.shape)

print("Loading saved weights...")
checkpoint = torch.load(
    MODEL_DIR / "pytorch_model.bin",
    map_location="cpu",
    weights_only=True,
)

if not isinstance(checkpoint, dict):
    raise TypeError("The checkpoint is not a dictionary.")

# Some checkpoints wrap the weights in another dictionary.
if "model_state_dict" in checkpoint:
    weights = checkpoint["model_state_dict"]
elif "state_dict" in checkpoint:
    weights = checkpoint["state_dict"]
else:
    weights = checkpoint

print("\nFirst 10 weight names:")
for name in list(weights)[:10]:
    print(name)

print("\nCustom layer weights:")
for name, value in weights.items():
    if any(part in name for part in (
        "context_projection", "feature_projection",
        "gate.", "fusion.", "classifier"
    )):
        if isinstance(value, torch.Tensor):
            print(name, tuple(value.shape))

print("\nCheckpoint inspection complete.")