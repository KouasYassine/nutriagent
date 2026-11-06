
import torch
from pathlib import Path

for path in sorted(Path(".").glob("*.pth")):
    print(f"\n{'=' * 55}")
    print(f"Checkpoint: {path.name}")

    ckpt = torch.load(
        path,
        map_location="cpu",
        weights_only=True
    )

    print("Type:", type(ckpt).__name__)

    if isinstance(ckpt, dict):
        print("Top-level keys:", list(ckpt.keys())[:20])

        # Check for common checkpoint formats
        if "model_state_dict" in ckpt:
            state = ckpt["model_state_dict"]
        elif "state_dict" in ckpt:
            state = ckpt["state_dict"]
        else:
            state = ckpt

        print("\nLast model parameters:")
        for name, value in list(state.items())[-8:]:
            if isinstance(value, torch.Tensor):
                print(name, tuple(value.shape))
