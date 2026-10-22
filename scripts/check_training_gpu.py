"""Verify CUDA training and checkpoint portability using synthetic inputs only."""

import json
import platform
import sys
import tempfile
from pathlib import Path
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main():
    import numpy as np
    import torch
    from marvel.cognition.cnn_architecture import EyeCNN
    from marvel.cognition.cnn_inference import CNN
    from marvel.training.transforms import PREPROCESS

    if not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable: use the GPU environment and check the NVIDIA driver.")
    torch.manual_seed(7)
    model = EyeCNN().to("cuda")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    x = torch.rand(32, 1, 32, 64, device="cuda")
    y = (torch.arange(32, device="cuda") % 2).float()
    before = model.head.weight.detach().clone()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = perf_counter()
    for _ in range(5):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(x), y)
        if not torch.isfinite(loss).item():
            raise RuntimeError("Nonfinite loss")
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()
    elapsed = perf_counter() - start
    if torch.equal(before, model.head.weight):
        raise RuntimeError("Optimizer did not change model weights")

    # Temporary synthetic weights must never be mistaken for a trained eye model.
    with tempfile.TemporaryDirectory(prefix="marvel_gpu_smoke_") as temporary:
        checkpoint = Path(temporary) / "synthetic_only.pt"
        torch.save({
            "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "class_map": {"open": 0, "closed": 1},
            "preprocess": PREPROCESS,
        }, checkpoint)
        crop = np.zeros((32, 64, 3), dtype=np.uint8)
        cpu_value = CNN(checkpoint, device="cpu").logit(crop)
        gpu_value = CNN(checkpoint, device="cuda").logit(crop)
        if not np.isclose(cpu_value, gpu_value, atol=1e-5, rtol=1e-4):
            raise RuntimeError("CPU and GPU checkpoint predictions disagree")
    print(json.dumps({
        "status": "passed", "evidence": "synthetic software smoke test; no accuracy claim",
        "python": platform.python_version(), "torch": str(torch.__version__),
        "cuda_runtime": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
        "gpu_memory_mib": torch.cuda.get_device_properties(0).total_memory / 2**20,
        "batch_size": 32, "training_steps": 5,
        "elapsed_seconds_including_initialization": elapsed,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "checkpoint_cpu_gpu_parity": True,
    }, indent=2))


if __name__ == "__main__":
    main()
