"""Train with: python scripts/train_cnn.py prepared.csv models/eye.pt 10 cpu."""

import random
import math
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from marvel.training.dataset import EyeDataset, binary_rows, read_manifest


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 4:
        raise SystemExit("Usage: train_cnn.py PREPARED.csv OUTPUT.pt EPOCHS DEVICE")
    manifest, output, epochs_text, device = argv
    epochs = int(epochs_text)
    if epochs <= 0:
        raise ValueError("Epoch count must be positive")
    output_path = Path(output)
    if output_path.exists() or Path(str(output_path) + ".history.json").exists():
        raise FileExistsError(output_path)

    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from marvel.cognition.cnn_architecture import EyeCNN
    from marvel.cognition.cnn_inference import sha256
    from marvel.training.transforms import PREPROCESS
    from marvel.training.audit import audit_manifest

    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)
    audit = audit_manifest(manifest)
    rows = read_manifest(manifest)
    loaders = {
        split: DataLoader(EyeDataset(binary_rows(rows, split)), batch_size=32,
                          shuffle=split == "train", num_workers=0)
        for split in ("train", "validation")
    }
    model = EyeCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    best_loss, history = float("inf"), []
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        metrics: dict[str, float | int] = {"epoch": epoch + 1}
        for split, loader in loaders.items():
            model.train(split == "train")
            total, count = 0.0, 0
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                with torch.set_grad_enabled(split == "train"):
                    logits = model(x)
                    loss = loss_fn(logits, y)
                    if not math.isfinite(loss.item()):
                        raise ValueError("Nonfinite loss; no valid model can be selected")
                    if split == "train":
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        optimizer.step()
                total += loss.item() * len(y)
                count += len(y)
            metrics[f"{split}_loss"] = total / count
        history.append(metrics)
        print(metrics)
        if metrics["validation_loss"] < best_loss:
            best_loss = float(metrics["validation_loss"])
            torch.save({
                "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                "class_map": {"open": 0, "closed": 1},
                "preprocess": PREPROCESS,
                "manifest_sha256": sha256(manifest),
                "epoch": epoch + 1,
                "seed": 7,
                "training_subjects": sorted({r["subject_id"] for r in rows if r["split"] == "train"}),
                "validation_subjects": sorted({r["subject_id"] for r in rows if r["split"] == "validation"}),
                "dataset_audit": audit,
            }, output_path)

    import json
    Path(str(output_path) + ".history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
