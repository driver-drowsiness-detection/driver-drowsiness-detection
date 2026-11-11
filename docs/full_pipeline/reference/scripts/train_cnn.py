"""python scripts/train_cnn.py prepared.csv models/eye.pt 10 cpu"""
import random
import sys
from pathlib import Path
from marvel.training.dataset import read_manifest, binary_rows, EyeDataset


def main():
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from marvel.cognition.cnn_architecture import EyeCNN
    from marvel.cognition.cnn_inference import sha256
    from marvel.training.transforms import PREPROCESS
    manifest, output, epochs, device = sys.argv[1:]
    if int(epochs) <= 0:
        raise ValueError("Epoch count must be positive")
    if Path(output).exists():
        raise FileExistsError(output)
    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)
    rows = read_manifest(manifest)
    loaders = {s: DataLoader(EyeDataset(binary_rows(rows, s)), batch_size=32,
                           shuffle=s == "train", num_workers=0) for s in ("train", "validation")}
    model = EyeCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    best, history = float("inf"), []
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(int(epochs)):
        metrics = {"epoch": epoch+1}
        for split, loader in loaders.items():
            model.train(split == "train")
            total, count = 0.0, 0
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                with torch.set_grad_enabled(split == "train"):
                    logits = model(x)
                    loss = loss_fn(logits, y)
                    if split == "train":
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        optimizer.step()
                total += loss.item()*len(y)
                count += len(y)
            metrics[split+"_loss"] = total/count
        history.append(metrics)
        print(metrics)
        if metrics["validation_loss"] < best:
            best = metrics["validation_loss"]
            torch.save({"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                        "class_map": {"open": 0, "closed": 1}, "preprocess": PREPROCESS,
                        "manifest_sha256": sha256(manifest), "epoch": epoch+1}, output)
    import json
    Path(output+".history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
