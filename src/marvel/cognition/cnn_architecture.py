"""Small, inspectable eye-state network used for both training and inference."""

from torch import nn


class EyeCNN(nn.Module):
    """Map ``(N, 1, 32, 64)`` grayscale eye crops to one closure logit per eye."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(64, 1)

    def forward(self, x):
        return self.head(self.features(x).flatten(1)).squeeze(1)
