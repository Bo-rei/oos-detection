import torch
import torch.nn as nn


class ProjectionHead(nn.Module):
    def __init__(self, hidden: int, projection_dim: int = 128, bias: bool = False):
        super().__init__()
        width = max(hidden, projection_dim)
        self.net = nn.Sequential(
            nn.Linear(hidden, width, bias=bias),
            nn.ReLU(),
            nn.Linear(width, projection_dim, bias=bias),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
