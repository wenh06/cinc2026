from typing import Literal

import torch
import torch.nn as nn


class DemographicEncoder(nn.Module):
    """
    Encodes demographic data (Age, Sex, BMI, etc.) and provides fusion mechanisms.
    Adapted from CINC2025 style.
    """

    def __init__(
        self,
        dem_input_dim: int,
        feature_dim: int,
        mode: Literal["concat", "film"] = "concat",
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.mode = mode
        self.encoder = nn.Sequential(
            nn.Linear(dem_input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        if mode == "concat":
            self.output_layer = nn.Linear(hidden_dim, feature_dim)
        elif mode == "film":
            self.gain = nn.Linear(hidden_dim, feature_dim)
            self.bias = nn.Linear(hidden_dim, feature_dim)

    def forward(self, x: torch.Tensor):
        # x shape: (batch, dem_input_dim)
        feat = self.encoder(x)
        if self.mode == "concat":
            return self.output_layer(feat)
        elif self.mode == "film":
            return self.gain(feat), self.bias(feat)

    def modulate_features(self, x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
        """Apply FiLM modulation: x = x * scale + shift"""
        return x * scale + shift


class SignalEncoder(nn.Module):
    """
    A simple 1D-CNN encoder to map raw signal segments to a latent vector.
    """

    def __init__(self, in_channels: int = 1, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, out_dim, kernel_size=3, stride=2, padding=1),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )

    def forward(self, x: torch.Tensor):
        # x shape: (batch * N, 1, Length)
        return self.net(x)
