from copy import deepcopy
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from torch_ecg.cfg import CFG
from torch_ecg.models._nets import MultiConv  # Correct component from torch_ecg
from torch_ecg.models.loss import setup_criterion
from torch_ecg.utils.utils_data import one_hot_encode
from torch_ecg.utils.utils_nn import CkptMixin, SizeMixin

from cfg import ModelCfg
from outputs import CINC2026Outputs

from .building_blocks import DemographicEncoder

__all__ = ["ChannelTransformer"]


class ChannelTransformer(nn.Module, SizeMixin, CkptMixin):
    """
    Channel-as-Tokens Transformer for CinC 2026.
    """

    __name__ = "ChannelTransformer"

    def __init__(self, config: Optional[CFG] = None, **kwargs: Any) -> None:
        super().__init__()
        if config is None:
            self.config = deepcopy(ModelCfg.transformer)
        else:
            self.config = deepcopy(config)
        self.config.update(kwargs)

        self.classes = self.config.classes
        self.n_classes = len(self.classes)
        self.d_model = self.config.d_model

        # Using torch_ecg's MultiConv as the backbone for each channel
        # It takes lists of parameters for each layer
        self.signal_encoder = nn.Sequential(
            MultiConv(
                in_channels=1,
                out_channels=[32, 64, self.d_model],
                filter_lengths=[15, 7, 3],
                subsample_lengths=[2, 2, 2],
                # MultiConv automatically handles BatchNorm and Activation based on its default or passed config
            ),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )

        self.channel_embedding = nn.Embedding(self.config.max_channels, self.d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.config.nhead,
            dim_feedforward=self.config.dim_feedforward,
            dropout=self.config.dropout,
            activation=self.config.activation,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.config.num_layers)

        if hasattr(self.config, "dem_encoder") and self.config.dem_encoder.enable:
            self.dem_encoder = DemographicEncoder(
                dem_input_dim=self.config.dem_encoder.input_dim,
                feature_dim=self.d_model,
                mode=self.config.dem_encoder.mode,
                hidden_dim=self.config.dem_encoder.hidden_dim,
            )
        else:
            self.dem_encoder = None

        self.clf = nn.Linear(self.d_model, self.n_classes)
        self.softmax = nn.Softmax(dim=-1)
        self.criterion = setup_criterion(self.config.criterion, **self.config.get("criterion_kw", {}))

    def forward(self, input_tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        signals = input_tensors["signals"].to(self.device).to(self.dtype)
        channel_ids = input_tensors["channel_ids"].to(self.device).long()
        demographics = input_tensors["demographics"].to(self.device).to(self.dtype)

        batch_size, num_channels, sig_len = signals.shape
        x = signals.view(batch_size * num_channels, 1, sig_len)
        channel_features = self.signal_encoder(x).view(batch_size, num_channels, self.d_model)
        x = channel_features + self.channel_embedding(channel_ids)
        x = self.transformer(x)
        pooled_features = x.mean(dim=1)

        if self.dem_encoder is not None:
            if self.config.dem_encoder.mode == "film":
                scale, shift = self.dem_encoder(demographics)
                pooled_features = self.dem_encoder.modulate_features(pooled_features, scale, shift)
            else:
                dem_feats = self.dem_encoder(demographics)
                pooled_features = torch.cat([pooled_features, dem_feats], dim=1)

        ci_logits = self.clf(pooled_features)
        ci_prob = self.softmax(ci_logits)
        cognitive_impairment = torch.argmax(ci_prob, dim=-1)

        ci_loss = None
        if "labels" in input_tensors:
            labels = input_tensors["labels"].to(self.device)
            if labels.ndim == 1 and self.config.criterion != "CrossEntropyLoss":
                oh_labels = (
                    torch.from_numpy(one_hot_encode(labels.cpu().numpy(), self.n_classes)).to(self.device).to(self.dtype)
                )
                ci_loss = self.criterion(ci_logits, oh_labels)
            else:
                ci_loss = self.criterion(ci_logits, labels)

        return {
            "ci_logits": ci_logits,
            "ci_prob": ci_prob,
            "cognitive_impairment": cognitive_impairment,
            "ci_loss": ci_loss,
        }

    @torch.no_grad()
    def inference(
        self,
        signals: Union[np.ndarray, torch.Tensor],
        channel_ids: Union[np.ndarray, torch.Tensor],
        demographics: Union[np.ndarray, torch.Tensor],
    ) -> CINC2026Outputs:
        self.eval()
        if isinstance(signals, np.ndarray):
            signals = torch.from_numpy(signals)
        if isinstance(channel_ids, np.ndarray):
            channel_ids = torch.from_numpy(channel_ids)
        if isinstance(demographics, np.ndarray):
            demographics = torch.from_numpy(demographics)
        if signals.ndim == 2:
            signals, channel_ids, demographics = signals.unsqueeze(0), channel_ids.unsqueeze(0), demographics.unsqueeze(0)

        output_dict = self.forward({"signals": signals, "channel_ids": channel_ids, "demographics": demographics})
        return CINC2026Outputs.from_dict(output_dict)
