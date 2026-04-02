from copy import deepcopy
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from torch_ecg.cfg import CFG
from torch_ecg.models._nets import MultiConv
from torch_ecg.models.loss import setup_criterion
from torch_ecg.utils.utils_data import one_hot_encode
from torch_ecg.utils.utils_nn import CkptMixin, SizeMixin

from cfg import ModelCfg
from outputs import CINC2026Outputs

from .building_blocks import DemographicEncoder

__all__ = ["MultiBranchNet"]


class MultiBranchNet(nn.Module, SizeMixin, CkptMixin):
    """
    Multi-branch architecture grouping channels by modality (EEG, ECG, etc.).
    """

    __name__ = "MultiBranchNet"

    def __init__(self, config: Optional[CFG] = None, **kwargs: Any) -> None:
        super().__init__()
        if config is None:
            self.config = deepcopy(ModelCfg.multibranch)
        else:
            self.config = deepcopy(config)
        self.config.update(kwargs)

        self.classes = self.config.classes
        self.n_classes = len(self.classes)
        self.d_model = self.config.d_model
        self.modalities = self.config.modalities

        self.encoders = nn.ModuleDict()
        for mod in self.modalities:
            # Using torch_ecg MultiConv
            self.encoders[mod] = nn.Sequential(
                MultiConv(
                    in_channels=1,
                    out_channels=[32, 64, self.d_model],
                    filter_lengths=[15, 7, 3],
                    subsample_lengths=[2, 2, 2],
                ),
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
            )

        self.fusion_attn = nn.MultiheadAttention(
            embed_dim=self.d_model, num_heads=self.config.get("nhead", 4), batch_first=True
        )

        if hasattr(self.config, "dem_encoder") and self.config.dem_encoder.enable:
            self.dem_encoder = DemographicEncoder(
                dem_input_dim=self.config.dem_encoder.input_dim,
                feature_dim=self.d_model,
                mode=self.config.dem_encoder.mode,
                hidden_dim=self.config.dem_encoder.hidden_dim,
            )
        else:
            self.dem_encoder = None

        self.clf = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.d_model // 2, self.n_classes),
        )
        self.softmax = nn.Softmax(dim=-1)
        self.criterion = setup_criterion(self.config.criterion, **self.config.get("criterion_kw", {}))

    def forward(self, input_tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        batch_size = input_tensors["demographics"].shape[0]
        device, dtype = self.device, self.dtype
        mod_features = []

        for mod in self.modalities:
            if mod in input_tensors and input_tensors[mod].numel() > 0:
                signals = input_tensors[mod].to(device).to(dtype)
                B, N, L = signals.shape
                feats = self.encoders[mod](signals.view(B * N, 1, L))
                mod_feat = feats.view(B, N, self.d_model).mean(dim=1)
            else:
                mod_feat = torch.zeros((batch_size, self.d_model), device=device, dtype=dtype)
            mod_features.append(mod_feat)

        x = torch.stack(mod_features, dim=1)
        attn_out, _ = self.fusion_attn(x, x, x)
        mod_summary = attn_out.mean(dim=1)

        if self.dem_encoder is not None:
            demographics = input_tensors["demographics"].to(device).to(dtype)
            if self.config.dem_encoder.mode == "film":
                scale, shift = self.dem_encoder(demographics)
                mod_summary = self.dem_encoder.modulate_features(mod_summary, scale, shift)
            else:
                dem_feats = self.dem_encoder(demographics)
                mod_summary = mod_summary + dem_feats

        ci_logits = self.clf(mod_summary)
        ci_prob = self.softmax(ci_logits)
        cognitive_impairment = torch.argmax(ci_prob, dim=-1)

        ci_loss = None
        if "labels" in input_tensors:
            labels = input_tensors["labels"].to(device)
            if labels.ndim == 1 and self.config.criterion != "CrossEntropyLoss":
                oh_labels = torch.from_numpy(one_hot_encode(labels.cpu().numpy(), self.n_classes)).to(device).to(dtype)
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
    def inference(self, input_tensors: Dict[str, Union[np.ndarray, torch.Tensor]]) -> CINC2026Outputs:
        self.eval()
        processed_input = {}
        for k, v in input_tensors.items():
            if isinstance(v, np.ndarray):
                processed_input[k] = torch.from_numpy(v)
            else:
                processed_input[k] = v

        output_dict = self.forward(processed_input)
        return CINC2026Outputs.from_dict(output_dict)
