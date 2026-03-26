from typing import Any, Dict, Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch_ecg.components.trainer import BaseTrainer
from torch_ecg.utils.utils_nn import default_collate_fn as collate_fn
from tqdm.auto import tqdm

from dataset import CINC2026Dataset

__all__ = [
    "CINC2026Trainer",
]


class CINC2026Trainer(BaseTrainer):
    """Trainer for the CinC2026 challenge."""

    __name__ = "CINC2026Trainer"

    def __init__(
        self,
        model: nn.Module,
        model_config: dict,
        train_config: dict,
        device: Optional[torch.device] = None,
        lazy: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model=model,
            dataset_cls=CINC2026Dataset,
            model_config=model_config,
            train_config=train_config,
            device=device,
            lazy=lazy,
        )

    def _setup_dataloaders(
        self,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
    ) -> None:
        if train_dataset is None:
            train_dataset = self.dataset_cls(config=self.train_config, training=True)

        # num_workers depends on the environment
        num_workers = 4 if (self.device and self.device.type == "cuda") else 0

        self.train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
        )

        if val_dataset is not None:
            self.val_loader = DataLoader(
                dataset=val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
                collate_fn=collate_fn,
            )
        else:
            self.val_loader = None

    def train_one_epoch(self, pbar: tqdm) -> None:
        self.model.train()
        for epoch_step, input_tensors in enumerate(self.train_loader):
            self.global_step += 1

            # input_tensors keys should match model forward requirements
            # e.g., signals, channel_ids, demographics, labels
            input_tensors = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in input_tensors.items()}

            out_tensors = self.model(input_tensors)
            loss = out_tensors["ci_loss"]

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if self.scheduler:
                self.scheduler.step()

            self.epoch_loss += loss.item()

            if self.global_step % self.train_config.get("log_step", 20) == 0:
                pbar.set_postfix(loss=loss.item())
                self.log_manager.log_metrics(
                    metrics={"loss": loss.item()},
                    step=self.global_step,
                    epoch=self.epoch,
                    part="train",
                )

            # Update pbar with batch size
            batch_size = next(iter(input_tensors.values())).shape[0] if input_tensors else 0
            pbar.update(batch_size)

    @torch.no_grad()
    def evaluate(self, data_loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        all_losses = []
        for input_tensors in data_loader:
            input_tensors = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in input_tensors.items()}
            out_tensors = self.model(input_tensors)
            all_losses.append(out_tensors["ci_loss"].item())

        avg_loss = np.mean(all_losses)
        self.model.train()
        return {"ci_loss": avg_loss}
