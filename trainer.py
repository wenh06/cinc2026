"""Trainer for the CinC 2026 Challenge.

Subclasses :class:`torch_ecg.components.trainer.BaseTrainer`.  The primary
model is :class:`models.EpochTransformer`, which accepts CAISR epoch-feature
sequences (site-agnostic, available for all dataset splits).

Key design decisions vs the generic BaseTrainer
------------------------------------------------
- No signal augmenters: inputs are already-derived CAISR features.
- ``BCEWithLogitsLoss`` is contained in the model; ``_setup_criterion`` is a
  no-op.
- ``collate_fn`` from ``dataset`` handles variable-length padding.
- Gradient clipping (``train_config.grad_clip``) is applied after each
  backward pass for Transformer numerical stability.
- Primary evaluation metric: AUROC (``train_config.monitor = "auroc"``).
- Per-site AUROC breakdown (S0001 / I0002 / I0006) is logged every epoch
  for domain-shift monitoring.
"""

import argparse
import os
import sys
from copy import deepcopy
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.nn.parallel import DataParallel as DP
from torch.utils.data import DataLoader, Dataset
from torch_ecg.cfg import CFG
from torch_ecg.components.trainer import BaseTrainer
from torch_ecg.utils.misc import str2bool
from tqdm.auto import tqdm

from cfg import ModelCfg, TrainCfg
from dataset import CINC2026Dataset, collate_fn
from models import EpochTransformer

__all__ = ["CINC2026Trainer"]


class CINC2026Trainer(BaseTrainer):
    """Trainer for the CinC 2026 Challenge.

    Parameters
    ----------
    model : nn.Module
        The model to train (typically :class:`models.EpochTransformer`).
    model_config : dict
        Model configuration; stored in checkpoints for reproducibility.
    train_config : dict
        Training configuration.  Must include at minimum:

        - ``db_dir``        path to the CinC 2026 data root
        - ``n_epochs``      total number of training epochs
        - ``batch_size``    mini-batch size
        - ``learning_rate`` initial learning rate
        - ``optimizer``     optimizer name (e.g. ``"adamw_amsgrad"``)
        - ``lr_scheduler``  scheduler name (e.g. ``"one_cycle"``)
        - ``log_step``      how often (in steps) to log training loss
        - ``monitor``       metric key for model selection (default ``"auroc"``)

    device : torch.device, optional
        Training device.  Defaults to the model's current device.
    lazy : bool, default True
        Defer ``DataLoader`` construction until :meth:`train` is called.
    """

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
        # BaseTrainer expects learning_rate; lr is kept as a user-facing alias
        tc = CFG(deepcopy(train_config))
        if "learning_rate" not in tc and "lr" in tc:
            tc["learning_rate"] = tc["lr"]

        super().__init__(
            model=model,
            dataset_cls=CINC2026Dataset,
            model_config=model_config,
            train_config=tc,
            collate_fn=collate_fn,
            device=device,
            lazy=lazy,
        )

    # ── Abstract property implementations ────────────────────────────────

    @property
    def batch_dim(self) -> int:
        return 0

    @property
    def extra_required_train_config_fields(self) -> List[str]:
        return ["db_dir"]

    @property
    def save_prefix(self) -> str:
        model_name = getattr(self._model, "__name__", self._model.__class__.__name__)
        return f"{model_name}-epoch"

    def extra_log_suffix(self) -> str:
        model_name = getattr(self._model, "__name__", self._model.__class__.__name__)
        return f"{model_name}_{self.train_config.optimizer}"

    # ── DataLoader setup ─────────────────────────────────────────────────

    def _setup_dataloaders(
        self,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
    ) -> None:
        """Create train and validation :class:`DataLoader` instances."""
        num_workers = 1 if self.device == torch.device("cpu") else 4

        if train_dataset is None:
            train_dataset = CINC2026Dataset(
                config=self.train_config,
                training=True,
                lazy=True,
            )
        if val_dataset is None:
            val_dataset = CINC2026Dataset(
                config=self.train_config,
                training=False,
                lazy=True,
            )

        self.train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=collate_fn,
        )
        self.val_loader = DataLoader(
            dataset=val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=collate_fn,
        )
        # No train-set validation loader; the balanced training set would give
        # an overly optimistic AUROC estimate.
        self.val_train_loader = None

    # ── Training step ────────────────────────────────────────────────────

    def run_one_step(self, input_tensors: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Forward one batch through the model and return the output dict.

        The dataset emits ``"label"`` (singular) but the model expects
        ``"labels"`` (plural, float).  This method performs the rename and
        dtype conversion, and strips non-tensor string fields.
        """
        tensors: Dict[str, Any] = dict(input_tensors)
        if "label" in tensors and "labels" not in tensors:
            tensors["labels"] = tensors.pop("label").to(self.dtype)
        else:
            tensors.pop("label", None)
        tensors.pop("record_id", None)
        tensors.pop("site_id", None)
        return self.model(tensors)

    def train_one_epoch(self, pbar: tqdm) -> None:
        """Train for one epoch and update *pbar*."""
        for input_tensors in self.train_loader:
            self.global_step += 1
            n_samples = input_tensors["epoch_features"].shape[self.batch_dim]

            out = self.run_one_step(input_tensors)
            loss = out["ci_loss"]

            self.epoch_loss += loss.item()
            self.optimizer.zero_grad()
            loss.backward()

            grad_clip = self.train_config.get("grad_clip", 0.0)
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)

            self.optimizer.step()
            self._update_lr()

            if self.global_step % self.train_config.log_step == 0:
                step_metrics = {"loss": loss.item()}
                if self.scheduler:
                    step_metrics["lr"] = self.scheduler.get_last_lr()[0]
                    pbar.set_postfix(**{"loss (batch)": loss.item(), "lr": self.scheduler.get_last_lr()[0]})
                else:
                    pbar.set_postfix(**{"loss (batch)": loss.item()})
                self.log_manager.log_metrics(
                    metrics=step_metrics,
                    step=self.global_step,
                    epoch=self.epoch,
                    part="train",
                )

            pbar.update(n_samples)

    # ── Evaluation ───────────────────────────────────────────────────────

    @torch.no_grad()
    def evaluate(self, data_loader: DataLoader) -> Dict[str, float]:
        """Evaluate the model and return a metrics dict.

        Primary metric is ``"auroc"`` (area under the ROC curve).  Additional
        metrics include ``"auprc"`` and per-site AUROC values when site IDs
        are present in the batch.

        Parameters
        ----------
        data_loader : DataLoader
            Loader for the evaluation split.

        Returns
        -------
        dict
            ``{"auroc": float, "auprc": float, "auroc_<site>": float, ...}``
        """
        self.model.eval()

        all_probs: List[float] = []
        all_labels: List[int] = []
        all_site_ids: List[str] = []
        all_record_ids: List[str] = []

        with tqdm(
            total=len(data_loader.dataset),
            desc="Evaluation",
            unit="record",
            dynamic_ncols=True,
            mininterval=1.0,
            leave=False,
        ) as pbar:
            for batch in data_loader:
                labels = batch["label"].cpu().numpy()
                site_ids: List[str] = batch.get("site_id", [])
                record_ids: List[str] = batch.get("record_id", [])

                outputs = self._model.inference(
                    epoch_features=batch["epoch_features"],
                    demographics=batch["demographics"],
                    padding_mask=batch.get("padding_mask"),
                )

                all_probs.extend(outputs.ci_prob[:, 1].tolist())
                all_labels.extend(labels.tolist())
                all_site_ids.extend(site_ids)
                all_record_ids.extend(record_ids)

                pbar.update(batch["epoch_features"].shape[self.batch_dim])

        probs_arr = np.clip(np.nan_to_num(np.array(all_probs), nan=0.5), 0.0, 1.0)
        labels_arr = np.array(all_labels)

        auroc = float(roc_auc_score(labels_arr, probs_arr))
        auprc = float(average_precision_score(labels_arr, probs_arr))
        metrics: Dict[str, float] = {"auroc": auroc, "auprc": auprc}

        # Per-site AUROC — requires at least two classes present per site
        if all_site_ids:
            site_arr = np.array(all_site_ids)
            for site in sorted(np.unique(site_arr)):
                mask = site_arr == site
                if mask.sum() > 1 and len(np.unique(labels_arr[mask])) > 1:
                    metrics[f"auroc_{site}"] = float(roc_auc_score(labels_arr[mask], probs_arr[mask]))

        # Log a few sample predictions for a sanity check
        log_n = min(5, len(all_probs))
        for n in range(log_n):
            rid = all_record_ids[n] if all_record_ids else str(n)
            self.log_manager.log_message(f"  [{rid}]  CI prob: {all_probs[n]:.3f}  label: {bool(all_labels[n])}")

        self.model.train()
        return metrics

    # ── Boilerplate overrides ─────────────────────────────────────────────

    def _setup_criterion(self) -> None:
        # Criterion (BCEWithLogitsLoss) is embedded in the model; nothing to do.
        pass

    def _setup_augmenter_manager(self) -> None:
        # CAISR feature inputs require no signal augmentation.
        self.augmenter_manager = None

    def save_checkpoint(self, path: str) -> None:
        """Save model + optimizer state to *path*."""
        torch.save(
            {
                "model_state_dict": self._model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "model_config": self.model_config,
                "train_config": self.train_config,
                "epoch": self.epoch,
            },
            path,
        )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def get_args(**kwargs: Any) -> CFG:
    cfg = deepcopy(kwargs)
    parser = argparse.ArgumentParser(
        description="Train EpochTransformer on the CinC 2026 dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-d",
        "--db-dir",
        type=str,
        required=True,
        dest="db_dir",
        help="path to the CinC 2026 data root",
    )
    parser.add_argument("-b", "--batch-size", type=int, default=16, dest="batch_size")
    parser.add_argument("-e", "--epochs", type=int, default=50, dest="n_epochs")
    parser.add_argument("--lr", type=float, default=3e-4, dest="learning_rate")
    parser.add_argument(
        "--keep-checkpoint-max",
        type=int,
        default=5,
        dest="keep_checkpoint_max",
    )
    parser.add_argument("--debug", type=str2bool, default=False, dest="debug")
    args = vars(parser.parse_args())
    cfg.update(args)
    return CFG(cfg)


if __name__ == "__main__":
    train_config = get_args(**TrainCfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_config = deepcopy(ModelCfg.epoch_transformer)
    model = EpochTransformer(config=model_config)

    if torch.cuda.device_count() > 1:
        model = DP(model)
    model.to(device=device)

    trainer = CINC2026Trainer(
        model=model,
        model_config=model_config,
        train_config=train_config,
        device=device,
        lazy=True,
    )

    try:
        best_state_dict = trainer.train()
    except KeyboardInterrupt:
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
