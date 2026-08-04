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
- Primary evaluation metric: AUROC (``train_config.monitor = "auroc"``);
  official phase primary metric is the age-conditioned AUROC
  (``"auroc_age_cond"``), computed in :meth:`evaluate`.
- Per-site AUROC breakdown (S0001 / I0002 / I0006) is logged every epoch
  for domain-shift monitoring.
"""

import argparse
import logging
import os
import sys
import textwrap
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DataParallel as DP
from torch.utils.data import DataLoader, Dataset
from torch_ecg.cfg import CFG
from torch_ecg.components.trainer import BaseTrainer
from torch_ecg.utils.misc import get_date_str, str2bool
from tqdm.auto import tqdm

from cfg import ModelCfg, TrainCfg
from dataset import CINC2026Dataset, collate_fn
from models import EpochCRNN, EpochTransformer
from utils.scoring_metrics import compute_challenge_metrics

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
    __DEBUG__ = True

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
        # val_train_loader is enabled only in debug mode to monitor training-set
        # metrics.  Keeping it None in normal training avoids an overly optimistic
        # AUROC estimate from the (possibly imbalanced) training distribution.
        if self.train_config.get("debug", False):
            self.val_train_loader = DataLoader(
                dataset=train_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
                drop_last=False,
                collate_fn=collate_fn,
            )
        else:
            self.val_train_loader = None

    # ── Training loop ────────────────────────────────────────────────────

    def train(self) -> OrderedDict:
        """Train the model.

        Overrides :meth:`BaseTrainer.train` to guard the optional
        ``val_train_loader`` (active only when ``train_config.debug`` is
        ``True``) before calling :meth:`evaluate` — the base-class
        implementation does not perform this check.

        Returns
        -------
        OrderedDict
            State dict of the best model.
        """
        # lazy=True defers DataLoader construction; build them now
        if self.train_loader is None:
            self._setup_dataloaders()

        self._setup_optimizer()
        self._setup_scheduler()
        self._setup_criterion()

        # Fallback: if monitor is set but no validation loader exists at all,
        # use the training loader so model selection does not silently break.
        if self.train_config.monitor is not None:
            if self.val_loader is None and self.val_train_loader is None:
                self.val_train_loader = self.train_loader
                self.log_manager.log_message(
                    "No separate validation set provided while monitor is set; "
                    "falling back to training set for model selection.",
                    level=logging.WARNING,
                )

        self.log_manager.log_message(textwrap.dedent(f"""
                Starting training:
                ------------------
                Epochs:          {self.n_epochs}
                Batch size:      {self.batch_size}
                Learning rate:   {self.lr}
                Training size:   {self.n_train}
                Validation size: {self.n_val}
                Device:          {self.device.type}
                Optimizer:       {self.train_config.optimizer}
                Dataset classes: {self.train_config.classes}
                -----------------------------------------
            """))

        start_epoch = self.epoch
        for _ in range(start_epoch, self.n_epochs):
            self.model.train()
            self.epoch_loss = 0
            with tqdm(
                total=self.n_train,
                desc=f"Epoch {self.epoch}/{self.n_epochs}",
                unit="signals",
                dynamic_ncols=True,
                mininterval=1.0,
            ) as pbar:
                self.log_manager.epoch_start(self.epoch)
                self.train_one_epoch(pbar)

                # evaluate on train set only in debug mode
                if self.val_train_loader is not None:
                    eval_train_res = self.evaluate(self.val_train_loader)
                    self.log_manager.log_metrics(
                        metrics=eval_train_res,
                        step=self.global_step,
                        epoch=self.epoch,
                        part="train",
                    )
                else:
                    eval_train_res = {}

                # evaluate on validation set
                if self.val_loader is not None:
                    eval_res = self.evaluate(self.val_loader)
                    self.log_manager.log_metrics(
                        metrics=eval_res,
                        step=self.global_step,
                        epoch=self.epoch,
                        part="val",
                    )
                elif self.val_train_loader is not None:
                    eval_res = eval_train_res
                else:
                    eval_res = {}

                # model selection and early stopping
                if self.train_config.monitor is not None:
                    monitor_val = eval_res.get(self.train_config.monitor, -np.inf)
                    if monitor_val > self.best_metric:
                        self.best_metric = monitor_val
                        # Deep-copy: state_dict() returns *references* to the parameter
                        # tensors, so a shallow snapshot would silently track later
                        # training updates and the saved "best" weights would drift.
                        self.best_state_dict = deepcopy(self._model.state_dict())
                        self.best_eval_res = deepcopy(eval_res)
                        self.best_epoch = self.epoch
                        self.pseudo_best_epoch = self.epoch
                    elif self.train_config.early_stopping:
                        if monitor_val >= self.best_metric - self.train_config.early_stopping.min_delta:
                            self.pseudo_best_epoch = self.epoch
                        elif self.epoch - self.pseudo_best_epoch >= self.train_config.early_stopping.patience:
                            self.log_manager.log_message(f"early stopping triggered at epoch {self.epoch}")
                            break
                    self.log_manager.log_message(f"best metric = {self.best_metric:.4f},  obtained at epoch {self.best_epoch}")
                    save_suffix = f"epochloss_{self.epoch_loss:.5f}_metric_{monitor_val:.2f}"
                else:
                    save_suffix = f"epochloss_{self.epoch_loss:.5f}"

                save_filename = f"{self.save_prefix}_epoch{self.epoch}_{get_date_str()}_{save_suffix}.pth.tar"
                save_path = self.train_config.checkpoints / save_filename
                if self.train_config.keep_checkpoint_max != 0:
                    self.save_checkpoint(str(save_path))
                    self.saved_models.append(save_path)
                if len(self.saved_models) > self.train_config.keep_checkpoint_max > 0:
                    model_to_remove = self.saved_models.popleft()
                    try:
                        os.remove(model_to_remove)
                    except Exception:
                        self.log_manager.log_message(f"failed to remove {model_to_remove}")

                if self.train_config.lr_scheduler.lower() == "plateau":
                    self._update_lr(eval_res)

                self.log_manager.epoch_end(self.epoch)

            self.epoch += 1

        # save best model to model_dir
        if self.best_metric > -np.inf:
            if self.train_config.get("final_model_name"):
                save_filename = self.train_config.final_model_name
            else:
                monitor_val = self.best_eval_res.get(self.train_config.monitor, 0) if self.train_config.monitor else 0
                save_filename = (
                    f"BestModel_{self.save_prefix}{self.best_epoch}_{get_date_str()}_metric_{monitor_val:.2f}.pth.tar"
                )
            save_path = self.train_config.model_dir / save_filename
            self.save_checkpoint(str(save_path), state_dict=self.best_state_dict)
            self.log_manager.log_message(f"best model saved at {save_path}")
        elif self.train_config.monitor is None:
            self.log_manager.log_message("no monitor set; saving last model as best model")
            self.best_state_dict = self._model.state_dict()
            save_filename = f"BestModel_{self.save_prefix}{self.epoch}_{get_date_str()}.pth.tar"
            save_path = self.train_config.model_dir / save_filename
            self.save_checkpoint(str(save_path))
        else:
            raise ValueError("No best model found!")

        self.log_manager.close()

        if not self.best_state_dict:
            self.best_state_dict = self._model.state_dict()

        return self.best_state_dict

    # ── Training step ────────────────────────────────────────────────────

    def run_one_step(self, input_tensors: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Forward one batch through the model and return the output dict.

        The dataset emits ``"label"`` (singular) but the model expects
        ``"labels"`` (plural, float).  This method performs the rename,
        dtype conversion, optional label smoothing, and strips non-tensor
        string fields.

        Note: We use dict-based batches (not tuples), so torch_ecg's
        AugmenterManager (which expects positional tuple unpacking) is
        incompatible; label smoothing is applied here directly.
        """
        tensors: Dict[str, Any] = dict(input_tensors)
        if "label" in tensors and "labels" not in tensors:
            labels = tensors.pop("label").to(self.dtype)
            eps = float(self.train_config.get("label_smoothing", 0.0))
            if eps > 0.0 and self.model.training:
                labels = labels * (1.0 - eps) + eps * 0.5
            tensors["labels"] = labels
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
                    pbar.set_postfix(
                        **{
                            "loss (batch)": loss.item(),
                            "lr": self.scheduler.get_last_lr()[0],
                        }
                    )
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

        Primary metric is ``"auroc"`` (area under the ROC curve) with
        ``"auroc_age_cond"`` — the official age-conditioned AUROC over
        positive-negative pairs within ±2 years of age — computed alongside.
        Additional metrics include ``"auprc"`` and per-site AUROC values when
        site IDs are present in the batch.

        Parameters
        ----------
        data_loader : DataLoader
            Loader for the evaluation split.

        Returns
        -------
        dict
            ``{"auroc": float, "auprc": float, "auroc_age_cond": float, "auroc_<site>": float, ...}``
        """
        self.model.eval()

        all_probs: List[float] = []
        all_labels: List[int] = []
        all_ages: List[float] = []
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
                    night_features=batch.get("night_features"),
                )

                all_probs.extend(outputs.ci_prob[:, 1].tolist())
                all_labels.extend(labels.tolist())
                # demographics[:, 0] is Age normalised by /100 (FastDataReader)
                all_ages.extend((batch["demographics"][:, 0].cpu().numpy() * 100.0).tolist())
                all_site_ids.extend(site_ids)
                all_record_ids.extend(record_ids)

                pbar.update(batch["epoch_features"].shape[self.batch_dim])

        probs_arr = np.clip(np.nan_to_num(np.array(all_probs), nan=0.5), 0.0, 1.0)
        labels_arr = np.array(all_labels)
        ages_arr = np.array(all_ages)

        if len(np.unique(labels_arr)) < 2:
            self.log_manager.log_message(
                "Only one class present in evaluation split; AUROC/AUPRC set to chance level.",
                level=logging.WARNING,
            )
        metrics = compute_challenge_metrics(
            labels_arr,
            probs_arr,
            ages_arr,
            site_ids=np.array(all_site_ids) if all_site_ids else None,
            age_to_prevalence=None,
            threshold=getattr(self.model.config, "binary_threshold", 0.5),
        )

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

    def _setup_scheduler(self) -> None:
        """Override to pass ``pct_start`` to OneCycleLR.

        torch_ecg's base implementation builds OneCycleLR without ``pct_start``,
        defaulting to 0.3 (30 % warm-up).  For CRNN-style models that converge
        faster, a shorter warm-up (e.g. 0.1) helps.  We pass the value from
        ``train_config.pct_start`` if present, otherwise fall back to 0.3.
        """
        if self.train_config.get("lr_scheduler", "none").lower() not in (
            "one_cycle",
            "onecycle",
        ):
            super()._setup_scheduler()
            return

        pct_start = float(self.train_config.get("pct_start", 0.3))
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer=self.optimizer,
            max_lr=self.train_config.max_lr,
            epochs=self.n_epochs,
            steps_per_epoch=len(self.train_loader),
            pct_start=pct_start,
        )

    def _setup_augmenter_manager(self) -> None:
        # CAISR feature inputs require no signal augmentation.
        self.augmenter_manager = None

    def save_checkpoint(self, path: str, state_dict: Optional[Dict] = None) -> None:
        """Save model + optimizer state to *path*.

        If *state_dict* is ``None``, the current model state is used.
        """
        if state_dict is None:
            state_dict = self._model.state_dict()
        torch.save(
            {
                "model_state_dict": state_dict,
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
    parser.add_argument(
        "-m",
        "--model-dir",
        type=str,
        default=None,
        dest="model_dir",
        help="directory to save the final model (default: saved_models/)",
    )
    parser.add_argument(
        "-w",
        "--working-dir",
        type=str,
        default=None,
        dest="working_dir",
        help="working directory for logs and checkpoints (default: auto-generated under model_dir)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=TrainCfg.model_name,
        dest="model_name",
        help="model config name from ModelCfg, e.g. epoch_crnn_M, epoch_transformer_M",
    )
    # ── Age-adversarial branch (P0 mitigation) ────────────────────────────────
    parser.add_argument(
        "--age-adv",
        action="store_true",
        default=False,
        dest="age_adv_enable",
        help="enable the age-adversarial head (gradient reversal) to reduce age shortcut",
    )
    parser.add_argument(
        "--age-adv-alpha",
        type=float,
        default=0.5,
        dest="age_adv_alpha",
        help="GRL gradient-reversal coefficient for the age-adversarial head",
    )
    parser.add_argument(
        "--age-adv-lambda",
        type=float,
        default=1.0,
        dest="age_adv_lambda",
        help="weight of the age MSE in the total loss",
    )
    # ── Night-level aggregation features (P1, Phase 9) ─────────────────────────
    parser.add_argument(
        "--night-features",
        action="store_true",
        default=False,
        dest="night_features_enable",
        help="enable the night-level aggregation feature branch (Phase 9 / P1); "
        "EpochCRNN models only, ignored by EpochTransformer",
    )
    # ── O4: drop the age channel from FiLM demographics ───────────────────────
    parser.add_argument(
        "--no-age",
        action="store_true",
        default=False,
        dest="no_age",
        help="zero the age channel in the FiLM demographic conditioning "
        "(age-conditioned AUROC cannot be helped by age — constant within "
        "stratum); EpochCRNN models only, mutually exclusive with --age-adv",
    )
    args = vars(parser.parse_args())
    cfg.update(args)
    return CFG(cfg)


# ---------------------------------------------------------------------------
# Model lookup
# ---------------------------------------------------------------------------

_MODEL_CLASS_MAP = {
    "epoch_transformer": EpochTransformer,
    "epoch_crnn": EpochCRNN,
}

_MODEL_CONFIG_MAP = {}
for _name in dir(ModelCfg):
    if _name.startswith("epoch_transformer") or _name.startswith("epoch_crnn"):
        _MODEL_CONFIG_MAP[_name] = getattr(ModelCfg, _name)

# ---------------------------------------------------------------------------


if __name__ == "__main__":
    train_config = get_args(**TrainCfg)

    # Resolve model directory
    if train_config.get("model_dir", None) is not None:
        train_config.model_dir = Path(train_config.model_dir)
    else:
        train_config.model_dir = Path("saved_models")
    train_config.model_dir.mkdir(parents=True, exist_ok=True)

    # Resolve working directory
    if train_config.get("working_dir", None) is not None:
        train_config.working_dir = Path(train_config.working_dir)
    else:
        train_config.working_dir = train_config.model_dir / "working_dir"
    train_config.working_dir.mkdir(parents=True, exist_ok=True)

    train_config.checkpoints = train_config.working_dir / "checkpoints"
    train_config.log_dir = train_config.working_dir / "log"

    # Resolve model
    model_name = train_config.get("model_name", "epoch_crnn_M")
    # Determine model family: pick the first match in _MODEL_CLASS_MAP by prefix
    model_family = next(
        (prefix for prefix in _MODEL_CLASS_MAP if model_name.startswith(prefix)),
        "epoch_transformer",
    )
    model_cls = _MODEL_CLASS_MAP.get(model_family, EpochTransformer)
    model_config = deepcopy(_MODEL_CONFIG_MAP.get(model_name, ModelCfg.epoch_crnn))
    # Bridge TrainCfg.age_adv → model config (age-adversarial branch toggle).
    # CLI flags override the config block.
    age_adv_cfg = deepcopy(
        train_config.get(
            "age_adv",
            CFG(
                enable=False,
                alpha=0.5,
                lambda_=1.0,
                hidden_dim=32,
                position="before_film",
            ),
        )
    )
    if train_config.get("age_adv_enable", False):
        age_adv_cfg.enable = True
        age_adv_cfg.alpha = float(train_config.get("age_adv_alpha", age_adv_cfg.alpha))
        age_adv_cfg.lambda_ = float(train_config.get("age_adv_lambda", age_adv_cfg.lambda_))
    model_config.age_adv = age_adv_cfg
    # write back so the checkpoint's train_config mirrors what was actually trained
    train_config.age_adv = age_adv_cfg
    # Bridge TrainCfg.night_features → model config (night-level aggregation branch).
    night_cfg = deepcopy(
        train_config.get(
            "night_features",
            CFG(
                enable=False,
                dim=15,
                hidden_dim=[32, 16],
                activation="gelu",
                dropouts=0.1,
            ),
        )
    )
    if train_config.get("night_features_enable", False):
        night_cfg.enable = True
    model_config.night_features = night_cfg
    # write back so the checkpoint's train_config mirrors what was actually trained
    train_config.night_features = night_cfg
    # Bridge TrainCfg.no_age → model config (O4: drop the age channel from FiLM).
    no_age = bool(train_config.get("no_age", False))
    model_config.no_age = no_age
    # write back so the checkpoint's train_config mirrors what was actually trained
    train_config.no_age = no_age
    print(
        f"Model: {model_name} ({model_cls.__name__}), {sum(p.numel() for p in model_cls(config=model_config).parameters()):,} params"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_cls(config=model_config)

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
