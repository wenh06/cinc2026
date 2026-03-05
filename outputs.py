"""
Output class for CinC2026.
"""

from dataclasses import dataclass, fields, is_dataclass
from typing import Optional, Sequence, Union, get_args, get_origin

import numpy as np
import pandas as pd
import torch

__all__ = [
    "CINC2026Outputs",
]


def allow_extra(cls):
    """Allow extra fields in dataclass."""

    def _from_dict_recursive(dc_cls, data):
        if not is_dataclass(dc_cls):
            return data

        valid_fields = {f.name: f for f in fields(dc_cls)}
        init_kwargs = {}
        for key, value in data.items():
            if key not in valid_fields:
                continue  # ignore extra fields
            field_info = valid_fields[key]
            field_type = field_info.type

            origin = get_origin(field_type)
            args = get_args(field_type)

            if is_dataclass(field_type):
                init_kwargs[key] = _from_dict_recursive(field_type, value)
            elif origin in (list, Sequence, tuple) and args and is_dataclass(args[0]):
                init_kwargs[key] = type(value)(_from_dict_recursive(args[0], v) for v in value)
            elif origin is dict and args and is_dataclass(args[1]):
                init_kwargs[key] = {k: _from_dict_recursive(args[1], v) for k, v in value.items()}
            else:
                init_kwargs[key] = value

        return dc_cls(**init_kwargs)

    def from_dict(data: dict):
        return _from_dict_recursive(cls, data)

    def _to_dict_recursive(obj):
        if is_dataclass(obj):
            return {f.name: _to_dict_recursive(getattr(obj, f.name)) for f in fields(obj)}
        elif isinstance(obj, (list, tuple)):
            return type(obj)(_to_dict_recursive(v) for v in obj)
        elif isinstance(obj, dict):
            return {k: _to_dict_recursive(v) for k, v in obj.items()}
        else:
            return obj

    def to_dict(self):
        return _to_dict_recursive(self)

    cls.from_dict = staticmethod(from_dict)
    cls.to_dict = to_dict
    return cls


@allow_extra
@dataclass
class CINC2026Outputs:
    """Output class for CinC2026.

    Attributes
    ----------
    cognitive_impairment : Sequence[bool] or Sequence[int]
        Predicted cognitive impairment diagnosis (CI).
    ci_logits : Sequence[Sequence[float]]
        Logits of the CI diagnosis.
    ci_prob : Sequence[Sequence[float]]
        Probabilities of the CI diagnosis.
    ci_loss : Sequence[float]
        Loss for the CI diagnosis.
    ci_threshold : float, default 0.5
        Threshold for the CI diagnosis.

    """

    cognitive_impairment: Optional[Sequence[Union[bool, int, float]]] = None
    ci_logits: Optional[Sequence[Sequence[float]]] = None
    ci_prob: Optional[Sequence[Sequence[float]]] = None
    ci_loss: Optional[Sequence[float]] = None
    ci_threshold: float = 0.5

    def __post_init__(self) -> None:
        assert any(
            [
                self.cognitive_impairment is not None,
                self.ci_logits is not None,
                self.ci_prob is not None,
            ]
        ), "at least one of CI-related prediction should be provided"

        # Processing CI predictions
        if self.ci_logits is not None:
            if isinstance(self.ci_logits, torch.Tensor):
                self.ci_logits = self.ci_logits.cpu().detach().numpy()
            else:
                self.ci_logits = np.asarray(self.ci_logits)

            if self.ci_prob is None:
                # assuming binary classification, use softmax
                self.ci_prob = torch.softmax(torch.tensor(self.ci_logits), dim=-1).cpu().detach().numpy()

        if self.ci_prob is not None:
            if isinstance(self.ci_prob, torch.Tensor):
                self.ci_prob = self.ci_prob.cpu().detach().numpy()
            else:
                self.ci_prob = np.asarray(self.ci_prob)

            if self.cognitive_impairment is None:
                # Use threshold on the positive class (index 1)
                self.cognitive_impairment = (self.ci_prob[:, 1] > self.ci_threshold).astype(bool).tolist()

        if self.cognitive_impairment is not None:
            if isinstance(self.cognitive_impairment, torch.Tensor):
                self.cognitive_impairment = self.cognitive_impairment.cpu().detach().numpy().astype(bool)
            else:
                self.cognitive_impairment = np.asarray(self.cognitive_impairment).astype(bool)
            self.cognitive_impairment = self.cognitive_impairment.tolist()

        if self.ci_loss is not None:
            if isinstance(self.ci_loss, torch.Tensor):
                self.ci_loss = self.ci_loss.cpu().detach().numpy()
            else:
                self.ci_loss = np.asarray(self.ci_loss)

    def append(self, values: Union["CINC2026Outputs", Sequence["CINC2026Outputs"]]) -> None:
        """Append other :class:`CINC2026Outputs` to `self`

        Parameters
        ----------
        values : CINC2026Outputs or Sequence[CINC2026Outputs]
            The values to be appended.

        Returns
        -------
        None

        """
        if not isinstance(values, Sequence):
            values = [values]
        for v in values:
            assert v.__class__ == self.__class__, "`values` must be of the same type as `self`"
            for k in fields(v):
                v_ = getattr(v, k.name)
                self_ = getattr(self, k.name)
                if v_ is None or self_ is None:
                    continue
                if k.name in ["ci_threshold"]:
                    assert v_ == self_, f"the field `{k.name}` must be identical"
                    continue
                if isinstance(v_, np.ndarray):
                    setattr(self, k.name, np.concatenate((self_, v_)))
                elif isinstance(v_, pd.DataFrame):
                    setattr(self, k.name, pd.concat([self_, v_], axis=0, ignore_index=True))
                elif isinstance(v_, Sequence):
                    setattr(self, k.name, list(self_) + list(v_))
                else:
                    try:
                        setattr(self, k.name, self_ + v_)
                    except Exception:
                        raise ValueError(f"field `{k.name}` of type `{type(v_)}` is not supported for appending")

    def drop(self, keys: Union[str, Sequence[str]]) -> None:
        """Drop the specified keys from the :class:`CINC2026Outputs`.

        Used to remove unnecessary fields for specific purposes
        (e.g. evaluation, visualization, etc.) to save memory.

        Parameters
        ----------
        keys : str or Sequence[str]
            The keys to be dropped.

        Returns
        -------
        None

        """
        if isinstance(keys, str):
            keys = [keys]
        output_fields = [item.name for item in fields(self)]
        for k in keys:
            if k in output_fields:
                setattr(self, k, None)
            else:
                raise KeyError(f"field `{k}` is not in the class")
