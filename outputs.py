""" """

from dataclasses import dataclass, fields, is_dataclass
from typing import Sequence, Union, get_args, get_origin

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

        return dc_cls(**init_kwargs)  # type: ignore

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
    to add

    """

    def __post_init__(self) -> None:
        raise NotImplementedError

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
        raise NotImplementedError

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
