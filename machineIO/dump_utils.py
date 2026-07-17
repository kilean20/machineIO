from __future__ import annotations

import datetime as _datetime
import keyword
import re
from pathlib import Path
from typing import Any, Dict

import numpy as np


def to_builtin(value: Any) -> Any:
    """Convert common scientific Python objects to plain Python/numeric data."""
    if _is_torch_tensor(value):
        return to_builtin(value.detach().cpu().numpy())

    if _is_pandas_dataframe(value):
        return {
            "__kind__": "pandas.DataFrame",
            "columns": [str(col) for col in value.columns.tolist()],
            "index": to_builtin(value.index.tolist()),
            "data": to_builtin(value.to_numpy().tolist()),
        }

    if _is_pandas_series(value):
        return {
            "__kind__": "pandas.Series",
            "name": None if value.name is None else str(value.name),
            "index": to_builtin(value.index.tolist()),
            "data": to_builtin(value.tolist()),
        }

    if isinstance(value, np.ndarray):
        return to_builtin(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, _datetime.datetime):
        return {"__kind__": "datetime.datetime", "value": value.isoformat()}
    if isinstance(value, _datetime.date):
        return {"__kind__": "datetime.date", "value": value.isoformat()}
    if isinstance(value, dict):
        return {str(key): to_builtin(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(val) for val in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def from_builtin(value: Any) -> Any:
    """Restore tagged values produced by to_builtin."""
    if isinstance(value, list):
        return [from_builtin(item) for item in value]

    if not isinstance(value, dict):
        return value

    kind = value.get("__kind__")
    if kind == "datetime.datetime":
        return _datetime.datetime.fromisoformat(value["value"])
    if kind == "datetime.date":
        return _datetime.date.fromisoformat(value["value"])
    if kind == "pandas.Series":
        import pandas as pd

        return pd.Series(
            from_builtin(value.get("data", [])),
            index=from_builtin(value.get("index", [])),
            name=value.get("name"),
        )
    if kind == "pandas.DataFrame":
        import pandas as pd

        return pd.DataFrame(
            from_builtin(value.get("data", [])),
            columns=from_builtin(value.get("columns", [])),
            index=from_builtin(value.get("index", [])),
        )

    return {key: from_builtin(val) for key, val in value.items()}


def write_hdf5_dump(path: str | Path, payload: Dict[str, Any]) -> Path:
    """Write a nested plain-data payload to HDF5 without pickle."""
    try:
        import tables
    except ImportError as exc:
        raise ImportError("HDF5 dump requires PyTables (`tables`).") from exc

    path = Path(path)
    if path.suffix.lower() not in {".h5", ".hdf5"}:
        path = path.with_suffix(".h5")
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = to_builtin(payload)

    with tables.open_file(path, mode="w") as h5:
        for attr_name in ("format", "format_version", "created_at"):
            if attr_name in payload:
                h5.root._v_attrs[attr_name] = payload[attr_name]

        for key, value in payload.items():
            _write_value(h5, h5.root, key, value)

    return path


def read_hdf5_dump(path: str | Path) -> Dict[str, Any]:
    """Read an HDF5 dump written by write_hdf5_dump into plain Python data."""
    try:
        import tables
    except ImportError as exc:
        raise ImportError("HDF5 dump reading requires PyTables (`tables`).") from exc

    path = Path(path)
    with tables.open_file(path, mode="r") as h5:
        payload: Dict[str, Any] = {}
        for attr_name in ("format", "format_version", "created_at"):
            value = _attr(h5.root, attr_name)
            if value is not None:
                payload[attr_name] = _decode(value)

        for child_name, child in h5.root._v_children.items():
            payload[child_name] = _read_value(child)

    return payload


def _is_torch_tensor(value: Any) -> bool:
    return (
        type(value).__module__.startswith("torch")
        and hasattr(value, "detach")
        and hasattr(value, "cpu")
        and hasattr(value, "numpy")
    )


def _is_pandas_dataframe(value: Any) -> bool:
    return type(value).__module__.startswith("pandas") and hasattr(value, "columns") and hasattr(value, "to_numpy")


def _is_pandas_series(value: Any) -> bool:
    return type(value).__module__.startswith("pandas") and hasattr(value, "index") and hasattr(value, "tolist")


def _safe_name(name: str, used: set[str]) -> str:
    safe = re.sub(r"\W+", "_", str(name)).strip("_")
    if not safe:
        safe = "item"
    if keyword.iskeyword(safe):
        safe = f"{safe}_"
    if safe[0].isdigit():
        safe = f"n_{safe}"
    base = safe
    idx = 1
    while safe in used:
        safe = f"{base}_{idx}"
        idx += 1
    used.add(safe)
    return safe


def _string_array(values):
    encoded = [str(value).encode("utf-8") for value in values]
    itemsize = max([len(value) for value in encoded] + [1])
    return np.asarray(encoded, dtype=f"S{itemsize}")


def _try_create_array(h5, parent, name: str, value):
    if isinstance(value, bool):
        return h5.create_array(parent, name, np.asarray(value, dtype=np.bool_))
    if isinstance(value, int) and not isinstance(value, bool):
        return h5.create_array(parent, name, np.asarray(value, dtype=np.int64))
    if isinstance(value, float):
        return h5.create_array(parent, name, np.asarray(value, dtype=np.float64))
    if isinstance(value, str):
        node = h5.create_array(parent, name, _string_array([value]))
        node._v_attrs["python_type"] = "str"
        node._v_attrs["encoding"] = "utf-8"
        return node
    if isinstance(value, list):
        if not value:
            return None
        if all(item is None or isinstance(item, (bool, int, float)) for item in value):
            has_none = any(item is None for item in value)
            arr = (
                np.asarray([np.nan if item is None else item for item in value], dtype=np.float64)
                if has_none
                else np.asarray(value)
            )
            node = h5.create_array(parent, name, arr)
            if has_none:
                node._v_attrs["none_as_nan"] = True
            return node
        if all(isinstance(item, str) for item in value):
            node = h5.create_array(parent, name, _string_array(value))
            node._v_attrs["python_type"] = "list[str]"
            node._v_attrs["encoding"] = "utf-8"
            return node
        try:
            arr = np.asarray(value)
        except (TypeError, ValueError):
            return None
        if arr.dtype != object and arr.dtype.kind in "biufc":
            return h5.create_array(parent, name, arr)
    return None


def _write_value(h5, parent, name: str, value):
    value = to_builtin(value)
    node = _try_create_array(h5, parent, name, value)
    if node is not None:
        return node

    group = h5.create_group(parent, name)
    if value is None:
        group._v_attrs["python_type"] = "None"
        return group

    if isinstance(value, dict):
        group._v_attrs["python_type"] = "dict"
        used: set[str] = set()
        for key, child_value in value.items():
            child_name = _safe_name(key, used)
            child = _write_value(h5, group, child_name, child_value)
            child._v_attrs["original_key"] = str(key)
        return group

    if isinstance(value, list):
        group._v_attrs["python_type"] = "list"
        group._v_attrs["length"] = len(value)
        used: set[str] = set()
        width = max(6, len(str(max(len(value) - 1, 0))))
        for idx, child_value in enumerate(value):
            child_name = _safe_name(f"item_{idx:0{width}d}", used)
            child = _write_value(h5, group, child_name, child_value)
            child._v_attrs["list_index"] = idx
        return group

    group._v_attrs["python_type"] = type(value).__name__
    group._v_attrs["repr"] = repr(value)
    return group


def _attr(node, name: str, default=None):
    try:
        return getattr(node._v_attrs, name)
    except AttributeError:
        return default


def _decode(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.bytes_):
        return bytes(value).decode("utf-8")
    return value


def _nan_to_none(value):
    if isinstance(value, float) and np.isnan(value):
        return None
    if isinstance(value, list):
        return [_nan_to_none(item) for item in value]
    return value


def _read_value(node):
    import tables

    if isinstance(node, tables.Group):
        python_type = _attr(node, "python_type")
        if python_type == "None":
            return None
        if python_type == "list":
            children = list(node._v_children.values())
            children.sort(key=lambda child: _attr(child, "list_index", 0))
            return [_read_value(child) for child in children]

        children = {}
        for child_name, child in node._v_children.items():
            key = _attr(child, "original_key", child_name)
            children[str(key)] = _read_value(child)
        if python_type in {None, "dict"}:
            return children
        if "repr" in node._v_attrs:
            return _attr(node, "repr")
        return children

    python_type = _attr(node, "python_type")
    data = node.read()

    if python_type == "str":
        return _decode(data.tolist()[0])
    if python_type == "list[str]":
        return [_decode(item) for item in data.tolist()]

    if isinstance(data, np.ndarray) and data.dtype.kind == "S":
        decoded = [_decode(item) for item in data.tolist()]
        return decoded[0] if data.shape == (1,) else decoded

    value = data.tolist() if hasattr(data, "tolist") else data
    if _attr(node, "none_as_nan", False):
        value = _nan_to_none(value)
    return value
