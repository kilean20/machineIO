import os
import sys
import time
import datetime
import random
import warnings
import numpy as np
import pandas as pd
import concurrent
import concurrent.futures
import threading
from pathlib import Path
from typing import Any, Optional, List, Union, Dict, Callable, Tuple
from copy import deepcopy as copy
from abc import ABC, abstractmethod
from threading import Lock
import logging

logger = logging.getLogger(__name__)

# -------------------------
# Small utilities
# -------------------------
def unique_preserve_order(seq):
    """Order-preserving unique for hashable items (e.g. PV name strings)."""
    return list(dict.fromkeys(seq))


# -------------------------
# Optional GUI popup
# -------------------------
try:
    from .gui import popup_handler

    popup_ramping_not_OK = popup_handler(
        "Action required",
        "Ramping not OK. Manually adjust PV CSETs to jitter the power supply before continue.",
    )
except Exception:

    def popup_ramping_not_OK(message):
        """
        Fallback popup handler that prompts the user for input when ramping is not OK.
        Used in error handling scenarios to require manual intervention.
        """
        _ = input(message)


from .util import (
    display,
    cyclic_mean_var,
    suppress_outputs,
    sort_by_Dnum,
    validate_df_rows,
    df_mean,
    df_mean_var,
)
from .dump_utils import from_builtin, read_hdf5_dump, to_builtin, write_hdf5_dump
from .objFunc import SingleTaskObjectiveFunction

script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(script_dir, "models/BPMQ"))
from BPMQ_model import raw2Q_processor


# Default configuration values
DEFAULT_sample_interval = 0.2
DEFAULT_use_epics = False

# -------------------------
# EPICS import + "isOK" defaults
# -------------------------
try:
    from epics import caget as epics_caget
    from epics import caput as epics_caput
    from epics import caget_many as epics_caget_many
    from epics import caput_many as epics_caput_many

    epics_imported = True
    with suppress_outputs():
        # REA special-case: skip isOK check if REA probe exists
        if epics_caget("REA_EXP:ELMT") is not None:
            DEFAULT_isOK_PVs = None
            DEFAULT_isOK_vals = None
        else:
            if epics_caget("REA_EXP:ELMT") is not None:
                DEFAULT_isOK_PVs = None
                DEFAULT_isOK_vals = None
            else:
                DEFAULT_isOK_PVs = ["ACS_DIAG:CHP:STATE_RD"]  # is FRIB chopper on?
                DEFAULT_isOK_vals = [3]  # chopper on => 3
except ImportError:
    logger.warning("Failed to import 'epics'")
    epics_imported = False
    DEFAULT_isOK_PVs = None
    DEFAULT_isOK_vals = None


# -------------------------
# PHANTASY import
# -------------------------
try:
    from phantasy import fetch_data as phantasy_fetch_data_orig
    from phantasy import ensure_set as phantasy_ensure_set_orig

    phantasy_imported = True
except ImportError:
    logger.warning("Failed to import 'phantasy'")
    phantasy_imported = False
    if epics_imported:
        DEFAULT_use_epics = True
    else:
        raise ImportError("Neither 'epics' nor 'phantasy' could be imported.")
    
    DEFAULT_isOK_PVs = None
    DEFAULT_isOK_vals = None

    def phantasy_fetch_data_orig(*args, **kwargs):
        raise ImportError("phantasy is not available")

    def phantasy_ensure_set_orig(*args, **kwargs):
        raise ImportError("phantasy is not available")


# -------------------------
# PHANTASY wrappers
# -------------------------
if phantasy_imported:

    def _resample_df(df: pd.DataFrame, sample_interval: float = DEFAULT_sample_interval) -> pd.DataFrame:
        sample_interval = str(int(1000 * sample_interval)) + "ms"
        return df.bfill().ffill().resample(sample_interval).first().dropna()

    def phantasy_fetch_data(
        pvlist: List[str],
        time_span: float = 1.0,
        sample_interval: float = DEFAULT_sample_interval,
        **kws,
    ):
        pvlist = unique_preserve_order(pvlist)
        _, df = phantasy_fetch_data_orig(
            pvlist,
            time_span=time_span,
            with_data=True,
            data_opt={"with_timestamp": True, "fillna_method": "none"},
        )
        return _resample_df(df, sample_interval=sample_interval)

    def phantasy_ensure_set(
        setpoint_pv: List[str],
        readback_pv: List[str],
        goal: List[float],
        tol: List[float],
        timeout: float = 20.0,
        sample_interval: float = DEFAULT_sample_interval,
        extra_monitors: List[str] = None,
        **kws,
    ):
        setpoint_pv = list(setpoint_pv)
        readback_pv = list(readback_pv)
        extra_monitors = extra_monitors if extra_monitors is not None else []
        extra_monitors = unique_preserve_order(extra_monitors)

        ret, df = phantasy_ensure_set_orig(
            setpoint_pv,
            readback_pv,
            goal,
            tol=tol,
            timeout=timeout,
            extra_monitors=extra_monitors,
            keep_data=True,
            fillna_method="none",
        )
        return ret, _resample_df(df, sample_interval=sample_interval)


# -------------------------
# EPICS fetch_data / ensure_set
# -------------------------
if epics_imported:

    def epics_fetch_data(
        pvlist: List[str],
        time_span: float = 1.0,
        sample_interval: float = DEFAULT_sample_interval,
        **kws,
    ):
        pvlist = unique_preserve_order(pvlist)
        t0 = time.monotonic()
        index = [datetime.datetime.now()]

        first = epics_caget_many(pvlist)
        for pv, d in zip(pvlist, first):
            if d is None:
                raise ValueError(f"Failed to fetch data from {pv}")
        data = [first]

        while time.monotonic() - t0 < time_span:
            time.sleep(sample_interval)
            index.append(datetime.datetime.now())
            vals = epics_caget_many(pvlist)
            data.append(vals)

        df = pd.DataFrame(data, index=index, columns=pvlist).bfill().ffill()
        return df

    def epics_ensure_set(
        setpoint_pv: List[str],
        readback_pv: List[str],
        goal: List[float],
        tol: List[float],
        timeout: float = 30.0,
        sample_interval: float = DEFAULT_sample_interval,
        extra_monitors: List[str] = None,
        **kws,
    ):
        """
        EPICS ensure_set that:
        - caput_many(setpoint_pv, goal)
        - repeatedly reads readback_pv and checks |RB - goal| <= tol (aligned by order)
        - returns a dataframe with ordered columns setpoint_pv + readback_pv + extra_monitors
        """
        setpoint_pv = list(setpoint_pv)
        readback_pv = list(readback_pv)
        goal = np.asarray(goal, dtype=float).ravel()
        tol = np.asarray(tol, dtype=float).ravel()
        if len(goal) != len(setpoint_pv):
            raise ValueError("goal length must match setpoint_pv length")
        if len(readback_pv) != len(setpoint_pv):
            raise ValueError("readback_pv length must match setpoint_pv length")
        if len(tol) != len(setpoint_pv):
            raise ValueError("tol length must match setpoint_pv length")

        extra_monitors = extra_monitors or []
        extra_monitors = unique_preserve_order(extra_monitors)

        pvlist = unique_preserve_order(setpoint_pv + readback_pv + extra_monitors)

        t0 = time.monotonic()
        epics_caput_many(setpoint_pv, goal.tolist())

        index = []
        data = []

        def _read_all():
            vals = epics_caget_many(pvlist)
            for pv, v in zip(pvlist, vals):
                if v is None:
                    raise ValueError(f"Failed to caget {pv}")
            return vals

        def _read_rb():
            vals = epics_caget_many(readback_pv)
            for pv, v in zip(readback_pv, vals):
                if v is None:
                    raise ValueError(f"Failed to caget {pv}")
            return np.asarray(vals, dtype=float)

        rb = _read_rb()
        index.append(datetime.datetime.now())
        data.append(_read_all())

        while (time.monotonic() - t0) < timeout and np.any(np.abs(rb - goal) > tol):
            time.sleep(sample_interval)
            rb = _read_rb()
            index.append(datetime.datetime.now())
            data.append(_read_all())

        df = pd.DataFrame(data, index=index, columns=pvlist).bfill().ffill()
        ret = "PutFinish" if np.all(np.abs(rb - goal) <= tol) else "Timeout"
        return ret, df


# -------------------------
# isOK wrappers
# -------------------------
_ISOK_MISSING = object()


def _is_finite_number(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return False
    if not isinstance(value, (int, float, np.integer, np.floating)):
        return False
    return bool(np.isfinite(float(value)))


def _prepare_isOK_config(isOK_PVs, isOK_vals, test: bool = False):
    """Validate and normalize per-PV health-check specifications."""
    if test:
        return [], []
    if isOK_PVs is None and isOK_vals is None:
        return [], []
    if isOK_PVs is None or isOK_vals is None:
        raise ValueError("isOK_PVs and isOK_vals must either both be provided or both be None")

    pvs = list(isOK_PVs)
    specs = list(isOK_vals)
    if len(pvs) != len(specs):
        raise ValueError("isOK_PVs and isOK_vals must have the same length")
    if not all(isinstance(pv, str) and pv for pv in pvs):
        raise TypeError("isOK_PVs must contain non-empty PV-name strings")
    if len(set(pvs)) != len(pvs):
        raise ValueError("isOK_PVs must not contain duplicate PV names")

    for pv, spec in zip(pvs, specs):
        if isinstance(spec, str) or _is_finite_number(spec):
            continue
        if not isinstance(spec, dict):
            raise TypeError(
                f"isOK value for {pv!r} must be a string, finite number, "
                "or {'target': number, 'tolerance': non-negative number}"
            )
        if set(spec) != {"target", "tolerance"}:
            raise ValueError(
                f"range check for {pv!r} must contain exactly 'target' and 'tolerance'"
            )
        if not _is_finite_number(spec["target"]):
            raise TypeError(f"range target for {pv!r} must be a finite number")
        if not _is_finite_number(spec["tolerance"]) or float(spec["tolerance"]) < 0.0:
            raise ValueError(f"range tolerance for {pv!r} must be a non-negative finite number")

    return pvs, specs


def _as_finite_float(value: Any):
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return value if np.isfinite(value) else None


def _epics_isOK_value(pv: str, *, as_string: bool):
    """Read an enum's alternate representation when the fetch backend differs."""
    if not epics_imported:
        return _ISOK_MISSING
    try:
        value = epics_caget(pv, as_string=as_string)
    except Exception as exc:
        logger.debug("Could not read alternate isOK value for %s: %s", pv, exc)
        return _ISOK_MISSING
    return _ISOK_MISSING if value is None else value


def _isOK_value_matches(pv: str, actual: Any, spec: Any) -> bool:
    """Compare one fetched value against an exact or tolerance specification."""
    if isinstance(spec, str):
        if isinstance(actual, bytes):
            actual = actual.decode(errors="replace")
        if isinstance(actual, (str, np.str_)):
            return str(actual) == spec

        # PyEPICS may expose an enum as its integer index while PHANTASY exposes
        # the same value as its label. Resolve the label only when needed.
        label = _epics_isOK_value(pv, as_string=True)
        return label is not _ISOK_MISSING and str(label) == spec

    actual_number = _as_finite_float(actual)
    if actual_number is None:
        # Resolve a PHANTASY enum label to the raw numeric enum index.
        raw_value = _epics_isOK_value(pv, as_string=False)
        if raw_value is not _ISOK_MISSING:
            actual_number = _as_finite_float(raw_value)
    if actual_number is None:
        return False

    if isinstance(spec, dict):
        target = float(spec["target"])
        tolerance = float(spec["tolerance"])
        roundoff = 4.0 * np.finfo(float).eps * max(1.0, abs(actual_number), abs(target))
        return abs(actual_number - target) <= tolerance + roundoff
    return actual_number == float(spec)


def _check_isOK_values(df: pd.DataFrame, isOK_PVs, isOK_vals):
    """Return whether all checks pass plus details for failed checks."""
    if not isOK_PVs:
        return True, {}
    if df is None or df.empty:
        return False, {pv: {"actual": None, "expected": spec} for pv, spec in zip(isOK_PVs, isOK_vals)}

    missing = [pv for pv in isOK_PVs if pv not in df.columns]
    failures = {
        pv: {"actual": None, "expected": spec, "reason": "missing column"}
        for pv, spec in zip(isOK_PVs, isOK_vals)
        if pv in missing
    }
    for pv, spec in zip(isOK_PVs, isOK_vals):
        if pv in missing:
            continue
        actual = df[pv].iloc[-1]
        if not _isOK_value_matches(pv, actual, spec):
            failures[pv] = {"actual": actual, "expected": spec}
    return not failures, failures


class _fetch_data_wrapper:
    def __init__(
        self,
        isOK_PVs=DEFAULT_isOK_PVs,
        isOK_vals=DEFAULT_isOK_vals,
        fetch_data_base=epics_fetch_data if DEFAULT_use_epics and epics_imported else phantasy_fetch_data if phantasy_imported else None,
        test=False,
    ):
        if fetch_data_base is None and not test:
            raise AssertionError("epics or phantasy import failed")
        self.fetch_data_base = fetch_data_base
        self.isOK_PVs, self.isOK_vals = _prepare_isOK_config(isOK_PVs, isOK_vals, test=test)
        self.test = bool(test)

    def __call__(self, pvlist: List[str], time_span: float, sample_interval: float, **kws):
        pvlist = unique_preserve_order(pvlist)
        if self.test:
            return pd.DataFrame(
                [{pv: 0.0 for pv in pvlist}],
                index=[datetime.datetime.now()],
            )
        pvlist_expanded = unique_preserve_order(pvlist + [pv for pv in self.isOK_PVs if pv not in pvlist])
        df = self.fetch_data_base(pvlist_expanded, time_span, sample_interval=sample_interval)

        # Prefer the last sample for isOK PVs (more robust than a mean for state PVs).
        is_ok, failures = _check_isOK_values(df, self.isOK_PVs, self.isOK_vals)
        while not is_ok:
            logger.warning("notOK detected during fetch_data: %s. Re-try in 5 sec...", failures)
            time.sleep(5)
            df = self.fetch_data_base(pvlist_expanded, time_span, sample_interval=sample_interval)
            is_ok, failures = _check_isOK_values(df, self.isOK_PVs, self.isOK_vals)

        return df[pvlist]


class _ensure_set_wrapper:
    def __init__(
        self,
        isOK_PVs=DEFAULT_isOK_PVs,
        isOK_vals=DEFAULT_isOK_vals,
        ensure_set_base=epics_ensure_set if DEFAULT_use_epics and epics_imported else phantasy_ensure_set if phantasy_imported else None,
        test=False,
    ):
        if ensure_set_base is None and not test:
            raise AssertionError("epics or phantasy import failed")
        self.ensure_set_base = ensure_set_base
        self.isOK_PVs, self.isOK_vals = _prepare_isOK_config(isOK_PVs, isOK_vals, test=test)
        self.test = test

    def __call__(
        self,
        setpoint_pv: List[str],
        readback_pv: List[str],
        goal: List[float],
        tol: List[float],
        timeout: Optional[int],
        sample_interval: float,
        extra_monitors: Optional[List[str]] = None,
        **kws,
    ) -> Union[str, Union[None, pd.DataFrame]]:
        if self.test:
            return "PutFinish", None

        extra_monitors = extra_monitors or []
        extra_monitors_expanded = unique_preserve_order(list(extra_monitors) + self.isOK_PVs)

        ret, df = self.ensure_set_base(
            setpoint_pv,
            readback_pv,
            goal,
            tol,
            timeout=timeout,
            sample_interval=sample_interval,
            extra_monitors=extra_monitors_expanded,
            **kws,
        )

        if df is None:
            return ret, None

        is_ok, failures = _check_isOK_values(df, self.isOK_PVs, self.isOK_vals)
        if not is_ok:
            logger.warning("notOK detected during ensure_set: %s", failures)
            return ret, None

        cols = unique_preserve_order(list(setpoint_pv) + list(readback_pv) + list(extra_monitors))
        cols_in = [c for c in cols if c in df.columns]
        return ret, df[cols_in]


# -------------------------
# AbstractMachineIO + concrete constructor
# -------------------------
def _validate_manual_CSETs(manual_CSETs):
    if manual_CSETs is None:
        return []
    if not isinstance(manual_CSETs, (list, tuple, set)):
        raise TypeError("manual_CSETs must be a list/tuple/set of PV strings.")
    manual_CSETs = list(manual_CSETs)
    if not all(isinstance(pv, str) for pv in manual_CSETs):
        raise TypeError("manual_CSETs must contain only strings.")
    return unique_preserve_order(manual_CSETs)


class AbstractMachineIO(ABC):
    def __init__(
        self,
        ensure_set_timeout: int = 20,
        ensure_set_timewait_after_ramp: float = 0.2,
        fetch_data_time_span: float = 2.0,
        sample_interval: float = DEFAULT_sample_interval,
        verbose=False,
        manual_CSETs: Optional[List[str]] = None,
    ):
        self._ensure_set_timeout = ensure_set_timeout
        self._ensure_set_timewait_after_ramp = ensure_set_timewait_after_ramp
        self._fetch_data_time_span = fetch_data_time_span
        self._sample_interval = sample_interval
        self._verbose = verbose
        self._n_popup_ramping_issue = 0
        self._history_lock = Lock()
        self.history = []
        self.last_ensure_set_timing = {}
        self.last_fetch_data_timing = {}

        # NEW: configurable list of PVs that require manual control
        self.manual_CSETs = set(_validate_manual_CSETs(manual_CSETs))

    def _record_history(self, **kws):
        with self._history_lock:
            self.history.append({**kws, "time": datetime.datetime.now()})

    @abstractmethod
    def _caget(self, pvname: str):
        raise NotImplementedError

    def caget(self, pvname: str):
        value = self._caget(pvname)
        self._record_history(caller="caget", pvname=pvname, read=value)
        return value

    @abstractmethod
    def _caput(self, pvname: str, value: Union[float, int]):
        raise NotImplementedError

    def caput(self, pvname: str, value: Union[float, int]):
        self._caput(pvname, value)
        self._record_history(caller="caput", pvname=pvname, set=value)

    def _ensure_set(
        self,
        setpoint_pv: List[str],
        readback_pv: List[str],
        goal: List[float],
        tol: List[float],
        timeout: Optional[int],
        sample_interval: float,
        extra_monitors: Optional[List[str]] = None,
        **kws,
    ) -> Union[str, Union[None, pd.DataFrame]]:
        return "PutFinish", None

    def ensure_set(
        self,
        setpoint_pv: List[str],
        readback_pv: List[str],
        goal: List[float],
        tol: List[float],
        timeout: Optional[int] = None,
        sample_interval: Optional[float] = None,
        extra_monitors: Optional[List[str]] = None,
        verbose: Union[bool, None] = None,
        **kws,
    ):
        if self._verbose if verbose is None else verbose:
            print("Ramping in progress...")
            display(pd.DataFrame(np.asarray(goal).reshape(1, -1), columns=setpoint_pv))

        total_t0 = time.perf_counter()
        base_t0 = time.perf_counter()
        ret, data = self._ensure_set(
            setpoint_pv,
            readback_pv,
            goal,
            tol,
            timeout=timeout or self._ensure_set_timeout,
            sample_interval=sample_interval or self._sample_interval,
            extra_monitors=extra_monitors,
            **kws,
        )
        base_dt = time.perf_counter() - base_t0

        if ret == "Timeout":
            if self._n_popup_ramping_issue < 2:
                self._n_popup_ramping_issue += 1
                # popup_ramping_not_OK("Ramping timeout. Please check PVs and press Enter.")
            else:
                logger.warning("'ramping_not_OK' issued 2 times already. Ignoring from now on...")

        wait_t0 = time.perf_counter()
        time.sleep(self._ensure_set_timewait_after_ramp)
        wait_dt = time.perf_counter() - wait_t0
        self._record_history(
            caller="ensure_set",
            setpoint_pv=setpoint_pv,
            readback_pv=readback_pv,
            goal=goal,
            tol=tol,
            ret=ret,
            data=data,
        )
        self.last_ensure_set_timing = {
            "machineio_ensure_set_base": base_dt,
            "machineio_ensure_set_wait_after_ramp": wait_dt,
            "machineio_ensure_set_total": time.perf_counter() - total_t0,
        }
        return ret, data

    def _fetch_data(self, pvlist: List[str], time_span: float, sample_interval: float, **kws):
        raise NotImplementedError

    def fetch_data(
        self,
        pvlist: List[str],
        time_span: float = None,
        sample_interval: float = None,
        verbose: Optional[bool] = None,
        **kws,
    ):
        pvlist = unique_preserve_order(pvlist)
        total_t0 = time.perf_counter()
        fetch_t0 = time.perf_counter()
        data = self._fetch_data(
            pvlist,
            time_span=time_span or self._fetch_data_time_span,
            sample_interval=sample_interval or self._sample_interval,
            **kws,
        )
        fetch_dt = time.perf_counter() - fetch_t0
        if self._verbose if verbose is None else verbose:
            display(data)
        self.last_fetch_data_timing = {
            "machineio_fetch_data": fetch_dt,
            "machineio_fetch_data_total": time.perf_counter() - total_t0,
        }
        return data


class construct_machineIO(AbstractMachineIO):
    def __init__(
        self,
        ensure_set_timeout: int = 20,
        ensure_set_timewait_after_ramp: float = 0.3,
        fetch_data_time_span: float = 2.0,
        sample_interval: float = DEFAULT_sample_interval,
        verbose=False,
        use_epics: bool = DEFAULT_use_epics,
        isOK_PVs: Optional[List[str]] = DEFAULT_isOK_PVs,
        isOK_vals: Optional[List[Union[str, int, float, Dict[str, float]]]] = DEFAULT_isOK_vals,
        manual_CSETs: Optional[List[str]] = None,  # NEW
        test: bool = False,
    ):
        """Construct machine I/O with optional per-PV health checks.

        Each ``isOK_vals`` entry is either an exact string, an exact number, or
        ``{"target": number, "tolerance": non_negative_number}`` for an inclusive
        numeric range. Entries correspond positionally to ``isOK_PVs``.
        """
        super().__init__(
            ensure_set_timeout=ensure_set_timeout,
            ensure_set_timewait_after_ramp=ensure_set_timewait_after_ramp,
            fetch_data_time_span=fetch_data_time_span,
            sample_interval=sample_interval,
            verbose=verbose,
            manual_CSETs=manual_CSETs,
        )
        self.test = test
        self.use_epics = bool(use_epics)
        self.isOK_PVs = isOK_PVs
        self.isOK_vals = isOK_vals

        self._ensure_set = _ensure_set_wrapper(
            isOK_PVs=self.isOK_PVs,
            isOK_vals=self.isOK_vals,
            ensure_set_base=epics_ensure_set if use_epics and epics_imported else phantasy_ensure_set if phantasy_imported else None,
            test=self.test,
        )
        self._fetch_data = _fetch_data_wrapper(
            isOK_PVs=self.isOK_PVs,
            isOK_vals=self.isOK_vals,
            fetch_data_base=epics_fetch_data if use_epics and epics_imported else phantasy_fetch_data if phantasy_imported else None,
            test=self.test,
        )

    def _caget(self, pvname):
        if epics_imported:
            return epics_caget(pvname)
        if self.test:
            warnings.warn("EPICS is not imported. caget will return fake zero")
            return 0
        raise ValueError("EPICS is not imported. Cannot caget.")

    def _caput(self, pvname: str, value: Union[float, int]):
        if self.test:
            return
        if epics_imported:
            epics_caput(pvname, value)
            return
        raise ValueError("EPICS is not imported. Cannot caput.")

    def to_dump_dict(self, *, include_history: bool = True) -> Dict[str, Any]:
        payload = {
            "class": type(self).__name__,
            "module": type(self).__module__,
            "format_version": 1,
            "config": {
                "ensure_set_timeout": self._ensure_set_timeout,
                "ensure_set_timewait_after_ramp": self._ensure_set_timewait_after_ramp,
                "fetch_data_time_span": self._fetch_data_time_span,
                "sample_interval": self._sample_interval,
                "verbose": self._verbose,
                "use_epics": self.use_epics,
                "isOK_PVs": None if self.isOK_PVs is None else list(self.isOK_PVs),
                "isOK_vals": None if self.isOK_vals is None else to_builtin(self.isOK_vals),
                "manual_CSETs": sorted(self.manual_CSETs),
                "test": self.test,
            },
            "state": {
                "last_ensure_set_timing": self.last_ensure_set_timing,
                "last_fetch_data_timing": self.last_fetch_data_timing,
            },
        }
        if include_history:
            payload["state"]["history"] = self.history
        return to_builtin(payload)

    @classmethod
    def from_dump_dict(
        cls,
        data: Dict[str, Any],
        *,
        restore_history: bool = True,
        use_epics: Optional[bool] = None,
        test: Optional[bool] = None,
    ) -> "construct_machineIO":
        config = dict(data.get("config", data))
        if use_epics is not None:
            config["use_epics"] = bool(use_epics)
        if test is not None:
            config["test"] = bool(test)

        machine = cls(
            ensure_set_timeout=int(config.get("ensure_set_timeout", 20)),
            ensure_set_timewait_after_ramp=float(config.get("ensure_set_timewait_after_ramp", 0.3)),
            fetch_data_time_span=float(config.get("fetch_data_time_span", 2.0)),
            sample_interval=float(config.get("sample_interval", DEFAULT_sample_interval)),
            verbose=bool(config.get("verbose", False)),
            use_epics=bool(config.get("use_epics", DEFAULT_use_epics)),
            isOK_PVs=config.get("isOK_PVs", DEFAULT_isOK_PVs),
            isOK_vals=config.get("isOK_vals", DEFAULT_isOK_vals),
            manual_CSETs=config.get("manual_CSETs", None),
            test=bool(config.get("test", False)),
        )

        state = from_builtin(data.get("state", {}) or {})
        machine.last_ensure_set_timing = dict(state.get("last_ensure_set_timing", {}) or {})
        machine.last_fetch_data_timing = dict(state.get("last_fetch_data_timing", {}) or {})
        if restore_history:
            machine.history = list(state.get("history", []) or [])
        return machine

    def dump(self, path: str | Path, *, include_history: bool = True) -> Path:
        payload = {
            "format": "machineIO.construct_machineIO.dump",
            "format_version": 1,
            "object": self.to_dump_dict(include_history=include_history),
        }
        return write_hdf5_dump(path, payload)

    @staticmethod
    def read_dump(path: str | Path) -> Dict[str, Any]:
        return read_hdf5_dump(path)

    @classmethod
    def from_dump(
        cls,
        path: str | Path,
        *,
        restore_history: bool = True,
        use_epics: Optional[bool] = None,
        test: Optional[bool] = None,
    ) -> "construct_machineIO":
        payload = cls.read_dump(path)
        return cls.from_dump_dict(
            payload.get("object", payload),
            restore_history=restore_history,
            use_epics=use_epics,
            test=test,
        )


# -------------------------
# Validators (Evaluator inputs)
# -------------------------
def _validate_machineIO(machineIO):
    if not hasattr(machineIO, "ensure_set") or not callable(getattr(machineIO, "ensure_set", None)):
        raise TypeError("machineIO must have a callable `ensure_set` method.")
    if not hasattr(machineIO, "fetch_data") or not callable(getattr(machineIO, "fetch_data", None)):
        raise TypeError("machineIO must have a callable `fetch_data` method.")


def _validate_control_CSETs(control_CSETs):
    if not isinstance(control_CSETs, list) or not all(isinstance(cset, str) for cset in control_CSETs):
        raise TypeError("control_CSETs must be a list of strings.")
    if len(control_CSETs) != len(set(control_CSETs)):
        raise ValueError("control_CSETs contains duplicate entries.")


def _validate_control_RDs(control_RDs, control_CSETs):
    if not isinstance(control_RDs, list) or not all(isinstance(rd, str) for rd in control_RDs):
        raise TypeError("control_RDs must be a list of strings.")
    if len(control_RDs) != len(set(control_RDs)):
        raise ValueError("control_RDs contains duplicate entries.")
    if len(control_RDs) != len(control_CSETs):
        raise ValueError("The length of control_RDs must match the length of control_CSETs.")


def _validate_control_tols(control_tols, control_CSETs):
    if not isinstance(control_tols, (list, np.ndarray)) or not all(isinstance(tol, (int, float)) for tol in control_tols):
        raise TypeError("control_tols must be a list or numpy array of numbers.")
    if len(control_tols) != len(control_CSETs):
        raise ValueError("Length of control_tols must match length of control_CSETs.")


def _validate_monitor_PVs(monitor_PVs, control_CSETs, control_RDs):
    if monitor_PVs is None:
        return
    if not isinstance(monitor_PVs, list) or not all(isinstance(pv, str) for pv in monitor_PVs):
        raise TypeError("monitor_PVs must be a list of strings.")
    if len(monitor_PVs) != len(set(monitor_PVs)):
        raise ValueError("monitor_PVs contains duplicate entries.")
    if set(monitor_PVs).intersection(set(control_CSETs)):
        raise ValueError("monitor_PVs must be disjoint from control_CSETs.")
    if set(monitor_PVs).intersection(set(control_RDs)):
        raise ValueError("monitor_PVs must be disjoint from control_RDs.")


# -------------------------
# Evaluators
# -------------------------
TISRAW_VECTOR_LENGTH = 68


class EvaluatorBase:
    def __init__(
        self,
        machineIO,
        control_CSETs: List[str],
        control_RDs: List[str],
        control_tols: Union[List[float], np.ndarray],
        monitor_PVs: Optional[List[str]] = None,
        ensure_set_kwargs: Optional[Dict] = None,
        fetch_data_kwargs: Optional[Dict] = None,
        set_manually: Optional[bool] = False,
        df_manipulators: Optional[List[Callable]] = None,
    ):
        _validate_machineIO(machineIO)
        _validate_control_CSETs(control_CSETs)
        _validate_control_RDs(control_RDs, control_CSETs)
        _validate_control_tols(control_tols, control_CSETs)
        _validate_monitor_PVs(monitor_PVs, control_CSETs, control_RDs)

        self.machineIO = machineIO
        self.ensure_set_kwargs = ensure_set_kwargs or {}
        self.fetch_data_kwargs = fetch_data_kwargs or {}

        if monitor_PVs is None:
            monitor_PVs = []

        self.control_CSETs = list(control_CSETs)
        self.control_RDs = list(control_RDs)
        self.control_tols = np.asarray(control_tols, dtype=float)
        self.monitor_PVs = list(monitor_PVs)
        self.set_manually = bool(set_manually)
        self.df_manipulators = df_manipulators

        # Preserve order: controls first, then monitors
        self.fetch_data_monitors = unique_preserve_order(self.control_CSETs + self.control_RDs + self.monitor_PVs)
        self.ensure_set_monitors = [m for m in self.fetch_data_monitors if m not in self.control_RDs and m not in self.control_CSETs]

        self.TISRAW_PVs = [pv for pv in self.fetch_data_monitors if ":TISRAW" in pv]
        if self.TISRAW_PVs:
            self.vector_PVs = [self.TISRAW_PVs]
            self.vector_len = [TISRAW_VECTOR_LENGTH]
        else:
            self.vector_PVs = []
            self.vector_len = []
        self.scalar_PVs = [pv for pv in self.fetch_data_monitors if pv not in set(self.TISRAW_PVs)]

        self._history_lock = Lock()
        self.history = {"mean": [], "var": [], "ramping_mean": [], "ramping_var": []}
        self.last_read_timing = {}
        self.last_timing = {}

        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    def read(self, fetch_data_kwargs: Optional[Dict] = None):
        fetch_data_kwargs = fetch_data_kwargs or self.fetch_data_kwargs
        timing = {}
        fetch_t0 = time.perf_counter()
        df = self.machineIO.fetch_data(self.fetch_data_monitors, **fetch_data_kwargs)
        timing["oracle_fetch_data"] = time.perf_counter() - fetch_t0
        timing.update(getattr(self.machineIO, "last_fetch_data_timing", {}) or {})

        validate_t0 = time.perf_counter()
        df = validate_df_rows(df, self.vector_PVs, self.vector_len)
        timing["oracle_validate_data"] = time.perf_counter() - validate_t0

        manip_t0 = time.perf_counter()
        if self.df_manipulators is not None:
            for f in self.df_manipulators:
                if callable(f):
                    df = f(df)
                else:
                    raise ValueError(f"df_manipulators should be callable, got {type(f).__name__}")
        timing["oracle_df_manipulators"] = time.perf_counter() - manip_t0
        timing["oracle_read_total"] = sum(
            timing.get(k, 0.0)
            for k in ("oracle_fetch_data", "oracle_validate_data", "oracle_df_manipulators")
        )
        self.last_read_timing = timing
        return df

    # ---------- manual-controls support ----------
    def _split_manual_controls(self, x_arr: np.ndarray):
        """
        Split controls into (manual) and (auto) subsets based on machineIO.manual_CSETs.
        Returns: (manual_dict_or_None, auto_dict)
        """
        manual_set = getattr(self.machineIO, "manual_CSETs", set()) or set()
        manual_idxs = [i for i, pv in enumerate(self.control_CSETs) if pv in manual_set]

        if not manual_idxs:
            auto = {"csets": self.control_CSETs, "rds": self.control_RDs, "goals": x_arr, "tols": self.control_tols}
            return None, auto

        manual = {
            "indices": manual_idxs,
            "csets": [self.control_CSETs[i] for i in manual_idxs],
            "rds": [self.control_RDs[i] for i in manual_idxs],
            "goals": np.asarray([x_arr[i] for i in manual_idxs], dtype=float),
            "tols": np.asarray([self.control_tols[i] for i in manual_idxs], dtype=float),
        }

        keep = [i for i in range(len(self.control_CSETs)) if i not in manual_idxs]
        auto = {
            "csets": [self.control_CSETs[i] for i in keep],
            "rds": [self.control_RDs[i] for i in keep],
            "goals": np.asarray([x_arr[i] for i in keep], dtype=float),
            "tols": np.asarray([self.control_tols[i] for i in keep], dtype=float),
        }
        return manual, auto

    def _prompt_manual_set_subset(self, csets, rds, goals, tols, timeout=None, sample_interval=None):
        """
        Prompt user to manually set PVs. MUST be called from the main thread.
        Optionally verifies readbacks within tolerance.
        """
        df = pd.DataFrame({"CSET": csets, "RD": rds, "goal": goals, "tol": tols})
        print("\nManual set required for these PVs:\n")
        try:
            display(df)
        except Exception:
            print(df.to_string(index=False))

        # Copy-pastable helper
        try:
            goals_list = goals.tolist() if isinstance(goals, np.ndarray) else list(goals)
            tols_list = tols.tolist() if isinstance(tols, np.ndarray) else list(tols)
        except Exception:
            goals_list = list(goals)
            tols_list = list(tols)

        print(
            f"\nensure_set({csets}, {rds}, {goals_list}, tol={tols_list}, timeout={getattr(self.machineIO, '_ensure_set_timeout', 20)})"
        )
        input("\nSet the above PVs manually, then press Enter to continue...")

        # Verification loop (recommended)
        timeout = getattr(self.machineIO, "_ensure_set_timeout", 20) if timeout is None else timeout
        sample_interval = getattr(self.machineIO, "_sample_interval", 0.2) if sample_interval is None else sample_interval

        t0 = time.monotonic()
        while True:
            rb_vals = []
            for pv in rds:
                v = self.machineIO.caget(pv)
                rb_vals.append(float(v))
            rb_vals = np.asarray(rb_vals, dtype=float)

            if np.all(np.abs(rb_vals - goals) <= tols):
                break

            if time.monotonic() - t0 > timeout:
                raise TimeoutError(
                    f"Manual PVs not within tolerance after {timeout} s.\n"
                    f"readbacks={rb_vals}, goals={goals}, tols={tols}"
                )
            time.sleep(sample_interval)

    def _set_and_read(self, x, ensure_set_kwargs: Optional[Dict] = None, fetch_data_kwargs: Optional[Dict] = None):
        """
        Internal method to set the values and read the data.

        Fixes:
        - No hard-coded manual PVs
        - Uses machineIO.manual_CSETs for manual-control subset
        - set_manually now behaves correctly for direct calls (main thread)
        - avoids input() inside worker threads unless caller already prompted in main thread
        """
        ensure_set_kwargs = ensure_set_kwargs or self.ensure_set_kwargs
        fetch_data_kwargs = fetch_data_kwargs or self.fetch_data_kwargs

        x_arr = np.asarray(x, dtype=float).ravel()
        if len(x_arr) != len(self.control_CSETs):
            raise ValueError(f"x length ({len(x_arr)}) must match control_CSETs length ({len(self.control_CSETs)})")

        timing = {}
        total_t0 = time.perf_counter()
        manual, auto = self._split_manual_controls(x_arr)

        # FULL manual mode: prompt only if in main thread (direct calls)
        if self.set_manually:
            manual_t0 = time.perf_counter()
            if threading.current_thread() is threading.main_thread():
                df = pd.DataFrame([x_arr], columns=self.control_CSETs)
                try:
                    display(df)
                except Exception:
                    print(df.to_string(index=False))
                input("Set the above PVs manually, then press Enter to continue...")
            ret, ramping_data = "PutFinish", None
            timing["oracle_manual_prompt"] = time.perf_counter() - manual_t0
        else:
            # If there is a manual subset, prompt only in main thread (avoid worker-thread hangs)
            if manual is not None and threading.current_thread() is threading.main_thread():
                manual_t0 = time.perf_counter()
                self._prompt_manual_set_subset(
                    manual["csets"],
                    manual["rds"],
                    manual["goals"],
                    manual["tols"],
                    timeout=ensure_set_kwargs.get("timeout", None),
                    sample_interval=ensure_set_kwargs.get("sample_interval", None),
                )
                timing["oracle_manual_prompt"] = time.perf_counter() - manual_t0

            extra_monitors = list(self.ensure_set_monitors)
            if manual is not None:
                extra_monitors = unique_preserve_order(extra_monitors + manual["csets"] + manual["rds"])

            ensure_t0 = time.perf_counter()
            ret, ramping_data = self.machineIO.ensure_set(
                auto["csets"],
                auto["rds"],
                auto["goals"],
                auto["tols"],
                extra_monitors=extra_monitors,
                **ensure_set_kwargs,
            )
            timing["oracle_ensure_set"] = time.perf_counter() - ensure_t0
            timing.update(getattr(self.machineIO, "last_ensure_set_timing", {}) or {})

            if ramping_data is not None:
                ramp_validate_t0 = time.perf_counter()
                ramping_data = validate_df_rows(ramping_data, self.vector_PVs, self.vector_len)
                timing["oracle_ramping_validate"] = time.perf_counter() - ramp_validate_t0

                ramp_manip_t0 = time.perf_counter()
                if self.df_manipulators is not None:
                    for f in self.df_manipulators:
                        if callable(f):
                            ramping_data = f(ramping_data)
                        else:
                            raise ValueError(f"df_manipulators should be callable, got {type(f).__name__}")
                timing["oracle_ramping_df_manipulators"] = time.perf_counter() - ramp_manip_t0

                ramp_stats_t0 = time.perf_counter()
                ramping_mean, ramping_var = df_mean_var(ramping_data)
                with self._history_lock:
                    self.history["ramping_mean"].append(ramping_mean)
                    self.history["ramping_var"].append(ramping_var)
                timing["oracle_ramping_stats"] = time.perf_counter() - ramp_stats_t0

        read_t0 = time.perf_counter()
        data = self.read(fetch_data_kwargs=fetch_data_kwargs)
        timing["oracle_read"] = time.perf_counter() - read_t0
        timing.update(self.last_read_timing)

        stats_t0 = time.perf_counter()
        mean, var = df_mean_var(data)
        with self._history_lock:
            self.history["mean"].append(mean)
            self.history["var"].append(var)
        timing["oracle_stats"] = time.perf_counter() - stats_t0
        timing["oracle_set_and_read_total"] = time.perf_counter() - total_t0
        self.last_timing = timing

        return data, ramping_data

    def submit(self, x, ensure_set_kwargs=None, fetch_data_kwargs=None):
        """
        Submit a task to set and read data asynchronously.

        Behavior:
        - If set_manually=True: prompts here (main thread), then worker thread runs _set_and_read without prompts.
        - Else if machineIO.manual_CSETs overlaps with controls: prompts subset here (main thread) to avoid worker-thread input().
        """
        x_arr = np.asarray(x, dtype=float).ravel()

        # If full manual mode: prompt in main thread
        if self.set_manually:
            df = pd.DataFrame([x_arr], columns=self.control_CSETs)
            try:
                display(df)
            except Exception:
                print(df.to_string(index=False))
            tol_list = self.control_tols.tolist()
            print(
                f"ensure_set({self.control_CSETs},{self.control_RDs},{x_arr.tolist()},tol={tol_list},timeout={getattr(self.machineIO,'_ensure_set_timeout',20)})"
            )
            input("Set the above PVs and press Enter to continue...")

        else:
            # If partial manual subset exists: prompt subset here (main thread)
            manual, _ = self._split_manual_controls(x_arr)
            if manual is not None:
                self._prompt_manual_set_subset(
                    manual["csets"], manual["rds"], manual["goals"], manual["tols"]
                )

        future = self.executor.submit(
            self._set_and_read,
            x,
            ensure_set_kwargs=ensure_set_kwargs,
            fetch_data_kwargs=fetch_data_kwargs,
        )
        return future

    def is_job_done(self, future: concurrent.futures.Future) -> bool:
        return future.done()

    def get_result(self, future: concurrent.futures.Future) -> Tuple[Union[np.ndarray, pd.DataFrame], Union[np.ndarray, pd.DataFrame]]:
        data, ramping_data = future.result()
        self._data = data
        self._ramping_data = ramping_data
        return data, ramping_data

    def get_history(self, ignore_index: bool = False, columns: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
        history = {}
        for k, v in self.history.items():
            if not v:
                history[k] = pd.DataFrame(columns=columns) if columns else pd.DataFrame()
            else:
                df = pd.concat([s.to_frame().T for s in v], ignore_index=ignore_index)
                if columns:
                    df = df.reindex(columns=columns, fill_value=np.nan)
                history[k] = df
        return history

    def clear_history(self):
        with self._history_lock:
            self.history = {"mean": [], "var": [], "ramping_mean": [], "ramping_var": []}

    def dump_history(self, filename: str):
        """Dump the history dict to an HDF5 file without pickle."""
        path = Path(filename)
        if path.suffix.lower() not in {".h5", ".hdf5"}:
            path = path.with_suffix(".h5")
        payload = {
            "format": "machineIO.EvaluatorBase.history.dump",
            "format_version": 1,
            "history": self.get_history(ignore_index=True),
        }
        return write_hdf5_dump(path, payload)


# -------------------------
# Control couplings
# -------------------------
def _validate_control_couplings(control_couplings, control_CSETs):
    if control_couplings is None:
        return
    if not isinstance(control_couplings, dict):
        raise TypeError("control_couplings must be a dictionary.")
    for key, value in control_couplings.items():
        if not isinstance(key, str) or key not in control_CSETs:
            raise ValueError(f"control_couplings: Key '{key}' must be a string and in control_CSETs.")
        if not isinstance(value, dict):
            raise TypeError(f"Value for '{key}' must be a dictionary.")

        required_keys = {"CSETs", "RDs", "coeffs", "tols"}
        if not required_keys.issubset(value.keys()):
            missing = required_keys - set(value.keys())
            raise ValueError(f"control_couplings: Value for '{key}' missing keys: {missing}")

        csets = value["CSETs"]
        rds = value["RDs"]
        coeff = value["coeffs"]
        tol = value["tols"]
        if not isinstance(csets, list) or not all(isinstance(c, str) for c in csets):
            raise TypeError(f"control_couplings: 'CSETs' for '{key}' must be a list of strings.")
        if not isinstance(rds, list) or not all(isinstance(r, str) for r in rds):
            raise TypeError(f"control_couplings: 'RDs' for '{key}' must be a list of strings.")
        if not isinstance(coeff, (list, np.ndarray)) or not all(isinstance(c, (int, float)) for c in coeff):
            raise TypeError(f"control_couplings: 'coeffs' for '{key}' must be a list/array of numbers.")
        if not isinstance(tol, (list, np.ndarray)) or not all(isinstance(t, (int, float)) for t in tol):
            raise TypeError(f"control_couplings: 'tols' for '{key}' must be a list/array of numbers.")

        if len(coeff) != len(csets) or len(tol) != len(csets) or len(rds) != len(csets):
            raise ValueError(f"control_couplings: lengths of coeffs/tols/RDs must match CSETs for '{key}'.")


def _precompute_control_couplings_and_indices(control_couplings, control_CSETs, control_RDs, control_tols):
    expanded_control_CSETs = list(control_CSETs)
    expanded_control_RDs = list(control_RDs)
    expanded_control_tols = list(control_tols)
    coupling_indices = {}

    if control_couplings:
        for pv, value in control_couplings.items():
            expanded_control_CSETs.extend(value["CSETs"])
            expanded_control_RDs.extend(value["RDs"])
            expanded_control_tols.extend(value["tols"])

            ipv = control_CSETs.index(pv)
            coupling_indices[pv] = {"index": ipv, "coeffs": np.array(value["coeffs"], dtype=float)}
    return expanded_control_CSETs, expanded_control_RDs, expanded_control_tols, coupling_indices


class Evaluator(EvaluatorBase):
    def __init__(
        self,
        machineIO,
        control_CSETs: List[str],
        control_RDs: List[str],
        control_tols: Union[List[float], np.ndarray],
        control_couplings: Optional[Dict[str, Dict]] = None,
        monitor_PVs: Optional[List[str]] = None,
        ensure_set_kwargs: Optional[Dict] = None,
        fetch_data_kwargs: Optional[Dict] = None,
        set_manually: Optional[bool] = False,
        df_manipulators: Optional[List[Callable]] = None,
    ):
        _validate_control_couplings(control_couplings, control_CSETs)
        self.control_couplings = copy(control_couplings) if control_couplings is not None else None
        self._constructor_control_CSETs = list(control_CSETs)
        self._constructor_control_RDs = list(control_RDs)
        self._constructor_control_tols = np.asarray(control_tols, dtype=float)
        if control_couplings is not None:
            control_CSETs, control_RDs, control_tols, coupling_indices = _precompute_control_couplings_and_indices(
                control_couplings, control_CSETs, control_RDs, control_tols
            )
            self.coupling_indices = coupling_indices
        else:
            self.coupling_indices = None

        super().__init__(
            machineIO,
            control_CSETs=control_CSETs,
            control_RDs=control_RDs,
            control_tols=control_tols,
            monitor_PVs=monitor_PVs,
            ensure_set_kwargs=ensure_set_kwargs,
            fetch_data_kwargs=fetch_data_kwargs,
            set_manually=set_manually,
            df_manipulators=df_manipulators,
        )

    def _apply_control_couplings_runtime(self, x):
        new_x_values = []
        for pv, data in self.coupling_indices.items():
            new_x_values.extend(data["coeffs"] * x[data["index"]])
        return np.concatenate([x, np.asarray(new_x_values, dtype=float)])

    def _set_and_read(self, x, ensure_set_kwargs: Optional[Dict] = None, fetch_data_kwargs: Optional[Dict] = None):
        if self.coupling_indices is not None:
            x = self._apply_control_couplings_runtime(np.asarray(x, dtype=float).ravel())
        return super()._set_and_read(x, ensure_set_kwargs=ensure_set_kwargs, fetch_data_kwargs=fetch_data_kwargs)


class Evaluator_wBPMQ(Evaluator):
    def __init__(
        self,
        machineIO,
        control_CSETs: List[str],
        control_RDs: List[str],
        control_tols: Union[List[float], np.ndarray],
        BPM_names: List[str],
        control_couplings: Optional[Dict[str, Dict]] = None,
        model_type: str = "TIS161",
        monitor_PVs: Optional[List[str]] = None,
        ensure_set_kwargs: Optional[Dict] = None,
        fetch_data_kwargs: Optional[Dict] = None,
        set_manually: Optional[bool] = False,
        df_manipulators: Optional[List[Callable]] = None,
    ):
        if monitor_PVs is None:
            monitor_PVs = []
        else:
            assert isinstance(monitor_PVs, list), f"Expected monitor_PVs list, got {type(monitor_PVs).__name__}"

        BPM_names = sort_by_Dnum(BPM_names)
        self.raw2Q = raw2Q_processor(BPM_names=BPM_names, model_type=model_type)
        monitor_PVs = unique_preserve_order(monitor_PVs + [pv for pv in self.raw2Q.PVs2read if pv not in monitor_PVs])

        if df_manipulators is None:
            df_manipulators = [self.raw2Q]
        else:
            df_manipulators.append(self.raw2Q)

        super().__init__(
            machineIO,
            control_CSETs=control_CSETs,
            control_RDs=control_RDs,
            control_tols=control_tols,
            control_couplings=control_couplings,
            monitor_PVs=monitor_PVs,
            ensure_set_kwargs=ensure_set_kwargs,
            fetch_data_kwargs=fetch_data_kwargs,
            set_manually=set_manually,
            df_manipulators=df_manipulators,
        )


def _to_float_array(series):
    if series.empty:
        return np.array([], dtype=float)

    numeric = pd.to_numeric(series, errors="coerce")
    n_nan_after = numeric.isna().sum()
    n_nan_before = series.isna().sum() if series.dtype != object else 0

    if n_nan_after == n_nan_before:
        return numeric.to_numpy(dtype=float)
    return series.to_numpy()


def _series_history_to_dump(history: Dict[str, List[pd.Series]]) -> Dict[str, List[Dict[str, Any]]]:
    out = {}
    for key, values in (history or {}).items():
        records = []
        for value in values:
            if hasattr(value, "to_dict"):
                records.append(to_builtin(value.to_dict()))
            else:
                records.append(to_builtin(value))
        out[str(key)] = records
    return out


def _series_history_from_dump(history: Dict[str, List[Dict[str, Any]]]) -> Dict[str, List[pd.Series]]:
    restored = {"mean": [], "var": [], "ramping_mean": [], "ramping_var": []}
    for key, values in (history or {}).items():
        series_list = []
        for value in values or []:
            value = from_builtin(value)
            if isinstance(value, pd.Series):
                series_list.append(value)
            elif isinstance(value, dict):
                series_list.append(pd.Series(value))
            else:
                series_list.append(pd.Series(value))
        restored[str(key)] = series_list
    for key in ("mean", "var", "ramping_mean", "ramping_var"):
        restored.setdefault(key, [])
    return restored


def _df_manipulator_descriptor(func: Callable) -> Dict[str, Any]:
    owner = getattr(func, "__self__", None)
    if (
        isinstance(owner, SingleTaskObjectiveFunction)
        and getattr(func, "__name__", "") == "calculate_objectives_from_df"
    ):
        return {"kind": "SingleTaskObjectiveFunction.calculate_objectives_from_df"}
    return {
        "kind": "unsupported_callable",
        "module": getattr(func, "__module__", None),
        "qualname": getattr(func, "__qualname__", None),
        "repr": repr(func),
    }


def _find_single_task_objective(df_manipulators: Optional[List[Callable]]) -> Optional[SingleTaskObjectiveFunction]:
    for func in df_manipulators or []:
        owner = getattr(func, "__self__", None)
        if (
            isinstance(owner, SingleTaskObjectiveFunction)
            and getattr(func, "__name__", "") == "calculate_objectives_from_df"
        ):
            return owner
    return None


class OracleEvaluator(Evaluator):
    def __init__(
        self,
        machineIO,
        control_CSETs: List[str],
        control_RDs: List[str],
        control_tols: Union[List[float], np.ndarray],
        monitor_PVs: List[str],
        oracle_key_names: Dict[str, List[str]],
        control_couplings: Optional[Dict[str, Dict]] = None,
        ensure_set_kwargs: Optional[Dict] = None,
        fetch_data_kwargs: Optional[Dict] = None,
        set_manually: Optional[bool] = False,
        df_manipulators: Optional[List[Callable]] = None,
    ):
        super().__init__(
            machineIO,
            control_CSETs=control_CSETs,
            control_RDs=control_RDs,
            control_tols=control_tols,
            control_couplings=control_couplings,
            monitor_PVs=monitor_PVs,
            ensure_set_kwargs=ensure_set_kwargs,
            fetch_data_kwargs=fetch_data_kwargs,
            set_manually=set_manually,
            df_manipulators=df_manipulators,
        )
        self.oracle_key_names = {k: (v if isinstance(v, list) else [v]) for k, v in oracle_key_names.items()}

    def __call__(self, x=None):
        call_t0 = time.perf_counter()
        if x is None:
            df = self.read()
            timing = dict(getattr(self, "last_read_timing", {}) or {})
            result_t0 = time.perf_counter()
            mean = df.mean()
            out = {k: _to_float_array(mean[names]) for k, names in self.oracle_key_names.items()}
            timing["oracle_result_processing"] = time.perf_counter() - result_t0
            timing["oracle_call_total"] = time.perf_counter() - call_t0
            out["timing"] = timing
            return out

        df, ramping_df = self._set_and_read(x)
        timing = dict(getattr(self, "last_timing", {}) or {})
        result_t0 = time.perf_counter()
        mean = df.mean()
        out = {k: _to_float_array(mean[names]) for k, names in self.oracle_key_names.items()}
        timing["oracle_result_processing"] = time.perf_counter() - result_t0
        timing["oracle_call_total"] = time.perf_counter() - call_t0

        # if ramping_df is not None:
        #     ramping_mean = ramping_df.mean()
        #     out.update(
        #         {f"ramping_{k}": _to_float_array(ramping_mean[names]) for k, names in self.oracle_key_names.items()}
        #     )
        out["timing"] = timing
        return out
    

    def submit(self, x, ensure_set_kwargs=None, fetch_data_kwargs=None):
        """
        Submit a task to set and read data. 
        Returns a future that resolves to (df, ramping_df).
        """
        # If x is None, we just read. We'll wrap this in a future for consistency.
        if x is None:
            return self.executor.submit(self.read, fetch_data_kwargs=fetch_data_kwargs)
        
        # Otherwise use the standard Evaluator submit logic
        return super().submit(x, ensure_set_kwargs=ensure_set_kwargs, fetch_data_kwargs=fetch_data_kwargs)
    

    def get_result(self, future: concurrent.futures.Future) -> Dict[str, np.ndarray]:
        """
        Processes the raw DataFrames from the future into the dictionary format 
        returned by __call__.
        """
        result = future.result()
        
        # handle case where x was None (returns only df) vs x provided (returns df, ramping_df)
        if isinstance(result, tuple):
            df, ramping_df = result
        else:
            df, ramping_df = result, None

        mean = df.mean()
        return {k: _to_float_array(mean[names]) for k, names in self.oracle_key_names.items()}

    def to_dump_dict(self, *, include_history: bool = True) -> Dict[str, Any]:
        objective_function = _find_single_task_objective(self.df_manipulators)
        df_manipulators = [
            _df_manipulator_descriptor(func)
            for func in (self.df_manipulators or [])
        ]
        payload = {
            "class": type(self).__name__,
            "module": type(self).__module__,
            "format_version": 1,
            "machineIO": (
                self.machineIO.to_dump_dict(include_history=include_history)
                if hasattr(self.machineIO, "to_dump_dict")
                else None
            ),
            "objective_function": objective_function.to_dump_dict() if objective_function is not None else None,
            "config": {
                "control_CSETs": getattr(self, "_constructor_control_CSETs", self.control_CSETs),
                "control_RDs": getattr(self, "_constructor_control_RDs", self.control_RDs),
                "control_tols": getattr(self, "_constructor_control_tols", self.control_tols),
                "monitor_PVs": self.monitor_PVs,
                "oracle_key_names": self.oracle_key_names,
                "control_couplings": getattr(self, "control_couplings", None),
                "ensure_set_kwargs": self.ensure_set_kwargs,
                "fetch_data_kwargs": self.fetch_data_kwargs,
                "set_manually": self.set_manually,
                "df_manipulators": df_manipulators,
            },
            "state": {
                "last_read_timing": self.last_read_timing,
                "last_timing": self.last_timing,
            },
        }
        if include_history:
            payload["state"]["history"] = _series_history_to_dump(self.history)
        return to_builtin(payload)

    def _restore_state_from_dump_dict(self, state: Dict[str, Any], *, restore_history: bool = True) -> None:
        state = from_builtin(state or {})
        self.last_read_timing = dict(state.get("last_read_timing", {}) or {})
        self.last_timing = dict(state.get("last_timing", {}) or {})
        if restore_history:
            self.history = _series_history_from_dump(state.get("history", {}) or {})

    @classmethod
    def from_dump_dict(
        cls,
        data: Dict[str, Any],
        *,
        machineIO=None,
        objective_function: Optional[SingleTaskObjectiveFunction] = None,
        custom_objective_function: Optional[Callable] = None,
        restore_history: bool = True,
        use_epics: Optional[bool] = None,
        test: Optional[bool] = None,
        allow_unsupported_df_manipulators: bool = False,
    ) -> "OracleEvaluator":
        data = dict(data)
        config = dict(data.get("config", {}) or {})

        if machineIO is None:
            machine_data = data.get("machineIO")
            if machine_data is None:
                raise ValueError("OracleEvaluator dump does not contain machineIO data.")
            machineIO = construct_machineIO.from_dump_dict(
                machine_data,
                restore_history=restore_history,
                use_epics=use_epics,
                test=test,
            )

        objective_data = data.get("objective_function")
        if objective_function is None and objective_data is not None:
            objective_function = SingleTaskObjectiveFunction.from_dump_dict(
                objective_data,
                custom_function=custom_objective_function,
            )

        df_manipulators = []
        for descriptor in config.get("df_manipulators", []) or []:
            kind = descriptor.get("kind") if isinstance(descriptor, dict) else None
            if kind == "SingleTaskObjectiveFunction.calculate_objectives_from_df":
                if objective_function is None:
                    raise ValueError(
                        "Dump requests a SingleTaskObjectiveFunction dataframe manipulator "
                        "but no objective_function was available."
                    )
                df_manipulators.append(objective_function.calculate_objectives_from_df)
            elif allow_unsupported_df_manipulators:
                continue
            else:
                raise ValueError(
                    "Cannot restore unsupported dataframe manipulator from dump: "
                    f"{descriptor}"
                )

        oracle = cls(
            machineIO,
            control_CSETs=list(config["control_CSETs"]),
            control_RDs=list(config["control_RDs"]),
            control_tols=np.asarray(config["control_tols"], dtype=float),
            monitor_PVs=list(config.get("monitor_PVs", [])),
            oracle_key_names=dict(config["oracle_key_names"]),
            control_couplings=config.get("control_couplings", None),
            ensure_set_kwargs=dict(config.get("ensure_set_kwargs", {}) or {}),
            fetch_data_kwargs=dict(config.get("fetch_data_kwargs", {}) or {}),
            set_manually=bool(config.get("set_manually", False)),
            df_manipulators=df_manipulators or None,
        )
        oracle._restore_state_from_dump_dict(data.get("state", {}) or {}, restore_history=restore_history)
        return oracle

    def dump(self, path: str | Path, *, include_history: bool = True) -> Path:
        payload = {
            "format": "machineIO.OracleEvaluator.dump",
            "format_version": 1,
            "object": self.to_dump_dict(include_history=include_history),
        }
        return write_hdf5_dump(path, payload)

    @staticmethod
    def read_dump(path: str | Path) -> Dict[str, Any]:
        return read_hdf5_dump(path)

    @classmethod
    def from_dump(
        cls,
        path: str | Path,
        *,
        machineIO=None,
        objective_function: Optional[SingleTaskObjectiveFunction] = None,
        custom_objective_function: Optional[Callable] = None,
        restore_history: bool = True,
        use_epics: Optional[bool] = None,
        test: Optional[bool] = None,
        allow_unsupported_df_manipulators: bool = False,
    ) -> "OracleEvaluator":
        payload = cls.read_dump(path)
        return cls.from_dump_dict(
            payload.get("object", payload),
            machineIO=machineIO,
            objective_function=objective_function,
            custom_objective_function=custom_objective_function,
            restore_history=restore_history,
            use_epics=use_epics,
            test=test,
            allow_unsupported_df_manipulators=allow_unsupported_df_manipulators,
        )


class StatefulOracleEvaluator(OracleEvaluator):
    def __init__(
        self,
        machineIO,
        control_CSETs: List[str],
        control_RDs: List[str],
        control_tols: Union[List[float], np.ndarray],
        state_CSETs: List[str],
        state_RDs: List[str],
        state_tols: Union[List[float], np.ndarray],
        state_key_vals: Dict[str, List[float]],
        oracle_key_names: Dict[str, List[str]],
        monitor_PVs: List[str],
        control_couplings: Optional[Dict[str, Dict]] = None,
        ensure_set_kwargs: Optional[Dict] = None,
        fetch_data_kwargs: Optional[Dict] = None,
        set_manually: Optional[bool] = False,
        df_manipulators: Optional[List[Callable]] = None,
        state_df_manipulators: Optional[List[Callable]] = None,
    ):
        super().__init__(
            machineIO,
            control_CSETs=list(control_CSETs) + list(state_CSETs),
            control_RDs=list(control_RDs) + list(state_RDs),
            control_tols=list(control_tols) + list(state_tols),
            control_couplings=control_couplings,
            monitor_PVs=monitor_PVs,
            oracle_key_names=oracle_key_names,
            ensure_set_kwargs=ensure_set_kwargs,
            fetch_data_kwargs=fetch_data_kwargs,
            set_manually=set_manually,
            df_manipulators=df_manipulators,
        )
        self.state_CSETs = list(state_CSETs)
        self.state_RDs = list(state_RDs)
        self.state_tols = np.asarray(state_tols, dtype=float)
        self.state_key_vals = state_key_vals
        self.state_df_manipulators = state_df_manipulators
        for k, v in self.state_key_vals.items():
            assert len(v) == len(self.state_CSETs)

        self.oracle_key_names["state"] = ["state"]

    def __call__(self, x=None, s=None):
        call_t0 = time.perf_counter()
        if x is None:
            df = self.read()
            timing = dict(getattr(self, "last_read_timing", {}) or {})
            result_t0 = time.perf_counter()
            state_vals = df[self.state_CSETs].mean()
            state = s
            if state is None:
                for s_name, v in self.state_key_vals.items():
                    if np.all(np.abs(state_vals - v) < self.state_tols):
                        state = s_name
                        break

            state_manip_t0 = time.perf_counter()
            if self.state_df_manipulators is not None:
                for f in self.state_df_manipulators:
                    if callable(f):
                        df = f(df, s=state)
                    else:
                        raise ValueError(f"state_df_manipulators should be callable, got {type(f).__name__}")
            timing["oracle_state_df_manipulators"] = time.perf_counter() - state_manip_t0

            mean = df.mean()
            mean["state"] = state
            out = {k: _to_float_array(mean[names]) for k, names in self.oracle_key_names.items()}
            timing["oracle_result_processing"] = time.perf_counter() - result_t0
            timing["oracle_call_total"] = time.perf_counter() - call_t0
            out["timing"] = timing
            return out

        assert s is not None
        x_arr = np.asarray(x, dtype=float).ravel()
        s_arr = np.asarray(self.state_key_vals[s], dtype=float).ravel()
        full_x = np.concatenate([x_arr, s_arr])

        df, ramping_df = self._set_and_read(full_x)
        timing = dict(getattr(self, "last_timing", {}) or {})

        state_manip_t0 = time.perf_counter()
        if self.state_df_manipulators is not None:
            for f in self.state_df_manipulators:
                if callable(f):
                    df = f(df, s=s)
                else:
                    raise ValueError(f"state_df_manipulators should be callable, got {type(f).__name__}")
        timing["oracle_state_df_manipulators"] = time.perf_counter() - state_manip_t0

        result_t0 = time.perf_counter()
        mean = df.mean()
        mean["state"] = s
        out = {k: _to_float_array(mean[names]) for k, names in self.oracle_key_names.items()}
        timing["oracle_result_processing"] = time.perf_counter() - result_t0
        timing["oracle_call_total"] = time.perf_counter() - call_t0

        if ramping_df is not None:
            state_ramp_manip_t0 = time.perf_counter()
            if self.state_df_manipulators is not None:
                for f in self.state_df_manipulators:
                    if callable(f):
                        ramping_df = f(ramping_df, s=s)
                    else:
                        raise ValueError(f"state_df_manipulators should be callable, got {type(f).__name__}")
            timing["oracle_state_ramping_df_manipulators"] = time.perf_counter() - state_ramp_manip_t0

            # ramping_mean = ramping_df.mean()
            # if np.all(np.abs(ramping_mean[self.state_RDs] - mean[self.state_RDs]) < self.state_tols):
            #     ramping_mean["state"] = s
            #     out.update(
            #         {f"ramping_{k}": _to_float_array(ramping_mean[names]) for k, names in self.oracle_key_names.items()}
            #     )
        out["timing"] = timing
        return out



    def submit(self, x, s, ensure_set_kwargs=None, fetch_data_kwargs=None):
        """
        Concatenates control and state variables before submitting.
        """
        if x is None:
            future = super().submit(None, fetch_data_kwargs=fetch_data_kwargs)
        else:
            assert s is not None, "State 's' must be provided when setting 'x'"
            x_arr = np.asarray(x, dtype=float).ravel()
            s_arr = np.asarray(self.state_key_vals[s], dtype=float).ravel()
            full_x = np.concatenate([x_arr, s_arr])
            future = super().submit(full_x, ensure_set_kwargs=ensure_set_kwargs, fetch_data_kwargs=fetch_data_kwargs)
        
        # Attach the state to the future so get_result knows how to process it
        future._state_context = s
        return future

    def get_result(self, future: concurrent.futures.Future) -> Dict[str, np.ndarray]:
        """
        Applies state manipulators and returns the dictionary format with state info.
        """
        result = future.result()
        s = getattr(future, "_state_context", None)

        if isinstance(result, tuple):
            df, ramping_df = result
        else:
            df, ramping_df = result, None

        # Logic to determine state if it wasn't explicitly provided (for x=None cases)
        if s is None and x is None:
            state_vals = df[self.state_CSETs].mean()
            for s_name, v in self.state_key_vals.items():
                if np.all(np.abs(state_vals - v) < self.state_tols):
                    s = s_name
                    break

        # Apply state-specific manipulators
        if self.state_df_manipulators is not None:
            for f in self.state_df_manipulators:
                df = f(df, s=s)

        mean = df.mean()
        mean["state"] = s
        
        return {k: _to_float_array(mean[names]) for k, names in self.oracle_key_names.items()}
