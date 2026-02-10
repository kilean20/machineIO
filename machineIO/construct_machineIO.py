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
from typing import Optional, List, Union, Dict, Callable, Tuple
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
class _fetch_data_wrapper:
    def __init__(
        self,
        isOK_PVs=DEFAULT_isOK_PVs,
        isOK_vals=DEFAULT_isOK_vals,
        fetch_data_base=epics_fetch_data if DEFAULT_use_epics and epics_imported else phantasy_fetch_data if phantasy_imported else None,
        test=False,
    ):
        assert fetch_data_base is not None, "epics or phantasy import failed"
        self.fetch_data_base = fetch_data_base
        self.isOK_PVs = [] if isOK_PVs is None or test else list(isOK_PVs)
        self.isOK_vals = np.asarray([] if isOK_vals is None or test else isOK_vals, dtype=float)
        if not test and isOK_PVs is not None and isOK_vals is not None:
            assert len(isOK_PVs) == len(isOK_vals), "isOK_PVs and isOK_vals must have the same length"

    def __call__(self, pvlist: List[str], time_span: float, sample_interval: float, **kws):
        pvlist = unique_preserve_order(pvlist)
        pvlist_expanded = unique_preserve_order(pvlist + [pv for pv in self.isOK_PVs if pv not in pvlist])
        df = self.fetch_data_base(pvlist_expanded, time_span, sample_interval=sample_interval)

        # Prefer last sample for isOK PVs (more robust than mean for state PVs)
        while self.isOK_PVs and np.any(df[self.isOK_PVs].iloc[-1].to_numpy(dtype=float) != self.isOK_vals):
            logger.warning(f"notOK from {self.isOK_PVs} detected during fetch_data. Re-try in 5 sec... ")
            time.sleep(5)
            df = self.fetch_data_base(pvlist_expanded, time_span, sample_interval=sample_interval)

        return df[pvlist]


class _ensure_set_wrapper:
    def __init__(
        self,
        isOK_PVs=DEFAULT_isOK_PVs,
        isOK_vals=DEFAULT_isOK_vals,
        ensure_set_base=epics_ensure_set if DEFAULT_use_epics and epics_imported else phantasy_ensure_set if phantasy_imported else None,
        test=False,
    ):
        assert ensure_set_base is not None, "epics or phantasy import failed"
        self.ensure_set_base = ensure_set_base
        self.isOK_PVs = [] if isOK_PVs is None or test else list(isOK_PVs)
        self.isOK_vals = np.asarray([] if isOK_vals is None or test else isOK_vals, dtype=float)
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

        if self.isOK_PVs:
            try:
                ok_last = df[self.isOK_PVs].iloc[-1].to_numpy(dtype=float)
                if np.any(ok_last != self.isOK_vals):
                    return ret, None
            except Exception:
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

        if ret == "Timeout":
            if self._n_popup_ramping_issue < 2:
                self._n_popup_ramping_issue += 1
                # popup_ramping_not_OK("Ramping timeout. Please check PVs and press Enter.")
            else:
                logger.warning("'ramping_not_OK' issued 2 times already. Ignoring from now on...")

        time.sleep(self._ensure_set_timewait_after_ramp)
        self._record_history(
            caller="ensure_set",
            setpoint_pv=setpoint_pv,
            readback_pv=readback_pv,
            goal=goal,
            tol=tol,
            ret=ret,
            data=data,
        )
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
        data = self._fetch_data(
            pvlist,
            time_span=time_span or self._fetch_data_time_span,
            sample_interval=sample_interval or self._sample_interval,
            **kws,
        )
        if self._verbose if verbose is None else verbose:
            display(data)
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
        isOK_PVs=DEFAULT_isOK_PVs,
        isOK_vals=DEFAULT_isOK_vals,
        manual_CSETs: Optional[List[str]] = None,  # NEW
        test: bool = False,
    ):
        super().__init__(
            ensure_set_timeout=ensure_set_timeout,
            ensure_set_timewait_after_ramp=ensure_set_timewait_after_ramp,
            fetch_data_time_span=fetch_data_time_span,
            sample_interval=sample_interval,
            verbose=verbose,
            manual_CSETs=manual_CSETs,
        )
        self.test = test
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

        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    def read(self, fetch_data_kwargs: Optional[Dict] = None):
        fetch_data_kwargs = fetch_data_kwargs or self.fetch_data_kwargs
        df = self.machineIO.fetch_data(self.fetch_data_monitors, **fetch_data_kwargs)
        df = validate_df_rows(df, self.vector_PVs, self.vector_len)
        if self.df_manipulators is not None:
            for f in self.df_manipulators:
                if callable(f):
                    df = f(df)
                else:
                    raise ValueError(f"df_manipulators should be callable, got {type(f).__name__}")
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

        manual, auto = self._split_manual_controls(x_arr)

        # FULL manual mode: prompt only if in main thread (direct calls)
        if self.set_manually:
            if threading.current_thread() is threading.main_thread():
                df = pd.DataFrame([x_arr], columns=self.control_CSETs)
                try:
                    display(df)
                except Exception:
                    print(df.to_string(index=False))
                input("Set the above PVs manually, then press Enter to continue...")
            ret, ramping_data = "PutFinish", None
        else:
            # If there is a manual subset, prompt only in main thread (avoid worker-thread hangs)
            if manual is not None and threading.current_thread() is threading.main_thread():
                self._prompt_manual_set_subset(
                    manual["csets"],
                    manual["rds"],
                    manual["goals"],
                    manual["tols"],
                    timeout=ensure_set_kwargs.get("timeout", None),
                    sample_interval=ensure_set_kwargs.get("sample_interval", None),
                )

            extra_monitors = list(self.ensure_set_monitors)
            if manual is not None:
                extra_monitors = unique_preserve_order(extra_monitors + manual["csets"] + manual["rds"])

            ret, ramping_data = self.machineIO.ensure_set(
                auto["csets"],
                auto["rds"],
                auto["goals"],
                auto["tols"],
                extra_monitors=extra_monitors,
                **ensure_set_kwargs,
            )

            if ramping_data is not None:
                ramping_data = validate_df_rows(ramping_data, self.vector_PVs, self.vector_len)
                if self.df_manipulators is not None:
                    for f in self.df_manipulators:
                        if callable(f):
                            ramping_data = f(ramping_data)
                        else:
                            raise ValueError(f"df_manipulators should be callable, got {type(f).__name__}")

                ramping_mean, ramping_var = df_mean_var(ramping_data)
                with self._history_lock:
                    self.history["ramping_mean"].append(ramping_mean)
                    self.history["ramping_var"].append(ramping_var)

        data = self.read(fetch_data_kwargs=fetch_data_kwargs)
        mean, var = df_mean_var(data)
        with self._history_lock:
            self.history["mean"].append(mean)
            self.history["var"].append(var)

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
        """Dump the history dict (dataframes) to a pkl file."""
        if not filename.endswith(".pkl"):
            filename += ".pkl"
        history = self.get_history(ignore_index=True)
        with open(filename, "wb") as f:
            pd.to_pickle(history, f)


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
        if x is None:
            mean = self.read().mean()
            return {k: _to_float_array(mean[names]) for k, names in self.oracle_key_names.items()}

        df, ramping_df = self._set_and_read(x)
        mean = df.mean()
        out = {k: _to_float_array(mean[names]) for k, names in self.oracle_key_names.items()}

        # if ramping_df is not None:
        #     ramping_mean = ramping_df.mean()
        #     out.update(
        #         {f"ramping_{k}": _to_float_array(ramping_mean[names]) for k, names in self.oracle_key_names.items()}
        #     )
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
        if x is None:
            df = self.read()
            state_vals = df[self.state_CSETs].mean()
            state = s
            if state is None:
                for s_name, v in self.state_key_vals.items():
                    if np.all(np.abs(state_vals - v) < self.state_tols):
                        state = s_name
                        break

            if self.state_df_manipulators is not None:
                for f in self.state_df_manipulators:
                    if callable(f):
                        df = f(df, s=state)
                    else:
                        raise ValueError(f"state_df_manipulators should be callable, got {type(f).__name__}")

            mean = df.mean()
            mean["state"] = state
            return {k: _to_float_array(mean[names]) for k, names in self.oracle_key_names.items()}

        assert s is not None
        x_arr = np.asarray(x, dtype=float).ravel()
        s_arr = np.asarray(self.state_key_vals[s], dtype=float).ravel()
        full_x = np.concatenate([x_arr, s_arr])

        df, ramping_df = self._set_and_read(full_x)

        if self.state_df_manipulators is not None:
            for f in self.state_df_manipulators:
                if callable(f):
                    df = f(df, s=s)
                else:
                    raise ValueError(f"state_df_manipulators should be callable, got {type(f).__name__}")

        mean = df.mean()
        mean["state"] = s
        out = {k: _to_float_array(mean[names]) for k, names in self.oracle_key_names.items()}

        if ramping_df is not None:
            if self.state_df_manipulators is not None:
                for f in self.state_df_manipulators:
                    if callable(f):
                        ramping_df = f(ramping_df, s=s)
                    else:
                        raise ValueError(f"state_df_manipulators should be callable, got {type(f).__name__}")

            # ramping_mean = ramping_df.mean()
            # if np.all(np.abs(ramping_mean[self.state_RDs] - mean[self.state_RDs]) < self.state_tols):
            #     ramping_mean["state"] = s
            #     out.update(
            #         {f"ramping_{k}": _to_float_array(ramping_mean[names]) for k, names in self.oracle_key_names.items()}
            #     )
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