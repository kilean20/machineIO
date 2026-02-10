import os
import sys
import time
import datetime
import random
import warnings
import numpy as np
import pandas as pd
import concurrent
from typing import Optional, List, Union, Dict, Callable, Tuple
from copy import deepcopy as copy
from abc import ABC, abstractmethod
from threading import Lock
import logging
# logging.basicConfig(
#         level=logging.INFO,
#         format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
#     )
logger = logging.getLogger(__name__)



try:
    from .gui import popup_handler
    popup_ramping_not_OK = popup_handler(
    "Action required",
    "Ramping not OK. Manually adjust PV CSETs to jitter the power supply before continue."
    )
except:
    def popup_ramping_not_OK(message):
        """
        Fallback popup handler that prompts the user for input when ramping is not OK.
        Used in error handling scenarios to require manual intervention.
        """
        dummy = input(message)


from .util import display, cyclic_mean_var, suppress_outputs, sort_by_Dnum, validate_df_rows, df_mean, df_mean_var
script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(script_dir, 'models/BPMQ'))
from BPMQ_model import raw2Q_processor


# Default configuration values
DEFAULT_sample_interval = 0.2
DEFAULT_use_epics = False

try:
    from epics import caget as epics_caget
    from epics import caput as epics_caput
    from epics import caget_many as epics_caget_many
    from epics import caput_many as epics_caput_many
    epics_imported = True
    with suppress_outputs():
        if epics_caget("REA_EXP:ELMT") is not None:
            DEFAULT_isOK_PVs = None  # Skip check if machine is REA
            DEFAULT_isOK_vals = None
        else:
            DEFAULT_isOK_PVs = ["ACS_DIAG:CHP:STATE_RD"]   # is FRIB chopper on?
            DEFAULT_isOK_vals = [3]   # ACS_DIAG:CHP:STATE_RD=3 when FRIB chopper on
except ImportError:
    logger.warning("Failed to import 'epics'")
    epics_imported = False
    DEFAULT_isOK_PVs = None
    DEFAULT_isOK_vals = None


try:
    from phantasy import fetch_data as phantasy_fetch_data_orig
    from phantasy import ensure_set as phantasy_ensure_set_orig
    phantasy_imported = True
except ImportError:
    logger.warning("Failed to import 'phantasy'")
    phantasy_imported = False
    def phantasy_fetch_data_orig(pvlist, *args, **kwargs):
        return np.random.randn_like(pvlist), None



if phantasy_imported:
    def _resample_df(df: pd.DataFrame, 
                     sample_interval: float = DEFAULT_sample_interval
                     ) -> pd.DataFrame:    
        sample_interval = str(int(1000*sample_interval))+'ms'
        return df.bfill().ffill().resample(sample_interval).first().dropna()


    def phantasy_fetch_data(pvlist: List[str],
                            time_span: float = 1.0,
                            sample_interval: float = DEFAULT_sample_interval,
                            **kws,
                            ):
        _, df = phantasy_fetch_data_orig(pvlist, time_span=time_span, with_data=True, 
                                        data_opt={'with_timestamp': True,'fillna_method': 'none'})
        return _resample_df(df, sample_interval=sample_interval)

    def phantasy_ensure_set(setpoint_pv: List[str], 
                            readback_pv: List[str], 
                            goal: List[float], 
                            tol: List[float], 
                            timeout: float = 20.0, 
                            sample_interval = DEFAULT_sample_interval,
                            extra_monitors: List[str] = None, 
                            **kws,
                            ):
        ret, df = phantasy_ensure_set_orig(setpoint_pv, readback_pv, goal, 
                                          tol=tol, timeout=timeout, extra_monitors=extra_monitors,
                                          keep_data=True, fillna_method = 'none')
        return ret, _resample_df(df, sample_interval=sample_interval)


if epics_imported:
    def epics_fetch_data(
        pvlist: List[str], 
        time_span: float = 1.0,   
        sample_interval: float = DEFAULT_sample_interval,
        **kws,
        ):
        t0 = time.monotonic()
        index = [datetime.datetime.now()]
        data = [epics_caget_many(pvlist)]
        for pv,d in zip(pvlist,data[0]):
            if d is None:
                raise ValueError(f"Failed to fetch data from {pv}")
        while time.monotonic()-t0 < time_span:
            time.sleep(sample_interval)
            index.append(datetime.datetime.now())
            data.append(epics_caget_many(pvlist))
        df = pd.DataFrame(data,index=index,columns=pvlist).bfill().ffill()
        return df
    
    def epics_ensure_set(setpoint_pv: List[str], 
                         readback_pv: List[str], 
                         goal: List[float], 
                         tol: List[float], 
                         timeout: float = 30.0, 
                         sample_interval: float = DEFAULT_sample_interval,
                         extra_monitors: List[str] = None, 
                         **kws,
                         ):
        t0 = time.monotonic()
        epics_caput_many(setpoint_pv,goal)
        tol = np.asarray(tol)
        goal = np.asarray(goal)
        extra_monitors = extra_monitors if extra_monitors is not None else []
        pvlist = list(set(setpoint_pv + readback_pv + extra_monitors))
        val = epics_caget_many(pvlist)
        index = [datetime.datetime.now()]
        data = [val]
        nset = len(setpoint_pv)
        while time.monotonic()-t0 < timeout and np.any(np.abs(val[nset:2*nset]-goal)>tol):
            time.sleep(sample_interval)
            val = epics_caget_many(pvlist)
            index.append(datetime.datetime.now())
            data.append(val)
        df = pd.DataFrame(data,index=index,columns=pvlist).bfill().ffill()
        ret = 'PutFinish' if time.monotonic()-t0 < timeout else 'Timeout'
        return ret, df
    

class _fetch_data_wrapper:
    def __init__(self,
                 isOK_PVs  = DEFAULT_isOK_PVs, 
                 isOK_vals = DEFAULT_isOK_vals,
                 fetch_data_base = epics_fetch_data if DEFAULT_use_epics and epics_imported else phantasy_fetch_data if phantasy_imported else None,
                 test = False,
                 ):
        assert fetch_data_base is not None, "epics or fantasy import failed"
        self.fetch_data_base = fetch_data_base
        self.isOK_PVs = [] if isOK_PVs is None or test else isOK_PVs
        self.isOK_vals = np.asarray([] if isOK_vals is None or test else isOK_vals)
        if not test and isOK_PVs is not None and isOK_vals is not None:
            assert len(isOK_PVs) == len(isOK_vals), "isOK_PVs and isOK_vals must have the same length"
        self.test = test

    def __call__(self, pvlist: List[str], 
                 time_span: float, 
                 sample_interval: float,
                 **kws):
        pvlist = list(set(pvlist))
        pvlist_expanded = pvlist + [pv for pv in self.isOK_PVs if pv not in pvlist]
        df = self.fetch_data_base(pvlist_expanded,time_span,sample_interval=sample_interval)
        # while np.any(df[self.isOK_PVs].mean().values != self.isOK_vals):
        #     logger.warning(f"notOK from {self.isOK_PVs} detected during fetch_data. Re-try in 5 sec... ")
        #     time.sleep(5)
        #     df = self.fetch_data_base(pvlist_expanded,time_span,sample_interval=sample_interval)
        while np.any(df[self.isOK_PVs].mean().values != self.isOK_vals):
            logger.warning(f"notOK from {self.isOK_PVs} detected during fetch_data.")
            input(f"notOK from {self.isOK_PVs} detected during fetch_data. Enter any key after resolved... ")
            df = self.fetch_data_base(pvlist_expanded,time_span,sample_interval=sample_interval)
        return df[pvlist]
    
class _ensure_set_wrapper:
    def __init__(self,
                 isOK_PVs  = DEFAULT_isOK_PVs, 
                 isOK_vals = DEFAULT_isOK_vals,
                 ensure_set_base = epics_ensure_set if DEFAULT_use_epics and epics_imported else phantasy_ensure_set if phantasy_imported else None,
                 test = False,
                 ):
        assert ensure_set_base is not None, "epics or fantasy import failed"
        self.ensure_set_base = ensure_set_base
        self.isOK_PVs = [] if isOK_PVs is None or test else isOK_PVs
        self.isOK_vals = np.asarray([] if isOK_vals is None or test else isOK_vals)
        self.test = test

    def __call__(self, 
                setpoint_pv: List[str], 
                readback_pv: List[str], 
                goal: List[float], 
                tol: List[float],
                timeout: Optional[int],
                sample_interval: float,
                extra_monitors: Optional[List[str]] = None,
                **kws) -> Union[str, Union[None, pd.DataFrame]]:
        
        if self.test:
            return 'PutFinish', None
        extra_monitors = extra_monitors or []
        extra_monitors_expanded = list(set(extra_monitors + self.isOK_PVs))
        ret, df = self.ensure_set_base(setpoint_pv, readback_pv, goal, tol,
                                      timeout=timeout,
                                      sample_interval = sample_interval,
                                      extra_monitors = extra_monitors_expanded,
                                      **kws,
                                      )
        if np.any(df[self.isOK_PVs].mean().values != self.isOK_vals):
            return ret, None

        return ret, df[list(set(setpoint_pv+readback_pv+extra_monitors))]


class AbstractMachineIO(ABC):
    def __init__(self,
                 ensure_set_timeout: int = 20, 
                 ensure_set_timewait_after_ramp: float = 0.2,
                 fetch_data_time_span: float = 2.0,
                 sample_interval: float = DEFAULT_sample_interval,
                 verbose = False,
                ):
        self._ensure_set_timeout = ensure_set_timeout
        self._ensure_set_timewait_after_ramp = ensure_set_timewait_after_ramp
        self._fetch_data_time_span = fetch_data_time_span
        self._sample_interval = sample_interval
        self._verbose = verbose
        self._n_popup_ramping_issue = 0
        self._history_lock = Lock()
        self.history = []
      
    def _record_history(self, **kws):
        with self._history_lock:
            self.history.append({**kws, 'time': datetime.datetime.now()})
        
    @abstractmethod
    def _caget(self, pvname: str):
        raise NotImplementedError
        
    def caget(self, pvname: str):
        value = self._caget(pvname)
        self._record_history(caller='caget', pvname=pvname, read=value)
        return value
        
    @abstractmethod
    def _caput(self, pvname: str, value: Union[float, int]):
        raise NotImplementedError

    def caput(self, pvname: str, value: Union[float, int]):
        self._caput(pvname, value)
        self._record_history(caller='caput', pvname=pvname, set=value) 
    
    # @abstractmethod
    def _ensure_set(self,
                    setpoint_pv: List[str], 
                    readback_pv: List[str], 
                    goal: List[float], 
                    tol: List[float],
                    timeout: Optional[int],
                    sample_interval: float,
                    extra_monitors: Optional[List[str]] = None,
                    **kws) -> Union[str, Union[None, pd.DataFrame]]:
        return 'PutFinish', None

    def ensure_set (self,
                    setpoint_pv: List[str], 
                    readback_pv: List[str], 
                    goal: List[float], 
                    tol: List[float],
                    timeout: Optional[int] = None,
                    sample_interval: Optional[float] = None,
                    extra_monitors: Optional[List[str]] = None,
                    verbose: Union[bool, None] = None,
                    **kws):

        if self._verbose if verbose is None else verbose:
            print('Ramping in progress...')
            display(pd.DataFrame(np.asarray(goal).reshape(1, -1), columns=setpoint_pv))

        ret, data = self._ensure_set(setpoint_pv,readback_pv,goal,tol,
                                     timeout=timeout or self._ensure_set_timeout,
                                     sample_interval = sample_interval or self._sample_interval,
                                     extra_monitors = extra_monitors,
                                     **kws,
                                     )
        if ret == "Timeout":
            if self._n_popup_ramping_issue < 2:
                # popup_ramping_issue()
                self._n_popup_ramping_issue += 1
            else:
                logger.warning("'ramping_not_OK' issued 2 times already. Ignoring 'ramping_not_OK' issue from now on...")

        time.sleep(self._ensure_set_timewait_after_ramp)
        self._record_history(caller='ensure_set', setpoint_pv=setpoint_pv, readback_pv=readback_pv, goal=goal, tol=tol, ret=ret, data=data)
        return ret, data
                

    # @abstractmethod
    def _fetch_data(self, pvlist: List[str], 
                    time_span: float, 
                    sample_interval: float,
                    **kws):
        pass


    def fetch_data(self,
                   pvlist: List[str],
                   time_span: float = None, 
                   sample_interval : float = None,
                   verbose: Optional[bool] = None,
                   **kws):
        data = self._fetch_data(pvlist,
                                     time_span = time_span or self._fetch_data_time_span, 
                                     sample_interval = sample_interval or self._sample_interval,
                                     **kws,
                                     )
        if self._verbose if verbose is None else verbose:
            display(data)
        return data
    
    
class construct_machineIO(AbstractMachineIO):
    def __init__(self,
                 ensure_set_timeout: int = 20, 
                 ensure_set_timewait_after_ramp: float = 0.2,
                 fetch_data_time_span: float = 2.0,
                 sample_interval: float = DEFAULT_sample_interval,
                 verbose = False,
                 use_epics: bool = DEFAULT_use_epics,
                 isOK_PVs  = DEFAULT_isOK_PVs, 
                 isOK_vals = DEFAULT_isOK_vals,
                 test: bool = False,
                 ):
        super().__init__(
                    ensure_set_timeout = ensure_set_timeout, 
                    ensure_set_timewait_after_ramp = ensure_set_timewait_after_ramp,
                    fetch_data_time_span = fetch_data_time_span,
                    sample_interval = sample_interval,
                    verbose = verbose,
                    )
        self.test = test
        self.isOK_PVs = isOK_PVs
        self.isOK_vals = isOK_vals
        self._ensure_set = _ensure_set_wrapper(
            isOK_PVs = self.isOK_PVs, 
            isOK_vals = self.isOK_vals,
            ensure_set_base = epics_ensure_set if use_epics and epics_imported else phantasy_ensure_set if phantasy_imported else None,
            test = self.test,
        )
        self._fetch_data = _fetch_data_wrapper(
            isOK_PVs = self.isOK_PVs, 
            isOK_vals = self.isOK_vals,
            fetch_data_base = epics_fetch_data if use_epics and epics_imported else phantasy_fetch_data if phantasy_imported else None,
            test = self.test,
        )
        
    def _caget(self,pvname):
        if epics_imported:
            f = epics_caget(pvname)
        else:
            if self.test:
                warnings.warn("EPICS is not imported. caget will return fake zero")
                f = 0
            else:
                raise ValueError("EPICS is not imported. Cannot caget.")
        return f
            
    def _caput(self, pvname: str, value: Union[float, int]):
        if self.test:
            pass
        elif epics_imported:
            epics_caput(pvname, value)
        else:
            raise ValueError("EPICS is not imported. Cannot caput.")
    


def _validate_machineIO(machineIO):
    if not hasattr(machineIO, 'ensure_set') or not callable(getattr(machineIO, 'ensure_set', None)):
        raise TypeError("machineIO must have a callable `ensure_set` method.")
    if not hasattr(machineIO, 'fetch_data') or not callable(getattr(machineIO, 'fetch_data', None)):
        raise TypeError("machineIO must have a callable `fetch_data` method.")
    # if not hasattr(machineIO, 'history'):
    #     raise AttributeError("machineIO must have a `history` attribute.")

def _validate_control_CSETs(control_CSETs):
    """Validate control_CSETs."""
    if not isinstance(control_CSETs, list) or not all(isinstance(cset, str) for cset in control_CSETs):
        raise TypeError("control_CSETs must be a list of strings.")
    if len(control_CSETs) != len(set(control_CSETs)):
        raise ValueError("control_CSETs contains duplicate entries.")

def _validate_control_RDs(control_RDs, control_CSETs):
    """Validate control_RDs."""
    if not isinstance(control_RDs, list) or not all(isinstance(rd, str) for rd in control_RDs):
        raise TypeError("control_RDs must be a list of strings.")
    if len(control_RDs) != len(set(control_RDs)):
        raise ValueError("control_RDs contains duplicate entries.")
    if len(control_RDs) != len(control_CSETs):
        raise ValueError("The length of control_RDs must match the length of control_CSETs.")

def _validate_control_tols(control_tols, control_CSETs):
    """Validate control_tols."""
    if not isinstance(control_tols, (list, np.ndarray)) or not all(isinstance(tol, (int, float)) for tol in control_tols):
        raise TypeError("control_tols must be a list or numpy array of numbers.")
    if len(control_tols) != len(control_CSETs):
        raise ValueError("Length of control_tols must match length of control_CSETs.")

def _validate_monitor_PVs(monitor_PVs, control_CSETs, control_RDs):
    """Validate monitor_PVs."""
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
    

        
TISRAW_VECTOR_LENGTH = 68        
class EvaluatorBase:
    def __init__(self,
                 machineIO,
                 control_CSETs: List[str],
                 control_RDs  : List[str],
                 control_tols : Union[List[float], np.ndarray],
                 monitor_RDs : Optional[List[str]] = None,
                 ensure_set_kwargs: Optional[Dict] = None,
                 fetch_data_kwargs: Optional[Dict] = None,
                 set_manually : Optional[bool] = False, 
                 df_manipulators : Optional[List[Callable]] = None,
                 ):
        """
        Initialize the evaluator with machine I/O and data sets.

        Args:
            machineIO: Instance of AbstractMachineIO for hardware interaction.
            control_CSETs: List of control setpoint PVs.
            control_RDs: List of readback PVs corresponding to setpoints.
            control_tols: Tolerances for setpoint verification.
            monitor_RDs: Optional list of additional readback PVs (default: []).
            ensure_set_kwargs: Optional kwargs for ensure_set method (default: {}).
            fetch_data_kwargs: Optional kwargs for fetch_data method (default: {}).
            set_manually: If True, skip automatic setting (default: False).

        Raises:
            AssertionError: If input types are invalid.
        """
        _validate_machineIO(machineIO)
        _validate_control_CSETs(control_CSETs)
        _validate_control_RDs(control_RDs, control_CSETs)
        _validate_control_tols(control_tols, control_CSETs)
        _validate_monitor_PVs(monitor_RDs, control_CSETs, control_RDs)

        self.machineIO = machineIO
        self.ensure_set_kwargs = ensure_set_kwargs or {}
        self.fetch_data_kwargs = fetch_data_kwargs or {}
        # assert isinstance(control_CSETs, list), f"Expected control_CSETs to be of type list, but got {type(control_CSETs).__name__}"
        # assert isinstance(control_RDs  , list), f"Expected control_RDs to be of type list, but got {type(control_RDs).__name__}"
        # assert isinstance(control_tols , (list, np.ndarray)), f"Expected control_tols to be of type list or np.ndarray, but got {type(control_tols).__name__}"
        if monitor_RDs is None:
            monitor_RDs = []
        # assert isinstance(monitor_RDs , list), f"Expected monitor_RD to be of type list, but got {type(monitor_RDs).__name__}"
        
        self.control_CSETs = control_CSETs
        self.control_RDs   = control_RDs
        self.control_tols  = control_tols
        self.monitor_RDs = monitor_RDs
        self.set_manually = set_manually
        self.df_manipulators = df_manipulators
        
        self.fetch_data_monitors = list(set(control_CSETs + control_RDs + monitor_RDs))
        self.ensure_set_monitors = [m for m in self.fetch_data_monitors if m not in control_RDs and m not in control_CSETs]

        self.TISRAW_PVs = [pv for pv in self.fetch_data_monitors if ':TISRAW' in pv]
        self.vector_PVs = [self.TISRAW_PVs]
        self.vector_len = [TISRAW_VECTOR_LENGTH]
        self.scalar_PVs = list(set(self.fetch_data_monitors) - set(self.TISRAW_PVs))

        self._history_lock = Lock()
        self.history = {'mean':[],
                        'var':[],
                        'ramping_mean':[],
                        'ramping_var':[]}
        
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

        
    def read(self, fetch_data_kwargs: Optional[Dict] = None):
        fetch_data_kwargs = fetch_data_kwargs or self.fetch_data_kwargs
        df = self.machineIO.fetch_data(self.fetch_data_monitors,**fetch_data_kwargs)
        df = validate_df_rows(df, self.vector_PVs, self.vector_len)
        if self.df_manipulators is not None:
            for f in self.df_manipulators:
                if callable(f):
                    df = f(df)
                else:
                    raise ValueError(f"df_manipulators should be a callable, but got {type(f).__name__}")
        return df
        
    def _set_and_read(self, x,                 
        ensure_set_kwargs: Optional[Dict] = None,
        fetch_data_kwargs: Optional[Dict] = None,
        ):
        """
        Internal method to set the values and read the data.
        """
        ensure_set_kwargs = ensure_set_kwargs or self.ensure_set_kwargs
        fetch_data_kwargs = fetch_data_kwargs or self.fetch_data_kwargs
        
        if self.set_manually:
            ret, ramping_data = 'PutFinish', None
        else:
            ret, ramping_data = self.machineIO.ensure_set(self.control_CSETs, 
                                                          self.control_RDs, 
                                                          x,
                                                          self.control_tols,
                                                          extra_monitors=self.ensure_set_monitors,
                                                          **ensure_set_kwargs)
            if ramping_data is not None:
                ramping_data = validate_df_rows(ramping_data, self.vector_PVs, self.vector_len)
                if self.df_manipulators is not None:
                    for f in self.df_manipulators:
                        if callable(f):
                            ramping_data = f(ramping_data)
                        else:
                            raise ValueError(f"df_manipulator should be a callable, but got {type(f).__name__}")
                ramping_mean, ramping_var = df_mean_var(ramping_data)
                with self._history_lock:
                    self.history['ramping_mean'].append(ramping_mean)
                    self.history['ramping_var'].append(ramping_var)
                                                          
        data = self.read()
        mean, var = df_mean_var(data)
        with self._history_lock:
            self.history['mean'].append(mean)
            self.history['var'].append(var)
                      
        return data, ramping_data


    def submit(self, x, 
        ensure_set_kwargs = None,
        fetch_data_kwargs = None,
        ):
        """
        Submit a task to set and read data asynchronously.
        """
        if self.set_manually:
            display(pd.DataFrame(x,index=self.control_CSETs).T)
            if isinstance(x,np.ndarray):
                x_ = x.tolist()
            else:
                x_ = x
            if isinstance(self.control_tols,np.ndarray):
                tol = self.control_tols.tolist()
            else:
                tol = self.control_tols
            print(f"ensure_set({self.control_CSETs},{self.control_RDs},{x_},tol={tol},timeout={self.machineIO._ensure_set_timeout})")
            input("Set the above PVs and press any key to continue...")
        
        future = self.executor.submit(self._set_and_read, x, 
                                     ensure_set_kwargs = ensure_set_kwargs,
                                     fetch_data_kwargs = fetch_data_kwargs)
        return future

    def is_job_done(self, future: concurrent.futures.Future) -> bool:
        """
        Check if the submitted job is done.
        """
        return future.done()

    def get_result(self, future: concurrent.futures.Future) -> Tuple[Union[np.ndarray, pd.DataFrame], Union[np.ndarray, pd.DataFrame]]:
        """
        Retrieve the result from the future.
        """
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
        """
        Clear the history of the evaluator.
        """
        with self._history_lock:
            self.history = {'mean':[],
                            'var':[],
                            'ramping_mean':[],
                            'ramping_var':[]}
    
    # def dump_history(self, filename: str):
    #     """
    #     Dump the history to a pkl file.
    #     """
    #     if not filename.endswith('.pkl'):
    #         filename += '.pkl'
    #     history = self.get_history_df()
    #     with open(filename, 'wb') as f:
    #         pd.to_pickle(history, f)


def _validate_control_couplings(control_couplings, control_CSETs):
        """Validate control_couplings."""
        if control_couplings is None:
            return
        if not isinstance(control_couplings, dict):
            raise TypeError("control_couplings must be a dictionary.")
        for key, value in control_couplings.items():
            if not isinstance(key, str) or key not in control_CSETs:
                raise ValueError(f"control_couplings: Key '{key}' in control_couplings must be a string and in control_CSETs.")
            if not isinstance(value, dict):
                raise TypeError(f"Value for '{key}' must be a dictionary.")

            required_keys = {"CSETs", "RDs", "coeffs", "tols"}
            if not required_keys.issubset(value.keys()):
                missing = required_keys - set(value.keys())
                raise ValueError(f"control_couplings: Value for '{key}' must contain keys: {required_keys}. Missing: {missing}")

            csets = value["CSETs"]
            rds = value["RDs"]
            coeff = value["coeffs"]
            tol = value["tols"]
            if not isinstance(csets, list) or not all(isinstance(c, str) for c in csets):
                raise TypeError(f"control_couplings: 'CSETs' for '{key}' must be a list of strings.")
            if not isinstance(rds, list) or not all(isinstance(r, str) for r in rds):
                raise TypeError(f"control_couplings: 'RDs' for '{key}' must be a list of strings.")
            if not isinstance(coeff, (list, np.ndarray)) or not all(isinstance(c, (int, float)) for c in coeff):
                raise TypeError(f"control_couplings: 'coeffs' for '{key}' must be a list or numpy array of numbers.")
            if not isinstance(tol, (list, np.ndarray)) or not all(isinstance(t, (int, float)) for t in tol):
                raise TypeError(f"control_couplings: 'tols' for '{key}' must be a list or numpy array of numbers.")

            if len(coeff) != len(csets) or len(tol) != len(csets) or len(rds) != len(csets):
                raise ValueError(f"control_couplings: Lengths of 'coeffs', 'tols', and 'RDs' must match the length of 'CSETs' for '{key}'.")
                

def _precompute_control_couplings_and_indices(control_couplings, control_CSETs, control_RDs, control_tols):
    """
    Precomputes the expanded control_CSETs, control_RDs, and control_tols
    by applying control couplings during initialization, and also precomputes
    the indices for control couplings to optimize runtime performance.

    Returns:
    - A tuple containing:
        1. Expanded control_CSETs
        2. Expanded control_RDs
        3. Expanded control_tols
        4. A dictionary of coupling indices with associated coefficients
    """
    expanded_control_CSETs = list(control_CSETs)  # Create mutable copies
    expanded_control_RDs = list(control_RDs)
    expanded_control_tols = list(control_tols)
    coupling_indices = {}

    if control_couplings:
        for pv, value in control_couplings.items():
            # Extend control CSETs, RDs, and tols with coupled values
            expanded_control_CSETs.extend(value["CSETs"])
            expanded_control_RDs.extend(value["RDs"])
            expanded_control_tols.extend(value["tols"])

            # Precompute the coupling indices for runtime
            ipv = control_CSETs.index(pv)
            coupling_indices[pv] = {
                "index": ipv,
                "coeffs": np.array(value["coeffs"])
            }
    return expanded_control_CSETs, expanded_control_RDs, expanded_control_tols, coupling_indices



class Evaluator(EvaluatorBase):
    def __init__(self,
                 machineIO,
                 control_CSETs: List[str],
                 control_RDs  : List[str],
                 control_tols : Union[List[float], np.ndarray],
                 control_couplings: Optional[Dict[str,Dict]] = None,
                 monitor_RDs : Optional[List[str]] = None,
                 ensure_set_kwargs: Optional[Dict] = None,
                 fetch_data_kwargs: Optional[Dict] = None,
                 set_manually : Optional[bool] = False, 
                 df_manipulators : Optional[List[Callable]] = None,
                 ):
        """
        Initialize the evaluator with machine I/O and data sets.

        Args:
            machineIO: Instance of AbstractMachineIO for hardware interaction.
            control_CSETs: List of control setpoint PVs.
            control_RDs: List of readback PVs corresponding to setpoints.
            control_tols: Tolerances for setpoint verification.
            control_couplings: CSETs that need to be coupled with control_CSETs.         
            monitor_RDs: Optional list of additional readback PVs (default: []).
            ensure_set_kwargs: Optional kwargs for ensure_set method (default: {}).
            fetch_data_kwargs: Optional kwargs for fetch_data method (default: {}).
            set_manually: If True, skip automatic setting (default: False).
        Raises:
            AssertionError: If input types are invalid.
        """
        _validate_control_CSETs(control_CSETs)
        _validate_control_RDs(control_RDs, control_CSETs)
        _validate_control_tols(control_tols, control_CSETs)
        _validate_control_couplings(control_couplings, control_CSETs)
        if control_couplings is not None:
            (control_CSETs,
             control_RDs,
             control_tols,
             coupling_indices) = _precompute_control_couplings_and_indices(
                control_couplings, control_CSETs, control_RDs, control_tols)
            self.coupling_indices = coupling_indices
        else:
            self.coupling_indices = None
        super().__init__(machineIO, 
                         control_CSETs= control_CSETs,      
                         control_RDs= control_RDs,
                         control_tols= control_tols,
                         monitor_RDs= monitor_RDs,
                         ensure_set_kwargs= ensure_set_kwargs,
                         fetch_data_kwargs= fetch_data_kwargs,
                         set_manually= set_manually,
                         df_manipulators= df_manipulators
                         )

    # def _apply_control_couplings_runtime(self, x):
    #     """
    #     # x is for the *original* control_CSETs only (without coupled control sets)
    #     Parameters:
    #     - x: Input array for initial control_CSETs.

    #     Returns:
    #     - Expanded x array after applying control couplings.
    #     """
    #     new_x_values = []
    #     for pv, data in self.coupling_indices.items():
    #         # print('_apply_control_couplings_runtime: data["index"],x[data["index"]]',data["index"],x[data["index"]])
    #         new_x_values.extend(data["coeffs"] * x[data["index"]])
    #     return np.concatenate([x, new_x_values])


    def _apply_control_couplings_runtime(self, x: np.ndarray) -> np.ndarray:
    # x is for the *original* control_CSETs only (pre-expansion)
        new_vals = []
        for pv, info in self.coupling_indices.items():
            idx = info["index"]
            coeffs = np.asarray(info["coeffs"], dtype=float).ravel()
            base = float(x[idx])
            new_vals.extend((coeffs * base).tolist())
        return np.concatenate([np.asarray(x, dtype=float).ravel(), np.asarray(new_vals, dtype=float)])


    def _set_and_read(self, x,                 
        ensure_set_kwargs: Optional[Dict] = None,
        fetch_data_kwargs: Optional[Dict] = None,
        ):
        if self.coupling_indices is not None:
            x = self._apply_control_couplings_runtime(np.asarray(x, dtype=float).ravel())
        return super()._set_and_read(x, 
                                     ensure_set_kwargs=ensure_set_kwargs, 
                                     fetch_data_kwargs=fetch_data_kwargs)


class Evaluator_wBPMQ(Evaluator):
    def __init__(self,
                 machineIO,
                 control_CSETs: List[str],
                 control_RDs  : List[str],
                 control_tols : Union[List[float], np.ndarray],
                 BPM_names  : List[str],
                 control_couplings: Optional[Dict[str,Dict]] = None,
                 model_type : str = 'TIS161',
                 monitor_RDs : Optional[List[str]] = None,
                 ensure_set_kwargs: Optional[Dict] = None,
                 fetch_data_kwargs: Optional[Dict] = None,
                 set_manually : Optional[bool] = False, 
                 df_manipulators : Optional[List[Callable]] = None,
                 ):
           
        if monitor_RDs is None:
            monitor_RDs = []
        else:
            assert isinstance(monitor_RDs, list), f"Expected monitor_RDs to be a list, but got {type(monitor_RDs).__name__}"

        BPM_names = sort_by_Dnum(BPM_names)
        self.raw2Q = raw2Q_processor(BPM_names=BPM_names,model_type=model_type)
        monitor_RDs = monitor_RDs + [pv for pv in self.raw2Q.PVs2read if pv not in monitor_RDs]

        if df_manipulators is None:
            df_manipulators = [self.raw2Q]
        else:
             df_manipulators.append(self.raw2Q)

        super().__init__(machineIO, 
                         control_CSETs= control_CSETs, 
                         control_RDs  = control_RDs,
                         control_tols = control_tols,
                         control_couplings = control_couplings,
                         monitor_RDs = monitor_RDs,
                         ensure_set_kwargs = ensure_set_kwargs,
                         fetch_data_kwargs = fetch_data_kwargs,
                         set_manually   = set_manually, 
                         df_manipulators = df_manipulators
                         )
        
def _to_float_array(series):
    """
    Convert a pandas Series to a float numpy array if elements are numeric-like.
    If the series contains strings or mixed types, return original .values unchanged.
    """
    if series.empty:
        return np.array([], dtype=float)

    # Try numeric conversion
    numeric = pd.to_numeric(series, errors='coerce')
    # Count how many became NaN after coercion
    n_nan_after = numeric.isna().sum()
    n_nan_before = series.isna().sum() if series.dtype != object else 0

    # If coercion did not introduce *additional* NaNs → numeric is safe
    if n_nan_after == n_nan_before:
        return numeric.to_numpy(dtype=float)

    # Otherwise, likely non-numeric strings present → return original
    return series.to_numpy()

class OracleEvaluator(Evaluator):        
    def __init__(self,
                 machineIO,
                 control_CSETs: List[str],
                 control_RDs  : List[str],
                 control_tols : Union[List[float], np.ndarray],
                 monitor_RDs : List[str],
                 oracle_key_names : Dict[str,List[str]],
                 control_couplings: Optional[Dict[str,Dict]] = None,
                 ensure_set_kwargs: Optional[Dict] = None,
                 fetch_data_kwargs: Optional[Dict] = None,
                 set_manually : Optional[bool] = False, 
                 df_manipulators : Optional[List[Callable]] = None,
                 ):
        super().__init__(machineIO, 
                         control_CSETs= control_CSETs, 
                         control_RDs  = control_RDs,
                         control_tols = control_tols,
                         control_couplings = control_couplings,
                         monitor_RDs = monitor_RDs,
                         ensure_set_kwargs = ensure_set_kwargs,
                         fetch_data_kwargs = fetch_data_kwargs,
                         set_manually   = set_manually, 
                         df_manipulators = df_manipulators,
                         )
        # Normalize mapping values to lists for consistency
        self.oracle_key_names = {k: (v if isinstance(v, list) else [v])
                                 for k, v in oracle_key_names.items()}
    def __call__(self,x=None):
        if x is None:
            mean = self.read().mean()
            return {k:_to_float_array(mean[names]) for k, names in self.oracle_key_names.items()}
        else:
            df, ramping_df = self._set_and_read(x)
            mean = df.mean()
            out = {k:_to_float_array(mean[names]) for k, names in self.oracle_key_names.items()}
            if ramping_df is not None:
                ramping_mean = ramping_df.mean()
                out.update({f'ramping_{k}':_to_float_array(ramping_mean[names]) for k, names in self.key_names.items()})
            return out
                

class StatefulOracleEvaluator(OracleEvaluator):        
    def __init__(self,
                 machineIO,
                 control_CSETs: List[str],
                 control_RDs  : List[str],
                 control_tols : Union[List[float], np.ndarray],
                 state_CSETs  : List[str],
                 state_RDs    : List[str],
                 state_tols   : Union[List[float], np.ndarray],
                 state_key_vals : Dict[str,List[float]],
                 oracle_key_names : Dict[str,List[str]],
                 monitor_RDs  : List[str],
                 control_couplings: Optional[Dict[str,Dict]] = None,
                 ensure_set_kwargs: Optional[Dict] = None,
                 fetch_data_kwargs: Optional[Dict] = None,
                 set_manually : Optional[bool] = False, 
                 df_manipulators : Optional[List[Callable]] = None,
                 state_df_manipulators : Optional[List[Callable]] = None,
                 ):
        super().__init__(machineIO, 
                         control_CSETs= list(control_CSETs) + list(state_CSETs),
                         control_RDs  = list(control_RDs)   + list(state_RDs),
                         control_tols = list(control_tols)  + list(state_tols),
                         control_couplings = control_couplings,
                         monitor_RDs = monitor_RDs,
                         oracle_key_names = oracle_key_names,
                         ensure_set_kwargs = ensure_set_kwargs,
                         fetch_data_kwargs = fetch_data_kwargs,
                         set_manually   = set_manually, 
                         df_manipulators = df_manipulators,
                         )
        self.state_CSETs = state_CSETs
        self.state_RDs = state_RDs
        self.state_tols = np.asarray(state_tols, dtype=float)
        self.state_key_vals = state_key_vals
        self.state_df_manipulators = state_df_manipulators
        for k,v in self.state_key_vals.items():
            assert len(v)==len(self.state_CSETs)
        self.oracle_key_names['state'] = ['state']
        
    def __call__(self,x=None,s=None):
        if x is None :
            df = self.read()
            state_vals = df[self.state_CSETs].mean()
            # print("s",s)
            state = s
            if state is None:
                for s,v in self.state_key_vals.items():
                    # print("s",s)
                    if np.all(np.abs(state_vals - v) < self.state_tols ):
                        state = s
                        break
            if state is None:
                raise ValueError('could not identify state')
            if self.state_df_manipulators is not None:
                for f in self.state_df_manipulators:
                    if callable(f):
                        df = f(df,s=state)
                    else:
                        raise ValueError(f"state_df_manipulators should be a callable, but got {type(f).__name__}")
            mean = df.mean()
            mean['state'] = state
            return {k:_to_float_array(mean[names]) for k, names in self.oracle_key_names.items()}
        else:
            assert s is not None
            x_arr = np.asarray(x, dtype=float).ravel()
            # print("s",s)
            # print("self.state_key_vals[s]",self.state_key_vals[s])
            # print("np.asarray(self.state_key_vals[s], dtype=float)",np.asarray(self.state_key_vals[s], dtype=float))
            s_arr = np.asarray(self.state_key_vals[s], dtype=float).ravel()
            full_x = np.concatenate([x_arr, s_arr])
            # print("full_x",full_x)
            # print("self.control_CSETs",self.control_CSETs)
            # print("self._set_and_read(full_x)",self._set_and_read(full_x))
            df, ramping_df = self._set_and_read(full_x)
            if self.state_df_manipulators is not None:
                for f in self.state_df_manipulators:
                    if callable(f):
                        df = f(df,s=s)
                    else:
                        raise ValueError(f"state_df_manipulators should be a callable, but got {type(f).__name__}")
            mean = df.mean()
            mean['state'] = s
            out = {k:_to_float_array(mean[names]) for k, names in self.oracle_key_names.items()}
            if ramping_df is not None:
                if self.state_df_manipulators is not None:
                    for f in self.state_df_manipulators:
                        if callable(f):
                            ramping_df = f(ramping_df,s=s)
                        else:
                            raise ValueError(f"state_df_manipulators should be a callable, but got {type(f).__name__}")
                ramping_mean = ramping_df.mean()
                if np.all(np.abs(ramping_mean[self.state_RDs] - mean[self.state_RDs]) < self.state_tols):
                    ramping_mean['state'] = s
                    out.update({f'ramping_{k}':_to_float_array(ramping_mean[names]) for k, names in self.oracle_key_names.items()})
            return out