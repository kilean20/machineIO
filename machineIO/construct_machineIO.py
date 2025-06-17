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
from .gui import popup_handler
popup_ramping_not_OK = popup_handler(
    "Action required",
    "Ramping not OK. Manually adjust PV CSETs to jitter the power supply before continue."
)
from .utils import display, cyclic_mean_var, suppress_outputs, sort_by_Dnum, validate_df_rows, df_mean, df_mean_var
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
        tol = np.array(tol)
        goal = np.array(goal)
        extra_monitors = extra_monitors if extra_monitors is not None else []
        pvlist = list(set(setpoint_pv + readback_pv + extra_monitors))
        val = epics_caget_many(pvlist)
        index = [datetime.datetime.now()]
        data = [val]
        while time.monotonic()-t0 < timeout and np.any(np.abs(val-goal)>tol):
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
        self.isOK_vals = np.array([] if isOK_vals is None or test else isOK_vals)
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
        while np.any(df[self.isOK_PVs].mean().values != self.isOK_vals):
            logger.warning(f"notOK from {self.isOK_PVs} detected during fetch_data. Re-try in 5 sec... ")
            time.sleep(5)
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
        self.isOK_vals = np.array([] if isOK_vals is None or test else isOK_vals)
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

        return df[list(set(setpoint_pv+readback_pv+extra_monitors))]


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
            display(pd.DataFrame(np.array(goal).reshape(1, -1), columns=setpoint_pv))

        ret, data = self._ensure_set(setpoint_pv,readback_pv,goal,tol,
                                     timeout=timeout or self._ensure_set_timeout,
                                     sample_interval = sample_interval or self._sample_interval,
                                     extra_monitors = extra_monitors,
                                     **kws,
                                     )
        if ret == "Timeout":
            if self._n_popup_ramping_issue < 2:
                popup_ramping_issue()
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
                warn("EPICS is not imported. caget will return fake zero")
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
    
        
TISRAW_VECTOR_LENGTH = 68        
class Evaluator:
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
                 return_meanvar_only : Optional[bool] = False, 
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
        self.machineIO = machineIO
        self.ensure_set_kwargs = ensure_set_kwargs or {}
        self.fetch_data_kwargs = fetch_data_kwargs or {}
        assert isinstance(control_CSETs, list), f"Expected control_CSETs to be of type list, but got {type(control_CSETs).__name__}"
        assert isinstance(control_RDs  , list), f"Expected control_RDs to be of type list, but got {type(control_RDs).__name__}"
        assert isinstance(control_tols , (list, np.ndarray)), f"Expected control_tols to be of type list or np.ndarray, but got {type(control_tols).__name__}"
        if monitor_RDs is None:
            monitor_RDs = []
        assert isinstance(monitor_RDs , list), f"Expected monitor_RD to be of type list, but got {type(monitor_RDs).__name__}"
        
        self.control_CSETs = control_CSETs
        self.control_RDs   = control_RDs
        self.control_tols  = control_tols
        self.monitor_RDs = monitor_RDs
        self.set_manually = set_manually
        self.df_manipulators = df_manipulators
        self.return_meanvar_only = return_meanvar_only
        
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
        
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        
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
                
        if self.return_meanvar_only:
            data = (mean, var)
            ramping_data = (ramping_mean, ramping_var) if ramping_data is not None else None
        
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
    
    def dump_history(self, filename: str):
        """
        Dump the history to a pkl file.
        """
        if not filename.endswith('.pkl'):
            filename += '.pkl'
        history = self.get_history_df()
        with open(filename, 'wb') as f:
            pd.to_pickle(history, f)
            
class Evaluator_wBPMQ(Evaluator):        
    def __init__(self,
                 machineIO,
                 control_CSETs: List[str],
                 control_RDs  : List[str],
                 control_tols : Union[List[float], np.ndarray],
                 BPM_names  : List[str],
                 model_type : str = 'TIS161',
                 monitor_RDs : Optional[List[str]] = None,
                 ensure_set_kwargs: Optional[Dict] = None,
                 fetch_data_kwargs: Optional[Dict] = None,
                 set_manually : Optional[bool] = False, 
                 df_manipulators : Optional[List[Callable]] = None,
                 return_meanvar_only : Optional[bool] = False, 
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
                         monitor_RDs = monitor_RDs,
                         ensure_set_kwargs = ensure_set_kwargs,
                         fetch_data_kwargs = fetch_data_kwargs,
                         set_manually   = set_manually, 
                         df_manipulators = df_manipulators,
                         return_meanvar_only = return_meanvar_only,
                         )