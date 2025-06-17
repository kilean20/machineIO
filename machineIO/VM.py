import numpy as np
import pandas as pd
import datetime
import time
from typing import List, Union, Optional, Callable
from threading import Lock

try:
    from IPython.display import display as _display
except ImportError:
    _display = print
def display(obj):
    try:
        _display(obj)
    except:
        print(obj)


# _PV_bounds_pattern = {
#     ':PSC':(-20,20),
#     ':PSQ_D':(10,200),
#     ':XPOS_RD':(-10,10),
#     ':YPOS_RD':(-10,10),
#     ':beamQ'  :(-30,30),
#     ':MAG_RD' :(0,0.1),
#     ':CURRENT_RD' :(0,30),
#     'FC_D' :(0,30),
#     'BCM_D' :(0,30),
#     #':PHASE_RD' :(-90,90),
# }

class RandomNeuralNetwork:
    def __init__(self, n_input: int, n_output: int, hidden_layers: Optional[List[int]] = None, 
                 activation_functions: Optional[List[str]] = None):
        self.n_input = n_input
        self.n_output = n_output
        
        # Determine the minimum and maximum number of nodes in hidden layers based on input and output dimensions
        min_node = int(np.clip(np.log2((n_input+n_output)**0.2), a_min=4, a_max=11))
        max_node = int(np.clip(np.log2((n_input+n_output)), a_min=min(min_node+3,10), a_max=12))
        
        # Initialize hidden_layers if not provided with random values within a specified range
        if hidden_layers is None:
            hidden_layers = [2**np.random.randint(min_node, max_node) for _ in range(np.random.randint(4, 7))]  # 4, 5, or 6 layers
        
        # Calculate the total number of layers and layer dimensions
        self.n_layers: int = len(hidden_layers) + 2
        self.layer_dims: List[int] = [n_input] + hidden_layers + [n_output]
        
        # Initialize activation functions if not provided with random choices for hidden layers and None for the output layer
        if activation_functions is None:
            # activation_functions = [np.random.choice(['elu', 'sin', 'cos', 'tanh', 'sinc']) for i in range(self.n_layers-1)]
            activation_functions = ['sin' for i in range(self.n_layers-1)]
            activation_functions.append(None)  # no activation on the last layer
        
        # Store layer activation functions and initialize network parameters
        self.activation_functions: List[Union[str, None]] = activation_functions
        self.parameters: dict = self.initialize_parameters()
        
        # Generate random inputs for normalization calculation
        self.output_min: float = 0
        self.output_max: float = 1
        random_inputs = np.random.rand(1024*64, n_input)
        random_outputs = self(random_inputs)
        self.output_min = np.min(random_outputs, axis=0)
        self.output_max = np.max(random_outputs, axis=0)
        

    def initialize_parameters(self) -> dict:
        # Initialize weights and biases for each layer using He initialization
        parameters = {}
        for l in range(1, self.n_layers):
            # scale_weights = np.sqrt(2.0 / self.layer_dims[l - 1])  # He initialization for weights
            scale_weights = np.sqrt(0.5 / self.layer_dims[l - 1])  
            parameters[f'W{l}'] = np.random.randn(self.layer_dims[l], self.layer_dims[l-1]) * scale_weights

            # Initialize biases with small random values
            # scale_biases = np.sqrt(0.5 / self.layer_dims[l])
            parameters[f'b{l}'] = np.random.randn(self.layer_dims[l], 1)# * scale_biases

        return parameters
    
    def normalize_output(self, output: np.ndarray) -> np.ndarray:
        # Normalize output based on mean and standard deviation
        return (output - self.output_min) / (self.output_max - self.output_min)

    def activate(self, Z: np.ndarray, activation_function: Union[str, None]) -> np.ndarray:
        # Apply activation functions based on the specified function
        if activation_function == 'sigmoid':
            return 1 / (1 + np.exp(-Z))
        elif activation_function == 'tanh':
            return np.tanh(Z)
        elif activation_function == 'relu':
            return np.maximum(0, Z)
        elif activation_function == 'elu':
            return np.where(Z > 0, Z, np.exp(Z) - 1)
        elif activation_function == 'sin':
            return np.sin(Z)
        elif activation_function == 'cos':
            return np.cos(Z)
        elif activation_function == 'sinc':
            return np.sinc(Z)
        else:
            return Z

    def __call__(self, X: np.ndarray) -> np.ndarray:
        if isinstance(X,list):
            X = np.array(X)
        # assert X.shape[1] == self.n_input
        A = np.atleast_2d(X).T  # Transpose the input to make it compatible with matrix multiplication
        for l in range(1, self.n_layers):
            W = self.parameters[f'W{l}']
            b = self.parameters[f'b{l}']
            Z = np.dot(W, A) + b
            A = self.activate(Z, self.activation_functions[l])
            
        # Transpose the result back to (n_batch, n_output) and then normalize
        y = self.normalize_output(A.T)
        if y.shape[0] == 1:
            return y[0]
        else:
            return y


class VirtualMachineIO:
    def __init__(self,
                 control_CSETs: List[str],
                 control_RDs: List[str],
                 monitor_RDs: List[str],
                 control_min: Optional[List[float]] = None,
                 control_max: Optional[List[float]] = None,
                 monitor_min: Optional[List[float]] = None,
                 monitor_max: Optional[List[float]] = None,
                 fun: Optional[Callable] = None,
                 fetch_data_time_span: Optional[float] = 2.0,
                 sample_interval: Optional[float] = 0.2,
                 ramping_rate: Optional[float] = None,
                 verbose: Optional[bool] = False,
                 ):
        if monitor_RDs is None:
            monitor_RDs = []
            
        assert isinstance(control_CSETs, list), f"Expected control_CSETs to be of type list, but got {type(control_CSETs).__name__}"
        assert isinstance(control_RDs,   list), f"Expected control_RDs to be of type list, but got {type(control_RDs).__name__}"
        assert isinstance(monitor_RDs,   list), f"Expected monitor_RDs to be of type list, but got {type(monitor_RDs).__name__}"

        assert len(control_CSETs) == len(set(control_CSETs)), "control_CSETs contains duplicates"
        assert len(control_RDs)   == len(set(control_RDs  )), "control_RDs contains duplicates"
        assert len(monitor_RDs)   == len(set(monitor_RDs  )), "monitor_RDs contains duplicates"

        assert set(control_CSETs).isdisjoint(set(control_RDs)), "control_CSETs and control_RDs should be disjoint. If control_RDs is None, all PVs in control_CSETs must end with '_CSET'."
        assert set(monitor_RDs).isdisjoint(set(control_CSETs + control_RDs)), "monitor_RDs should be disjoint from control_CSETs and control_RDs."

        self.control_CSETs = control_CSETs
        self.control_RDs = control_RDs
        self.monitor_RDs = monitor_RDs
        self.all_PVs = self.control_CSETs + self.control_RDs + self.monitor_RDs

        self.control_min = np.array(control_min) if control_min is not None else np.zeros(len(control_CSETs))
        self.control_max = np.array(control_max) if control_max is not None else np.ones (len(control_CSETs))
        self.monitor_min = np.array(monitor_min) if monitor_min is not None else np.zeros(len(monitor_RDs  ))
        self.monitor_max = np.array(monitor_max) if monitor_max is not None else np.ones (len(monitor_RDs  ))
            
        if fun is None:
            randNN = RandomNeuralNetwork(len(self.control_CSETs), len(self.monitor_RDs))
            x_test = np.random.rand(1024*32, len(self.control_CSETs))
            y_test = randNN(x_test)
            self.y_min = np.min(y_test, axis=0)
            self.y_max = np.max(y_test, axis=0)
            # Define normalized objective function
            def fun(x):
                xn = (np.array(x).reshape(1,-1)-self.control_min)/(self.control_max-self.control_min)
                yn = randNN(xn)
                return yn*(self.monitor_max-self.monitor_min) + self.monitor_min
        self.fun = fun
        self.dt = sample_interval
        self.fetch_data_time_span = fetch_data_time_span
        if ramping_rate is None:
            ramping_rate = 0.1*(self.control_max-self.control_min)
        self.ramping_rate = ramping_rate

        self(np.random.rand(len(self.control_CSETs))*(self.control_max-self.control_min) + self.control_min)
        self.t = datetime.datetime.now()

        self._verbose = verbose or False
        self._history_lock = Lock()
        self.history = []


    def _record_history(self, **kws):
        with self._history_lock:
            self.history.append({**kws, 'time': self.t})
        

    def __call__(self,x=None):
        if x is not None:
            self.x = np.array(x).copy()
        self.xrd = self.x + np.random.rand(len(self.control_CSETs))*(self.control_max-self.control_min)*1e-9
        self.y = self.fun(self.xrd)
        

    def get_df(self,PVs,x=None,tol=None,dt=None,time_span=None):
        dt = dt or 0.2
        time_span = time_span or self.fetch_data_time_span
        assert set(PVs) <= set(self.all_PVs)
        vals = []
        index = [self.t + datetime.timedelta(seconds=i*dt) for i in range(int(time_span/dt))]
        if x is None:
            x = self.x
            tol = None
        else:
            x = np.array(x)
            if tol is not None:
                tol = np.array(tol)
        x_target = np.array(x)
        current_x = self.x.copy()
        max_steps_per_dt = self.ramping_rate * dt
        ret = 'Timeout'
        for i in range(len(index)):
            delta = x_target - current_x
            step = np.clip(delta, -max_steps_per_dt, max_steps_per_dt)
            current_x += step
            self(current_x)
            vals.append(list(current_x) + list(self.xrd) + list(self.y))
            if tol is not None:
                if np.allclose(delta, 0, atol=tol):
                    index = index[:i+1]
                    ret = 'Put Finished'
                    break
        self(x_target)  # complete ramping assuming read after ramping will take at later time 
        df = pd.DataFrame(vals, index=index, columns=list(self.all_PVs))
        self.t += datetime.timedelta(seconds=time_span)
        return ret, df[PVs]

    def ensure_set(self,
                   setpoint_pv: List[str], 
                   readback_pv: List[str], 
                   goal: List[float], 
                   tol: List[float], 
                   timeout: float = 10.0, 
                   sample_interval: Optional[float] = 0.2,
                   extra_monitors: List[str] = None, 
                   verbose: Optional[bool] = None,
                   ):
        
        if self._verbose if verbose is None else verbose:
            print('Ramping in progress...')
            display(pd.DataFrame(np.array(goal).reshape(1, -1), columns=setpoint_pv))

        assert set(setpoint_pv) <= set(self.control_CSETs)
        assert set(readback_pv) <= set(self.control_RDs)
        if extra_monitors is None:
            extra_monitors = []
        assert set(extra_monitors).isdisjoint(set(setpoint_pv))
        assert set(extra_monitors).isdisjoint(set(readback_pv))
        newx = self.x.copy()
        for iset,setPV in enumerate(setpoint_pv):
            for icon,conPV in enumerate(self.control_CSETs):
                if setPV == conPV:
                    newx[icon] = goal[iset]
                    break
        extra_monitors = extra_monitors if extra_monitors is not None else []

        ret, df = self.get_df(setpoint_pv+readback_pv+extra_monitors,x=newx,tol=tol,dt=sample_interval,time_span=min(timeout,2))
        self._record_history(caller='ensure_set', setpoint_pv=setpoint_pv, readback_pv=readback_pv, goal=goal, tol=tol, ret=ret, data=df)
        return ret, df
        
    def fetch_data(self,
                   pvlist: List[str],
                   time_span: float = None, 
                   sample_interval : float = None,
                   verbose: Optional[bool] = None,
                   ):
        ret, df = self.get_df(pvlist,dt=sample_interval,time_span=time_span)
        if self._verbose if verbose is None else verbose:
            display(df)
        return df
        