import torch
import numpy as np
from collections import OrderedDict
from typing import List, Dict, Optional, Callable


try:
    from IPython.display import display as _display
except ImportError:
    _display = print
def display(obj):
    try:
        _display(obj)
    except:
        print(obj)

from .util import cyclic_distance, elu



class SingleTaskObjectiveFunction:
    def __init__(self,  
        objective_PVs : List[str],
        composite_objective_name: str,
        custom_function: Optional[Callable] = None,
        objective_goal: Dict = None, 
        objective_weight: Dict = None,
        objective_tolerance: Optional[Dict] = None,
        p_order: int = 2,
        apply_bilog: bool = False, 
    ):
        '''
        objective_goal: a Dict specifing goal of key=PVname, val=goal. 
                            (e.g.) obj = 1 - |(value-goal)/obj_tol|
                        if None and 'objective_fill_none_by_init' is True
                            the value at initialization will be set to goal 
                        if dict, must of the form 
                            {'less than': float or None}  or 
                            {'more than':float or None}
                        if the goal is {'more than':goal}
                            (e.g.) obj = -elu(-(value-goal)/obj_tol)
                        if the goal is {'less than':goal}
                            (e.g.) obj = -elu( (value-goal)/obj_tol)
                        where elu(x) = x (x>0)
                                     = e^x - 1 (x<=0)
        objective_weight: a Dict specifing weight of key=PVname, val=weight.  
                          if weight is 0 all corresponding objective will not be measured and calculated
        objective_tolerance: a Dict specifing normalization factor of key=PVname, val=obj_tol. 
                        This value effectively serves as an tolerace of corresponding objective
        objective_p_order: integer for objective power. default=2
                           (e.g) obj = sign(obj)*|obj|^p_order
                           large p_order is useful to strongly penalize values far from goal
        e.g.)
        objective_goal = { 
            'FE_MEBT:BPM_D1056:XPOS_RD' : 0.0,     #(mm)
            'FE_MEBT:BPM_D1056:YPOS_RD' : 0.0,     #(mm)
            'FE_MEBT:BPM_D1056:PHASE_RD': 80,      #(deg)
            'FE_MEBT:BPM_D1056:MAG_RD'  : {'more than': 0.01},
            'FE_MEBT:BPM_D1072:XPOS_RD' : 0.0,     #(mm)
            'FE_MEBT:BPM_D1072:YPOS_RD' : 0.0,     #(mm)
            'FE_MEBT:BPM_D1072:PHASE_RD': 90,      #(deg)
            'FE_MEBT:BPM_D1072:MAG_RD'  : {'more than': 0.01},
            'FE_MEBT:BPM_D1094:XPOS_RD' : 0.0,     #(mm)
            'FE_MEBT:BPM_D1094:YPOS_RD' : 0.0,     #(mm)
            'FE_MEBT:BPM_D1094:PHASE_RD': 90,      #(deg)
            'FE_MEBT:BPM_D1094:MAG_RD'  :{'more than': 0.01},
            'FE_MEBT:BCM_D1055:AVGPK_RD':{'more than': 30},
            'FE_MEBT:FC_D1102:PKAVG_RD': {'more than': 30},
                           },
        objective_weight = { 
            'FE_MEBT:BPM_D1056:XPOS_RD' : 1,     
            'FE_MEBT:BPM_D1056:YPOS_RD' : 1,     
            'FE_MEBT:BPM_D1056:PHASE_RD': 1, 
            'FE_MEBT:BPM_D1056:MAG_RD'  : 1, 
            'FE_MEBT:BPM_D1072:XPOS_RD' : 1,     
            'FE_MEBT:BPM_D1072:YPOS_RD' : 1,     
            'FE_MEBT:BPM_D1072:PHASE_RD': 1, 
            'FE_MEBT:BPM_D1072:MAG_RD'  : 1, 
            'FE_MEBT:BPM_D1094:XPOS_RD' : 1,     
            'FE_MEBT:BPM_D1094:YPOS_RD' : 1,     
            'FE_MEBT:BPM_D1094:PHASE_RD': 1,
            'FE_MEBT:BPM_D1094:MAG_RD'  : 1,
            'FE_MEBT:BCM_D1055:AVGPK_RD': 1,
            'FE_MEBT:FC_D1102:PKAVG_RD' : 1,
            },
        objective_tolerance = { 
            'FE_MEBT:BPM_D1056:XPOS_RD' : 1.,     
            'FE_MEBT:BPM_D1056:YPOS_RD' : 1.,     
            'FE_MEBT:BPM_D1056:PHASE_RD': 1., 
            'FE_MEBT:BPM_D1056:MAG_RD'  : 0.01, 
            'FE_MEBT:BPM_D1072:XPOS_RD' : 1.,     
            'FE_MEBT:BPM_D1072:YPOS_RD' : 1.,     
            'FE_MEBT:BPM_D1072:PHASE_RD': 1., 
            'FE_MEBT:BPM_D1072:MAG_RD'  : 0.01, 
            'FE_MEBT:BPM_D1094:XPOS_RD' : 1.,     
            'FE_MEBT:BPM_D1094:YPOS_RD' : 1.,     
            'FE_MEBT:BPM_D1094:PHASE_RD': 1.,
            'FE_MEBT:BPM_D1094:MAG_RD'  : 0.01,
            'FE_MEBT:BCM_D1055:AVGPK_RD': 3,
            'FE_MEBT:FC_D1102:PKAVG_RD' : 3,
            },
        '''
        self.objective_PVs = objective_PVs
        self.composite_objective_name = composite_objective_name
        self.custom_function = custom_function
        if custom_function is None:
            assert objective_goal is not None
            assert objective_weight is not None
            assert objective_tolerance is not None
            assert isinstance(p_order, int) and p_order > 0, "p_order must be a positive integer"
            assert isinstance(apply_bilog, bool), "apply_bilog must be a boolean"
            self.objective_weight = OrderedDict([(pv,objective_weight[pv]) for pv in objective_PVs])
            self.objective_goal   = OrderedDict([(pv,objective_goal[pv]) for pv in objective_PVs])
            self.objective_tolerance = OrderedDict([(pv,objective_tolerance[pv]) for pv in objective_PVs])
            self.p_order = p_order
            self.apply_bilog = apply_bilog


    def __call__(self, 
                 y: torch.Tensor,
                 return_all_objs: Optional[bool] = False):
        """
        Calculate objectives
        
        Args:
            y: numpy.ndarray or torch.Tensor of shape (n_rows, len(self.objective_PVs)) containing values
                for PVs in self.objective_PVs.
        
        Returns:
            tuple: (composite objective, all objectives)
        """
        if isinstance(y, np.ndarray):
            y = torch.from_numpy(y)
        if y.ndim == 1:
            y = y.unsqueeze(0)  # Add batch dimension for 1D input           
        assert y.shape[-1] == len(self.objective_PVs), "Array column count must match self.objective_PVs length"
        if self.custom_function is not None: 
            composite_obj = self.custom_function(y)
        else:
            composite_obj = torch.zeros(y.shape[:-1])
            objs = torch.zeros_like(y)
            flattened = y.view(-1, y.shape[-1])

            for idx, pv in enumerate(self.objective_PVs):
                goal = self.objective_goal[pv]
                values = flattened[:, idx]
                
                if isinstance(goal, float):
                    if 'BPM' in pv and 'PHASE' in pv:
                        obj = 1 - (torch.abs(cyclic_distance(values, goal, 180) / self.objective_tolerance[pv])) ** self.p_order
                    else:
                        obj = 1 - (torch.abs((values - goal) / self.objective_tolerance[pv])) ** self.p_order
                elif 'more than' in goal:
                    obj = -elu(-(values - goal['more than']) / self.objective_tolerance[pv]) ** self.p_order
                elif 'less than' in goal:
                    obj = -elu((values - goal['less than']) / self.objective_tolerance[pv]) ** self.p_order
                else:
                    raise RuntimeError("Goal must be float or dict with 'more than' or 'less than'")
                
                obj = obj.view(composite_obj.shape)
                objs[...,idx] = obj
                composite_obj += self.objective_weight[pv] * obj
        
            if self.apply_bilog:
                composite_obj = torch.sign(composite_obj) * torch.log(1 + torch.abs(composite_obj))
        
        if return_all_objs:
            return composite_obj, objs
        else:
            return composite_obj
    
    
    def calculate_objectives_from_df(self, df):
        if df.empty:
            raise ValueError("Input DataFrame is empty")
        # assert set(self.objective_PVs).issubset(df.columns)
        composite_obj = self(df[self.objective_PVs].values)
        df[self.composite_objective_name] = composite_obj
        return df