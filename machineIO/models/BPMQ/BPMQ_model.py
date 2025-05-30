import re
import os
import pickle
import time
import datetime
from typing import List, Optional, Dict
from copy import deepcopy as copy

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
from torch import nn
from botorch.models import SingleTaskGP

from TIS161_coeffs import TIS161_coeffs
try:
    from IPython.display import display as _display
except ImportError:
    _display = print

def display(obj):
    try:
        _display(obj)
    except:
        print(obj)

script_dir = os.path.dirname(os.path.realpath(__file__))


class dummy_scheduler:
    def __init__(self,lr):
        self.lr = lr
    def step(self,*args,**kwargs):
        pass
    def get_last_lr(self,*args,**kwargs):
        return [self.lr]


class BPMQ_model(nn.Module):
    def __init__(self, n_node: int = 16, n_hidden_layer: int = 4, dtype=torch.float32):
        super(BPMQ_model, self).__init__()
        self.pickup_calibration = nn.Parameter(torch.zeros(3, dtype=dtype))
        self.geometry_calibration = nn.Parameter(torch.zeros(1, dtype=dtype))
        
        layers = [nn.Linear(14, n_node, dtype=dtype), nn.ELU()]
        for _ in range(n_hidden_layer):
            layers.extend([nn.Linear(n_node, n_node, dtype=dtype), nn.ELU()])
        layers.append(nn.Linear(n_node, 1))
        
        self.nn = nn.Sequential(*layers)
        self.dtype = dtype

    def polynomial_features(self, u: torch.Tensor) -> torch.Tensor:
        poly_features = torch.cat([
            u, 
            u ** 2,
            (u[:, :3] - u[:, 1:]) ** 2, 
            (u[:, :2] - u[:, 2:]) ** 2,
            (u[:, :1] - u[:, 3:]) ** 2,
        ], dim=1)
        return poly_features

    def forward(self, bpm_U: torch.Tensor, bpm_x: torch.Tensor, bpm_y: torch.Tensor) -> torch.Tensor:
        bpm_U = bpm_U.to(dtype=self.dtype)
        c = torch.zeros(4, dtype=self.dtype, device=bpm_U.device)
        c[:3] = self.pickup_calibration
        c[3] = -self.pickup_calibration.sum()

        U = bpm_U / bpm_U.sum(dim=1, keepdim=True)
        u = (1.0 + 0.01 * c.view(1, 4)) * U
    
        Qtheory = (1.0 + 0.1 * self.geometry_calibration) * 241 * ((u[:, 1] + u[:, 2]) - (u[:, 0] + u[:, 3])) \
                  - (bpm_x ** 2 - bpm_y ** 2)
        
        poly_u = self.polynomial_features(4 * u)
        Residual = self.nn(poly_u).view(-1)

        return u, Qtheory + Residual
    

class TISRAW2BPMQ_model(nn.Module):
    def __init__(self, F=2, dtype=torch.float32, **kwarg):
        super(TISRAW2BPMQ_model, self).__init__()
        self.dtype = dtype
        self.F = F
        layers = [nn.Conv1d(1,   4*F, 9, stride=5, dtype=dtype), torch.nn.ELU(),
                  nn.Conv1d(4*F, 2*F, 8, stride=5, dtype=dtype), torch.nn.ELU(),
                 ]
        self.conv_res = torch.nn.Sequential(*layers)

        layers = [nn.Conv1d(1,   1*F, 7, stride=4, dtype=dtype), torch.nn.ELU(),
                  nn.Conv1d(1*F, 2*F, 5, stride=2, dtype=dtype), torch.nn.ELU(),
                  nn.Conv1d(2*F, 2*F, 4, stride=1, dtype=dtype), torch.nn.ELU(),
                  nn.Conv1d(2*F, 2*F, 3, stride=1, dtype=dtype), torch.nn.ELU(),
                 ]
        self.conv = torch.nn.Sequential(*layers)
        
        layers = [torch.nn.Linear(4, 8*F, dtype=dtype), torch.nn.ELU(),
                  torch.nn.Linear(8*F, 8*F, dtype=dtype), torch.nn.ELU(),
                  torch.nn.Linear(8*F, 2*F, dtype=dtype)]
        self.nn_xy = torch.nn.Sequential(*layers)
        
        layers = [torch.nn.Linear(10*F, 8*F, dtype=dtype), torch.nn.ELU(),
                  torch.nn.Linear(8*F,  1, dtype=dtype)]
        self.nn_res = torch.nn.Sequential(*layers)

        layers = [torch.nn.Linear(10*F, 4*F, dtype=dtype), torch.nn.ELU(),
                  torch.nn.Linear(4*F, 4*F, dtype=dtype), torch.nn.ELU(),
                  torch.nn.Linear(4*F, 2*F, dtype=dtype), torch.nn.ELU(),
                  torch.nn.Linear(2*F,  1, dtype=dtype)]
        self.nn = torch.nn.Sequential(*layers)
        
    def forward(self, TISRAW1, TISRAW2, TISRAW3, TISRAW4, xypos):
        TISRAW1 = TISRAW1.view(-1,1,68).to(dtype=self.conv_res[0].weight.dtype)
        TISRAW2 = TISRAW2.view(-1,1,68).to(dtype=self.conv_res[0].weight.dtype)
        TISRAW3 = TISRAW3.view(-1,1,68).to(dtype=self.conv_res[0].weight.dtype)
        TISRAW4 = TISRAW4.view(-1,1,68).to(dtype=self.conv_res[0].weight.dtype)
        
        TISRAW1 = self.conv(TISRAW1) + self.conv_res(TISRAW1)
        TISRAW2 = self.conv(TISRAW2) + self.conv_res(TISRAW2)
        TISRAW3 = self.conv(TISRAW3) + self.conv_res(TISRAW3)
        TISRAW4 = self.conv(TISRAW4) + self.conv_res(TISRAW4)  # shape of (batch_size,2*self.F,1)
        
        xypos = (xypos.view(-1,2).to(dtype=self.conv_res[0].weight.dtype)+4.0)/8.0
        xypos = torch.cat((xypos, xypos**2), dim=1)  # feature engineer
        xypos =self.nn_xy(xypos)  
#         print("nn_xy(xypos).shape",xypos.shape)
#         print("TISRAW1.shape",TISRAW1.shape)
        feature = torch.cat((TISRAW1[:,:,0], 
                             TISRAW2[:,:,0], 
                             TISRAW3[:,:,0], 
                             TISRAW4[:,:,0], 
                             xypos), dim=1)
        bpmQpred = 20*self.nn(feature).view(-1) + 20*self.nn_res(feature).view(-1)
        return feature, bpmQpred


def BPMQ_loss(model_Q, beam_Q, Qerr=None, weight=None, *args, **kwargs):#, beam_Qerr):
    if Qerr is None:
        if weight is None:
            return torch.mean(torch.abs(model_Q - beam_Q))
        else:
            return torch.mean(torch.abs(model_Q - beam_Q)*weight)
    else: 
        if weight is None:
            return torch.mean(torch.abs(model_Q - beam_Q)/(1.0 + Qerr))
        else: 
            return torch.mean(torch.abs(model_Q - beam_Q)/(1.0 + Qerr)*weight)
        
       
def train_BPMQ_model(
    model,
    epochs,lr,
    train_U,train_X,train_Y,train_Q,
    train_Qerr=None, train_weight=None,
    val_U=None,val_X=None,val_Y=None,val_Q=None,
    val_Qerr=None, val_weight=None,
    batch_size=None,
    shuffle=True,
    validation_split=0.0,
    criterion = BPMQ_loss,
    optimizer = torch.optim.Adam,
    optim_args = None,
    optimizer_state_dict = None,
    lr_scheduler = True,
    prev_history = None,
    load_best = True,
    training_timeout = np.inf,
    verbose = False,
    fname_model = 'model.pt',
    fname_opt = 'opt.pt',
    fname_history = 'history.pkl',
    ):
    
    if isinstance(optimizer,str):
        optimizer = getattr(torch.optim, optimizer)
    if isinstance(criterion,str):
        criterion =  getattr(torch.nn, criterion)()

    if verbose:
        print("Train Function Arguments:",datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
        print(f"  - model: {model.__class__.__name__}")
        print(f"  - epochs: {epochs}")
        print(f"  - lr: {lr}")
        print(f"  - batch_size: {batch_size}")
        print(f"  - shuffle: {shuffle}")
        print(f"  - validation_split: {validation_split}")
        print(f"  - criterion: {criterion.__class__.__name__}")
        print(f"  - optimizer: {optimizer.__name__}")
        print(f"  - optim_args: {optim_args}")
        print(f"  - optimizer_state_dict: {optimizer_state_dict}")
        print(f"  - lr_scheduler: {lr_scheduler}")
        print(f"  - prev_history: {prev_history}")
        print(f"  - load_best: {load_best}")
        print(f"  - training_timeout: {training_timeout}")
        print(f"  - verbose: {verbose}")
        print(f"  - fname_model: {fname_model}")
        print(f"  - fname_opt: {fname_opt}")
        print(f"  - fname_history: {fname_history}")
        print()
    
    # get dtype and device of the input layer of the model
    n,d = train_U.shape
    if verbose:
        print("Model Paramers:")
    for name, p in model.named_parameters():
        if verbose:
            print(f"  - name: {name}, shape: {p.shape}, dtype: {p.dtype}, device: {p.device}")
        if len(p.shape) == 1:
            continue
        if 'nn.0.' in name:
            device = p.device
            dtype = p.dtype
    if verbose:
        print()

    train_U=torch.tensor(train_U,dtype=dtype)
    train_X=torch.tensor(train_X,dtype=dtype)
    train_Y=torch.tensor(train_Y,dtype=dtype)
    train_Q=torch.tensor(train_Q,dtype=dtype)
    if train_Qerr is not None:
        train_Qerr=torch.tensor(train_Qerr,dtype=dtype)
    if train_weight is not None:
        train_weight=torch.tensor(train_weight,dtype=dtype)
    ntrain = len(train_U)
    assert ntrain == len(train_X) == len(train_Y) == len(train_Q)

    nval = 0
    if validation_split>0.0 and val_U is None:
        p = np.random.permutation(np.arange(ntrain))
        train_U = train_U[p]
        train_X = train_X[p]
        train_Y = train_Y[p]
        train_Q = train_Q[p]
        if train_Qerr is not None:
            train_Qerr = train_Qerr[p]
        if train_weight is not None:
            train_weight = train_weight[p]
    
        nval = int(validation_split*ntrain)
        ntrain = ntrain-nval
        
        val_U = train_U[:nval]
        val_X = train_X[:nval]
        val_Y = train_Y[:nval]
        val_Q = train_Q[:nval]
        if train_Qerr is not None:
            val_Qerr = train_Qerr[:nval]
        if train_weight is not None:
            val_weight = train_weight[:nval]
        train_U = train_U[nval:]
        train_X = train_X[nval:]
        train_Y = train_Y[nval:]
        train_Q = train_Q[nval:]
        if train_Qerr is not None:
            train_Qerr = train_Qerr[nval:]
        if train_weight is not None:    
            train_weight = train_weight[nval:]

    elif val_U is not None:
        val_U=torch.tensor(val_U,dtype=dtype)
        val_X=torch.tensor(val_X,dtype=dtype)
        val_Y=torch.tensor(val_Y,dtype=dtype)
        val_Q=torch.tensor(val_Q,dtype=dtype)
        if val_Qerr is not None:
            val_Qerr=torch.tensor(val_Qerr,dtype=dtype)
        if val_weight is not None:
            val_weight=torch.tensor(val_weight,dtype=dtype)
        nval = len(val_U)
        assert nval == len(val_X) == len(val_Y) == len(val_Q)
        
    batch_size = batch_size or ntrain
    nbatch_val = int(nval/batch_size)
    if nbatch_val==0 and nval > 0:
        val_batch_size = nval
        nbatch_val = 1
    else:
        val_batch_size = batch_size

    train_batch_size = min(batch_size,ntrain)
    nbatch_train = int(ntrain/train_batch_size)
    
    training_timeout = training_timeout
    t0 = time.monotonic()
    assert epochs>0
    optim_args = optim_args or {}
    

    opt = optimizer(model.parameters(filter(lambda p: p.requires_grad, model.parameters())),lr=lr,**optim_args)
    if optimizer_state_dict is not None:
        opt.load_state_dict(optimizer_state_dict)
        
               
    if prev_history is None:
        history = {
            'train_loss':[],
            'val_loss'  :[],
            'lr'        :[],
            }
    else:
        assert "train_loss" in prev_history
        history = prev_history 
        if "lr" not in history:
            history['lr'] = [None]*len(history["train_loss"])
    epoch_start = len(history['train_loss'])
        
    if lr_scheduler:
        # last_epoch = epoch_start*train_batch_size
        # if last_epoch == 0:
        #     last_epoch = -1
        scheduler = torch.optim.lr_scheduler.OneCycleLR(opt, 
            max_lr=lr,
            div_factor=int(np.clip(epochs/500,a_min=1,a_max=20)),
            pct_start=0.05, 
            final_div_factor=int(np.clip(epochs/50,a_min=10,a_max=1e2)),
            # epochs=epochs, steps_per_epoch=nbatch_train, last_epoch=last_epoch)
            epochs=epochs, steps_per_epoch=1, last_epoch=-1)
    else:       
        scheduler = dummy_scheduler(lr)
        
    best = np.inf
    model.train()
    epoch = epoch_start-1
    save_epoch = epoch
    
   
        
    if verbose:
        print("Training begin at: ", datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
        print()
    
    while(True):
        epoch += 1
        if epoch>=epoch_start + epochs:
            break
        lr_ = scheduler.get_last_lr()[0]
        history['lr'].append(lr_)
        
        if shuffle:
            p = np.random.permutation(len(train_U))
            train_U = train_U[p]
            train_X = train_X[p]
            train_Y = train_Y[p]
            train_Q = train_Q[p]
            if train_Qerr is not None:
                train_Qerr = train_Qerr[p]
            if train_weight is not None:
                train_weight = train_weight[p]
        train_loss = 0
        
        model.train()
        for i in range(nbatch_train):
            i1 = i*train_batch_size
            i2 = i1+train_batch_size
            U = train_U[i1:i2,:].to(device)
            X = train_X[i1:i2].to(device)
            Y = train_Y[i1:i2].to(device)
            Q = train_Q[i1:i2].to(device)
            if train_Qerr is not None:
                Qerr = train_Qerr[i1:i2].to(device)
            else:
                Qerr = None
            if train_weight is not None:
                weight = train_weight[i1:i2].to(device)
            else:
                weight = None
            opt.zero_grad()
            _u, Q_pred = model(U,X,Y)
            loss = criterion(Q_pred, Q, Qerr=Qerr, weight=weight)
            loss.backward()
            opt.step()
            train_loss = train_loss + loss.item()
        train_loss /= nbatch_train

        if i2 < ntrain-1 and ntrain < 100:
            U = train_U[i2:,:].to(device)
            X = train_X[i2:].to(device)
            Y = train_Y[i2:].to(device)
            Q = train_Q[i2:].to(device)
            if train_Qerr is not None:
                Qerr = train_Qerr[i2:].to(device)
            else:
                Qerr = None
            if train_weight is not None:
                weight = train_weight[i2:].to(device)
            else:
                weight = None
            opt.zero_grad()
            _u, Q_pred = model(U,X,Y)
            loss = criterion(Q_pred, Q, Qerr=Qerr, weight=weight)
            loss.backward()
            opt.step()
            train_loss = (train_loss*train_batch_size*nbatch_train + loss.item()*(ntrain-i2))/ntrain
        scheduler.step()
        history['train_loss'].append(train_loss)

        val_loss = 0.0
        if nbatch_val>0:
            model.eval()
            with torch.no_grad():
                val_loss = 0.
                for i in range(nbatch_val):
                    i1 = i*val_batch_size
                    i2 = i1+val_batch_size
                    U = val_U[i1:i2,:].to(device)
                    X = val_X[i1:i2].to(device)
                    Y = val_Y[i1:i2].to(device)
                    Q = val_Q[i1:i2].to(device)
                    if val_Qerr is not None:
                        Qerr = val_Qerr[i1:i2].to(device)
                    else:
                        Qerr = None
                    if val_weight is not None:
                        weight = val_weight[i1:i2].to(device)
                    else:
                        weight = None
                    _u, Q_pred = model(U,X,Y)
                    loss = criterion(Q, Q_pred, Qerr=Qerr, weight=weight)
                    val_loss += loss.item()
                val_loss /= nbatch_val

                if i2 < nval-1:
                    U = val_U[i2:,:].to(device)
                    X = val_X[i2:].to(device)
                    Y = val_Y[i2:].to(device)
                    Q = val_Q[i2:].to(device)
                    if val_Qerr is not None:
                        Qerr = val_Qerr[i2:].to(device)
                    else:
                        Qerr = None
                    if val_weight is not None:
                        weight = val_weight[i2:].to(device)
                    else:
                        weight = None
                    _u, Q_pred = model(U,X,Y)
                    loss = criterion(Q, Q_pred, Qerr=Qerr, weight=weight)
                    val_loss = (val_loss*val_batch_size*nbatch_val + loss.item()*(nval-i2))/nval
            history['val_loss'].append(val_loss)

            if val_loss < best:
                best = val_loss
                model_state_dict = copy(model.state_dict())
                opt_state_dict = copy(opt.state_dict())
                if epoch > save_epoch + 5:
                    save_epoch = epoch
                    torch.save(model_state_dict,fname_model)
                    torch.save(opt_state_dict, fname_opt)
                    pickle.dump(history,open(fname_history,'wb'))
        else:
            if train_loss < best:
                best = train_loss
                model_state_dict = copy(model.state_dict())
                opt_state_dict = copy(opt.state_dict())
                if epoch > save_epoch + 5:
                    save_epoch = epoch
                    torch.save(model_state_dict,fname_model)
                    torch.save(opt_state_dict, fname_opt)
                    pickle.dump(history,open(fname_history,'wb'))
        
        if verbose:
            nskip = int(epochs/100)
            if epoch%nskip==0:
                elapsed_t = datetime.timedelta(seconds=time.monotonic() - t0)
                if nbatch_val>0:
                    print(f' Epoch {epoch+0:04}: | Train Loss: {train_loss:.2E} | Val Loss: {val_loss:.2E} | lr: {lr_:.2E} | {elapsed_t}')
                else:
                    print(f' Epoch {epoch+0:04}: | Train Loss: {train_loss:.2E} | lr: {lr_:.2E} | {elapsed_t}')

    dt = time.monotonic()-t0                
    if load_best:
        model.load_state_dict(model_state_dict)
            
    return history,model_state_dict,opt_state_dict
       

def train_TISRAW2BPMQ_model(
    model,
    epochs,lr,
    train_TIS1,train_TIS2,train_TIS3,train_TIS4,train_XY,train_Qsim,
    train_Qerr=None,train_weight=None,
    val_TIS1=None,val_TIS2=None,val_TIS3=None,val_TIS4=None,val_XY=None,val_Qsim=None,
    val_Qerr=None,val_weight=None,
    batch_size=None,
    shuffle=True,
    validation_split=0.0,
    criterion = BPMQ_loss,
    optimizer = torch.optim.Adam,
    optim_args = None,
    optimizer_state_dict = None,
    lr_scheduler = True,
    prev_history = None,
    load_best = True,
    training_timeout = np.inf,
    verbose = False,
    fname_model = 'model_TISRAW2BPMQ.pt',
    fname_opt = 'opt.pt',
    fname_history = 'history.pkl',
    ):
    
    if isinstance(optimizer,str):
        optimizer = getattr(torch.optim, optimizer)
    if isinstance(criterion,str):
        criterion =  getattr(torch.nn, criterion)()

    if verbose:
        print("Train Function Arguments:",datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
        print(f"  - model: {model.__class__.__name__}")
        print(f"  - epochs: {epochs}")
        print(f"  - lr: {lr}")
        print(f"  - batch_size: {batch_size}")
        print(f"  - shuffle: {shuffle}")
        print(f"  - validation_split: {validation_split}")
        print(f"  - criterion: {criterion.__class__.__name__}")
        print(f"  - optimizer: {optimizer.__name__}")
        print(f"  - optim_args: {optim_args}")
        print(f"  - optimizer_state_dict: {optimizer_state_dict}")
        print(f"  - lr_scheduler: {lr_scheduler}")
        print(f"  - prev_history: {prev_history}")
        print(f"  - load_best: {load_best}")
        print(f"  - training_timeout: {training_timeout}")
        print(f"  - verbose: {verbose}")
        print(f"  - fname_model: {fname_model}")
        print(f"  - fname_opt: {fname_opt}")
        print(f"  - fname_history: {fname_history}")
        print()
    
    # get dtype and device of the input layer of the model
    if verbose:
        print("Model Paramers:")
    for name, p in model.named_parameters():
        if verbose:
            print(f"  - name: {name}, shape: {p.shape}, dtype: {p.dtype}, device: {p.device}")
        if len(p.shape) == 1:
            continue
        if 'nn.0.' in name:
            device = p.device
            dtype = p.dtype
    if verbose:
        print()

    train_TIS1=torch.tensor(train_TIS1,dtype=dtype)
    train_TIS2=torch.tensor(train_TIS2,dtype=dtype)
    train_TIS3=torch.tensor(train_TIS3,dtype=dtype)
    train_TIS4=torch.tensor(train_TIS4,dtype=dtype)
    train_XY  =torch.tensor(train_XY,dtype=dtype)
    train_Qsim=torch.tensor(train_Qsim,dtype=dtype)
    if train_Qerr is not None:
        train_Qerr=torch.tensor(train_Qerr,dtype=dtype)
    if train_weight is not None:
        train_weight=torch.tensor(train_weight,dtype=dtype)
    ntrain = len(train_TIS1)
    assert ntrain == len(train_TIS2) == len(train_TIS3) == len(train_TIS4) == len(train_XY) == len(train_Qsim) == len(train_Qerr)

    if val_TIS1 is not None:
        val_TIS1=torch.tensor(val_TIS1,dtype=dtype)
        val_TIS2=torch.tensor(val_TIS2,dtype=dtype)
        val_TIS3=torch.tensor(val_TIS3,dtype=dtype)
        val_TIS4=torch.tensor(val_TIS4,dtype=dtype)
        val_XY=torch.tensor(val_XY,dtype=dtype)
        val_Qsim=torch.tensor(val_Qsim,dtype=dtype)
        if val_Qerr is not None:
            val_Qerr=torch.tensor(val_Qerr,dtype=dtype)
        if val_weight is not None:
            val_weight=torch.tensor(val_weight,dtype=dtype)
        nval = len(val_TIS1)
        assert nval == len(val_TIS2) == len(val_TIS3) == len(val_TIS4) == len(val_XY) == len(val_Qsim)
        if val_Qerr is not None:
            assert nval == len(val_Qerr)
        

    elif validation_split>0.0 :
        p = np.random.permutation(np.arange(ntrain))
        train_TIS1 = train_TIS1[p]
        train_TIS2 = train_TIS2[p]
        train_TIS3 = train_TIS3[p]
        train_TIS4 = train_TIS4[p]
        train_XY   = train_XY[p]
        train_Qsim = train_Qsim[p]
        if train_Qerr is not None:
            train_Qerr = train_Qerr[p]
        if train_weight is not None:
            train_weight = train_weight[p]

        nval = int(validation_split*ntrain)
        ntrain = ntrain-nval
        
        val_TIS1 = train_TIS1[:nval]
        val_TIS2 = train_TIS2[:nval]
        val_TIS3 = train_TIS3[:nval]
        val_TIS4 = train_TIS4[:nval]
        val_XY = train_XY[:nval]
        val_Qsim = train_Qsim[:nval]
        if train_Qerr is not None:
            val_Qerr = train_Qerr[:nval]
        if train_weight is not None:
            val_weight = train_weight[:nval]
        train_TIS1 = train_TIS1[nval:]
        train_TIS2 = train_TIS2[nval:]
        train_TIS3 = train_TIS3[nval:]
        train_TIS4 = train_TIS4[nval:]
        train_XY   = train_XY[nval:]
        train_Qsim = train_Qsim[nval:]
        if train_Qerr is not None:
            train_Qerr = train_Qerr[nval:]
        if train_weight is not None:
            train_weight = train_weight[nval:]

    else: 
        nval = 0

    batch_size = batch_size or ntrain
    nbatch_val = int(nval/batch_size)
    if nbatch_val==0 and nval > 0:
        val_batch_size = nval
        nbatch_val = 1
    else:
        val_batch_size = batch_size

    train_batch_size = min(batch_size,ntrain)
    nbatch_train = int(ntrain/train_batch_size)
    
    training_timeout = training_timeout
    t0 = time.monotonic()
    assert epochs>0
    optim_args = optim_args or {}
    
    opt = optimizer(model.parameters(filter(lambda p: p.requires_grad, model.parameters())),lr=lr,**optim_args)
    if optimizer_state_dict is not None:
        opt.load_state_dict(optimizer_state_dict)
        
               
    if prev_history is None:
        history = {
            'train_loss':[],
            'val_loss'  :[],
            'lr'        :[],
            }
    else:
        assert "train_loss" in prev_history
        history = prev_history 
        if "lr" not in history:
            history['lr'] = [None]*len(history["train_loss"])
    epoch_start = len(history['train_loss'])
        

    if lr_scheduler:
        # last_epoch = epoch_start*train_batch_size
        # if last_epoch == 0:
        #     last_epoch = -1
        scheduler = torch.optim.lr_scheduler.OneCycleLR(opt, 
            max_lr=lr,
            div_factor=int(np.clip(epochs/500,a_min=1,a_max=20)),
            pct_start=0.05, 
            final_div_factor=int(np.clip(epochs/50,a_min=10,a_max=1e2)),
            # epochs=epochs, steps_per_epoch=nbatch_train, last_epoch=last_epoch)
            epochs=epochs, steps_per_epoch=1, last_epoch=-1)
    else:       
        scheduler = dummy_scheduler(lr)
        
    best = np.inf
    model.train()
    epoch = epoch_start-1
    save_epoch = epoch

        
    if verbose:
        print("Training begin at: ", datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
        print()
    
    while(True):
        epoch += 1
        if epoch>=epoch_start + epochs:
            break
        lr_ = scheduler.get_last_lr()[0]
        history['lr'].append(lr_)
        
        if shuffle:
            p = np.random.permutation(len(train_TIS1))
            train_TIS1 = train_TIS1[p]
            train_TIS2 = train_TIS2[p]
            train_TIS3 = train_TIS3[p]
            train_TIS4 = train_TIS4[p]
            train_XY   = train_XY[p]
            train_Qsim = train_Qsim[p]
            if train_Qerr is not None:
                train_Qerr = train_Qerr[p]
            if train_weight is not None:
                train_weight = train_weight[p]
        train_loss = 0
        
        model.train()
        for i in range(nbatch_train):
            i1 = i*train_batch_size
            i2 = i1+train_batch_size
            TIS1 = train_TIS1[i1:i2,:].to(device)
            TIS2 = train_TIS2[i1:i2,:].to(device)
            TIS3 = train_TIS3[i1:i2,:].to(device)
            TIS4 = train_TIS4[i1:i2,:].to(device)
            XY   = train_XY[i1:i2,:].to(device)
            Qsim = train_Qsim[i1:i2].to(device)
            if train_Qerr is not None:
                Qerr = train_Qerr[i1:i2].to(device)
            else:
                Qerr = None
            if train_weight is not None:
                weight = train_weight[i1:i2].to(device)
            else:
                weight = None
            opt.zero_grad()
            feature, Q_pred = model(TIS1, TIS2, TIS3, TIS4, XY)
            loss = criterion(Q_pred, Qsim, Qerr=Qerr, weight=weight)
            loss.backward()
            opt.step()
            train_loss = train_loss + loss.item()
        train_loss /= nbatch_train

        if i2 < ntrain-1 and ntrain < 100:
            TIS1 = train_TIS1[i2:,:].to(device)
            TIS2 = train_TIS2[i2:,:].to(device)
            TIS3 = train_TIS3[i2:,:].to(device)
            TIS4 = train_TIS4[i2:,:].to(device)
            XY   = train_XY[i2:,:].to(device)
            Qsim = train_Qsim[i2:].to(device)
            if train_Qerr is not None:
                Qerr = train_Qerr[i2:].to(device)
            else:
                Qerr = None
            if train_weight is not None:
                weight = train_weight[i2:].to(device)
            else:
                weight = None
            opt.zero_grad()
            feature, Q_pred = model(TIS1, TIS2, TIS3, TIS4, XY)
            loss = criterion(Q_pred, Qsim, Qerr=Qerr, weight=weight)
            loss.backward()
            opt.step()
            train_loss = (train_loss*train_batch_size*nbatch_train + loss.item()*(ntrain-i2))/ntrain
        scheduler.step()
        history['train_loss'].append(train_loss)
        
        val_loss = 0.0
        if nbatch_val>0:
            model.eval()
            with torch.no_grad():
                val_loss = 0.
                for i in range(nbatch_val):
                    i1 = i*val_batch_size
                    i2 = i1+val_batch_size
                    TIS1 = val_TIS1[i1:i2,:].to(device)
                    TIS2 = val_TIS2[i1:i2,:].to(device)
                    TIS3 = val_TIS3[i1:i2,:].to(device)
                    TIS4 = val_TIS4[i1:i2,:].to(device)
                    XY   = val_XY[i1:i2,:].to(device)
                    Qsim = val_Qsim[i1:i2].to(device)
                    if val_Qerr is not None:
                        Qerr = val_Qerr[i1:i2].to(device)
                    else:
                        Qerr = None
                    if val_weight is not None:
                        weight = val_weight[i1:i2].to(device)
                    else:
                        weight = None
                    feature, Q_pred = model(TIS1, TIS2, TIS3, TIS4, XY)
                    loss = criterion(Q_pred, Qsim, Qerr=Qerr, weight=weight)
                    val_loss += loss.item()
                val_loss /= nbatch_val

                if i2 < nval-1:
                    TIS1 = val_TIS1[i2:,:].to(device)
                    TIS2 = val_TIS2[i2:,:].to(device)
                    TIS3 = val_TIS3[i2:,:].to(device)
                    TIS4 = val_TIS4[i2:,:].to(device)
                    XY   = val_XY[i2:,:].to(device)
                    Qsim = val_Qsim[i2:].to(device)
                    if val_Qerr is not None:
                        Qerr = val_Qerr[i2:].to(device)
                    else:
                        Qerr = None
                    if val_weight is not None:
                        weight = val_weight[i2:].to(device)
                    else:
                        weight = None
                    feature, Q_pred = model(TIS1, TIS2, TIS3, TIS4, XY)
                    loss = criterion(Q_pred, Qsim, Qerr=Qerr, weight=weight)
                    val_loss = (val_loss*val_batch_size*nbatch_val + loss.item()*(nval-i2))/nval
            history['val_loss'].append(val_loss)

            if val_loss < best:
                best = val_loss
                checkpoint = {'state_dict': copy(model.state_dict()),
                              'F': model.F,
                              'dtype': model.dtype}
                # model_state_dict = copy(model.state_dict())
                opt_state_dict = copy(opt.state_dict())
                if epoch > save_epoch + 5:
                    save_epoch = epoch
                    torch.save(checkpoint,fname_model)
                    torch.save(opt_state_dict, fname_opt)
                    pickle.dump(history,open(fname_history,'wb'))
        else:
            if train_loss < best:
                best = train_loss
                checkpoint = {'state_dict': copy(model.state_dict()),
                              'F': model.F,
                              'dtype': model.dtype}
                # model_state_dict = copy(model.state_dict())
                opt_state_dict = copy(opt.state_dict())
                if epoch > save_epoch + 5:
                    save_epoch = epoch
                    torch.save(checkpoint,fname_model)
                    torch.save(opt_state_dict, fname_opt)
                    pickle.dump(history,open(fname_history,'wb'))
        
        if verbose:
            nskip = int(epochs/100)
            if epoch%nskip==0:
                elapsed_t = datetime.timedelta(seconds=time.monotonic() - t0)
                if nbatch_val>0:
                    print(f' Epoch {epoch+0:04}: | Train Loss: {train_loss:.2E} | Val Loss: {val_loss:.2E} | lr: {lr_:.2E} | {elapsed_t}')
                else:
                    print(f' Epoch {epoch+0:04}: | Train Loss: {train_loss:.2E} | lr: {lr_:.2E} | {elapsed_t}')

    dt = time.monotonic()-t0                
    if load_best:
        model.load_state_dict(checkpoint['state_dict'])
            
    return history,checkpoint,opt_state_dict



D_NUM_PATTERN = re.compile(r'\D(\d{4})$')  # Precompiled
def sort_by_Dnum(strings: list[str]) -> list[str]:
    """Sort BPM names by trailing 4-digit number."""
    return sorted(strings, key=lambda s: int(m.group(1)) if (m := D_NUM_PATTERN.search(s)) else 0)

# def sort_by_Dnum(strings):
#     """
#     Sort a list of PVs by dnum.
#     """
#     # Define a regular expression pattern to extract the 4-digit number at the end of each string
#     pattern = re.compile(r'\D(\d{4})$')

#     # Define a custom sorting key function that extracts the 4-digit number using the regex pattern
#     def sorting_key(s):
#         match = pattern.search(s)
#         if match:
#             return int(match.group(1))
#         return 0  # Default value if no match is found

#     # Sort the strings based on the custom sorting key
#     sorted_strings = sorted(strings, key=sorting_key)
#     return sorted_strings


class raw2Q_processor:
    def __init__(self,
        BPM_names  : List[str],
        model_type : Optional[str] = 'TIS161',
        ):
        self.BPM_names = sort_by_Dnum(BPM_names)    
        self.model_type = model_type 
        BPM_TIS161_PVs = []  
        BPM_TISRAW_PVs = []
        calibrated_PVs = []
        BPM_TIS161_coeffs = np.zeros(4*len(self.BPM_names))
        PVs2read = []
        self.BPMQ_models = {}
        self.BPMQ_GPs    = {}

        for i,name in enumerate(self.BPM_names):
            if name not in TIS161_coeffs:
                raise ValueError(f"{name} not found in TIS161_coeffs")
                
            if 'TISRAW' in self.model_type:
                TISRAW_PVs = [f"{name}:TISRAW{i + 1}_RD" for i in range(4)]
                BPM_TISRAW_PVs += TISRAW_PVs
            else:
                TISRAW_PVs = []
                
            TIS161_PVs = [f"{name}:TISMAG161_{i + 1}_RD" for i in range(4)]
            BPM_TIS161_PVs += TIS161_PVs
            
            calibrated_PVs += [f"{name}:U{i + 1}" for i in range(4)]
            BPM_TIS161_coeffs[4*i:4*(i+1)] = TIS161_coeffs[name]
            PVs2read += TISRAW_PVs + TIS161_PVs + [
                f"{name}:{tag}" for tag in ["XPOS_RD", "YPOS_RD", "PHASE_RD", "MAG_RD", "CURRENT_RD"
                ]]
            try:
                # load default BPMQ model
                if 'TIS161' in self.model_type:
                    fname = name.replace('_D','')[-7:]
                    state_dict = torch.load(os.path.join(script_dir,fname,'model.pt'))
                    model_info = {'n_node':len(state_dict['nn.2.bias']),
                                  'n_hidden_layer':int((len(state_dict)-2)/2)-2,
                                  'dtype':state_dict['nn.2.bias'].dtype}
                    model = BPMQ_model(**model_info)
                    model.load_state_dict(state_dict)
                    self.BPMQ_models[name] = model
                    if 'GP' in self.model_type:
                        checkpoints = torch.load(os.path.join(script_dir,fname,'gp.pt'))
                        # Extract data from the checkpoint
                        train_x = checkpoints['train_x']
                        train_y = checkpoints['train_y']
                        train_yvar = checkpoints['train_yvar']
                        # Initialize and load the GP model
                        gp = SingleTaskGP(train_x, train_y, train_yvar)
                        gp.load_state_dict(checkpoints['state_dict'])
                        checkpoints['model'] = gp
                        checkpoints.pop('state_dict')
                        self.BPMQ_GPs[name] = checkpoints
                elif 'TISRAW' in self.model_type:
                    fname = name.replace('_D','')[-7:]
                    checkpoints = torch.load(os.path.join(script_dir,fname,'model_TISRAW2BPMQ.pt'))
                    model = TISRAW2BPMQ_model(F=checkpoints['F'],dtype=checkpoints['dtype'])
                    model.load_state_dict(checkpoints['state_dict'])
                    self.BPMQ_models[name] = model
                    if 'GP' in self.model_type:
                        checkpoints = torch.load(os.path.join(script_dir,fname,'gp_TISRAW2BPMQ.pt'))
                        # Extract data from the checkpoint
                        train_x = checkpoints['train_x']
                        train_y = checkpoints['train_y']
                        train_yvar = checkpoints['train_yvar']
                        # Initialize and load the GP model
                        gp = SingleTaskGP(train_x, train_y, train_yvar)
                        gp.load_state_dict(checkpoints['state_dict'])
                        checkpoints['model'] = gp
                        checkpoints.pop('state_dict')
                        self.BPMQ_GPs[name] = checkpoints

            except Exception as e:
                print(f"Failed to load BPMQ model for {name}: {e}. BPMQ formula will be used instead.")
                
                
        self.PVs2read = PVs2read
        self.BPM_TISRAW_PVs = BPM_TISRAW_PVs
        self.BPM_TIS161_PVs = BPM_TIS161_PVs
        self.calibrated_PVs = calibrated_PVs
        self.BPM_TIS161_coeffs = np.array(BPM_TIS161_coeffs)


    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        '''
        df: DataFrame whose columns should include BPM_name:TIS161_i_RD, XPOS, YPOS,
        '''
        df = df.copy()  # Avoid modifying the original DataFrame
        if all(col in df.columns for col in self.BPM_TIS161_PVs):
            df[self.calibrated_PVs] = df[self.BPM_TIS161_PVs].values*self.BPM_TIS161_coeffs[None,:]
        
        for i,name in enumerate(self.BPM_names):
            model = self.BPMQ_models[name]
            if model and'TIS161' in self.model_type:
                U = self.calibrated_PVs[4*i:4*(i+1)]
                with torch.no_grad():
                    u_ = torch.tensor(df[U].astype(float).values,dtype=model.dtype)
                    x_ = torch.tensor(df[name+':XPOS_RD'].astype(float).values,dtype=model.dtype)
                    y_ = torch.tensor(df[name+':YPOS_RD'].astype(float).values,dtype=model.dtype)
                    # print(f"u_.shape: {u_.shape}, x_.shape: {x_.shape}, y_: {y_.shape}")
                    feature, beamQpred = model(u_,x_,y_)
                    # print(f"feature.shape: {feature.shape}, beamQpred.shape: {beamQpred.shape}")
                    df[name+':beamQ'] = beamQpred.cpu().numpy()
                    if 'GP' in self.model_type:
                        GP = self.BPMQ_GPs[name]
                        # print(f"GP['x_min'].shape: {GP['x_min'].shape}, GP['x_max'].shape: {GP['x_max'].shape}, GP['y_mean'].shape: {GP['y_mean'].shape}, GP['y_std'].shape: {GP['y_std'].shape}")
                        x_scaled = (feature - GP['x_min']) / (GP['x_max'] - GP['x_min'])
                        y_pred = GP['model'].posterior(x_scaled)
                        y_pred_mean = y_pred.mean[:,0] * GP['y_std'] + GP['y_mean'] + beamQpred  # y_pred.mean is shape of (batdh_size, 1)
                        y_pred_std  = y_pred.stddev * GP['y_std']                                # y_pred.stddev is shape of (batdh_size)
                        # print(f"y_pred_mean.shape: {y_pred_mean.shape}, y_pred_std.shape: {y_pred_std.shape}")
                        df[name+':beamQ'] = y_pred_mean.cpu().numpy().copy()
                        df[name+':beamQ_model_err'] = y_pred_std.cpu().numpy().copy()
                        
            elif model and 'TISRAW' in self.model_type:
                # test = df[name+f':TISRAW1_RD'].values
                # print(test.shape, test.dtype)
                # plt.figure(figsize=(4,2))
                # for t in test:
                #     print(t.shape, t.dtype, t[:4])
                #     if t.dtype == np.float64:
                #         plt.plot(t)
                #     else:
                #         print([type(_) for _ in t])
                # plt.show()
                ltis = [np.stack(df[name+f':TISRAW{j+1}_RD'].values,dtype=np.float32)*TIS161_coeffs[name][j]*1.1e-11 +0.5 for j in range(4)]
                with torch.no_grad():
                    ltis = [torch.tensor(tis,dtype=model.dtype) for tis in ltis]
                    # tis1_ = torch.tensor(df[name+':TISRAW1_RD'].values,dtype=model.dtype)
                    # tis2_ = torch.tensor(df[name+':TISRAW2_RD'].values,dtype=model.dtype)
                    # tis3_ = torch.tensor(df[name+':TISRAW3_RD'].values,dtype=model.dtype)
                    # tis4_ = torch.tensor(df[name+':TISRAW4_RD'].values,dtype=model.dtype)
                    
#                         print("ltis[0].shape",ltis[0].shape)
#                         print("ltis[1].shape",ltis[1].shape)
                    xy_ = torch.tensor(df[[name+':XPOS_RD',name+':YPOS_RD']].values,dtype=model.dtype)
                    #print("xy_.shape",xy_.shape)
                    #print("====",name,"====")
                    #print("utis1_:",tis1_)
                    #print(",model(tis1_,tis2_,tis3_,tis4_,xy_).cpu().numpy())
                    feature, beamQpred = model(*ltis,xy_)
                    df[name+':beamQ'] = beamQpred.cpu().numpy()
                    # df[name+':beamQ'] = model(tis1_,tis2_,tis3_,tis4_,xy_).cpu().numpy()
                    if 'GP' in self.model_type:
                        GP = self.BPMQ_GPs[name]
                        x_scaled = (feature - GP['x_min']) / (GP['x_max'] - GP['x_min'])
                        y_pred = GP['model'].posterior(x_scaled)
                        y_pred_mean = y_pred.mean[:,0] * GP['y_std'] + GP['y_mean']
                        y_pred_std  = y_pred.stddev * GP['y_std']
                        # print(f"y_pred_mean.shape: {y_pred_mean.shape}, y_pred_std.shape: {y_pred_std.shape}")
                        df[name+':beamQ'] = y_pred_mean.cpu().numpy()
                        df[name+':beamQ_model_err'] = y_pred_std.cpu().numpy()
            else:
                U = self.calibrated_PVs[4*i:4*(i+1)]
                diffsum = (df[[U[1],U[2]]].sum(axis=1) -df[[U[0],U[3]]].sum(axis=1)) / df[U].sum(axis=1)
                df[name+':beamQ'] = (241*diffsum - (df[name+':XPOS_RD']**2 - df[name+':YPOS_RD']**2))
        return df



