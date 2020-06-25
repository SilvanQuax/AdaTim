#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 10:34:08 2017

@author: michele
"""

import sys
import numpy as np
sys.path.append('folder')

import numpy
import chainer
from chainer.functions.activation import sigmoid
from chainer.functions.activation import tanh
from chainer.functions.activation import relu
#from chainer.functions.math import linear_interpolate
import chainer.functions as F
from chainer import link
#from chainer.links.connection import linear
from chainer import links
from My_Scale import Alpha


class Offset_C(link.Link):
    """
    Implementation of offset term to initialize Elman hidden states at t=0
    """

    def __init__(self, n_params):
        super(Offset_C, self).__init__()
        self.add_param('X', (1, n_params), initializer=chainer.initializers.Constant(0.2, dtype='float32'))
    def __call__(self, z):
        return F.broadcast_to(self.X, z.shape)

class Offset_G(link.Link):
    """
    Implementation of offset term to initialize Elman hidden states at t=0
    """
    def __init__(self, n_params):
        super(Offset_G, self).__init__()
        self.add_param('X', (1, n_params), initializer=chainer.initializers.HeNormal(1, dtype='float32'))
    def __call__(self, z):
        return F.broadcast_to(self.X, z.shape)    

class DRUBase(link.Chain):

    def __init__(self, n_units, n_inputs1, dim, n_inputs2=None, n_inputs3=None, Uinit=None,
                 Winit=None, bias_init=0):
        
        super(DRUBase, self).__init__()
        with self.init_scope():
            self.U1=links.Linear(n_inputs1, n_units,
                                  initialW=Uinit, nobias=True)#initial_bias=bias_init),
            if n_inputs2 is not None:
                self.U2=links.Linear(n_inputs2, n_units,
                                      initialW=Uinit, nobias=True)#initial_bias=bias_init),
            if n_inputs3 is not None:
                self.U3=links.Linear(n_inputs3, n_units,
                                      initialW=Uinit, nobias=True)#initial_bias=bias_init),
            self.W=links.Linear(n_units, n_units,
                                 initialW=Winit, nobias=True)
                            #initialW=inner_init, initial_bias=bias_init),
            self.Ou=Offset_G(n_units)
            if dim is not None:
                self.alphaS = Alpha(axis=0, W_shape=(1))
                self.alphaR = Alpha(axis=0, W_shape=(1))
            else:
                self.alphaS = Alpha(axis=0, W_shape=(1,n_units))
                self.alphaR = Alpha(axis=0, W_shape=(1,n_units))
                    

""" Class with LINKS-alike constants ---------------------------------------"""
class DRU(DRUBase):

    def __init__(self, out_size, input1_size, dim=None, input2_size=None, input3_size=None, U_init=None,
                 W_init=None, bias_init=0, act_func = sigmoid.sigmoid):
        super(DRU, self).__init__(
            out_size, input1_size, dim, input2_size, input3_size, U_init, W_init, bias_init)
        self.state_size = out_size
        #self.reset_state()
        self.act_func = act_func
#        self.alphaS.add_persistent('alphaR_init', np.array([[1.]]))
#        self.alphaR.add_persistent('alphaS_init', np.array([[1.]]))
        
    def to_cpu(self):
        super(DRU, self).to_cpu()
        if self.y is not None:
            self.y.to_cpu()
        if self.u is not None:
            self.u.to_cpu()

    def to_gpu(self, device=None):
        super(DRU, self).to_gpu(device)
        if self.y is not None:
            self.y.to_gpu(device)
        if self.u is not None:
            self.u.to_gpu(device)

    def set_state(self, u, y):        #to be deprecated
        assert isinstance(u, chainer.Variable)
        u_ = u
        if self.xp == numpy:
            u_.to_cpu()
        else:
            u_.to_gpu(self._device_id)
        self.u = u_
        assert isinstance(y, chainer.Variable)
        y_ = y
        if self.xp == numpy:
            y_.to_cpu()
        else:
            y_.to_gpu(self._device_id)
        self.y = y_
        

    def reset_state(self):              
        self.y = None
        self.u = None
    
    def initialize_state(self, batch_size=1):          #initialize to learnable initial value
#        o = np.zeros((batch_size, self.state_size))
#        for i in range(batch_size):
#            o[i,:] = self.Oy(np.zeros((1,self.state_size)))
#        self.y = o
        self.u = self.Ou(np.zeros((batch_size,self.state_size)))
        self.y = self.act_func(self.u)
        

    def __call__(self, x1, x2=None, x3=None):
        
        z = self.U1(x1)
        if x2 is not None:
            z+= self.U2(x2)
        if x3 is not None:
            z+= self.U3(x3)
        if self.u is None:
            self.u = self.Ou(z)
#        if self.y is None:
#            self.y = self.act_func(self.u)
        
        z += self.W(self.y)           
        
#        if self.u is None:
#            self.u = self.inverse_sigmoid(self.y)
            
        u_new = self.u + self.alphaS(z-self.u)

        self.u = u_new
        
        return self.u
    
    def firing_rate(self):
        self.y = self.y + self.alphaR(self.act_func(self.u)-self.y)
        #self.y = self.act_func(self.u)
        return self.y
    
    def return_hidden_state_as_array(self):
        h = np.zeros([self.state_size])
        if self.y is not None:
            h = self.y
            h = h.data
        return h   
        
    def inverse_sigmoid(self, x):
        y = -F.log((1./x) -1. )
        return y

    def set_this_neural_state(self, state):
        assert isinstance(state, chainer.Variable)
        self.y = state
        _u = self.inverse_sigmoid(state)
        assert isinstance(_u, chainer.Variable)
        self.u = _u
        
        
""" Static time constants version for grid search speed up --------------------
    use only to run gridsearch,
    and to check how much computation the alpha as Scale take"""


class DRUBase_static(link.Chain):

    def __init__(self, R, S, n_units, n_inputs1, n_inputs2=None, n_inputs3=None, Uinit=None,
                 Winit=None, bias_init=0):
        
        super(DRUBase_static, self).__init__()
        with self.init_scope():
            self.U1=links.Linear(n_inputs1, n_units,
                                  initialW=Uinit, nobias=True)#initial_bias=bias_init),
            if n_inputs2 is not None:
                self.U2=links.Linear(n_inputs2, n_units,
                                      initialW=Uinit, nobias=True)#initial_bias=bias_init),
            if n_inputs3 is not None:
                self.U3=links.Linear(n_inputs3, n_units,
                                      initialW=Uinit, nobias=True)#initial_bias=bias_init),
            self.W=links.Linear(n_units, n_units,
                                 initialW=Winit, nobias=True)
                            #initialW=inner_init, initial_bias=bias_init),
            
            self.alphaS = S
            self.alphaR = R
            self.Ou=Offset_G(n_units)
            #self.Oy=Offset_C(n_units)
        

""" Class with float-alike constants ---------------------------------------"""

class DRU_static(DRUBase_static):

    def __init__(self, R, S, out_size, input1_size, input2_size=None, input3_size=None, U_init=None,
                 W_init=None, bias_init=0, act_func = relu.relu):#sigmoid.sigmoid):
        super(DRU_static, self).__init__(
            R, S, out_size, input1_size, input2_size, input3_size, U_init, W_init, bias_init)
        self.state_size = out_size
        #self.reset_state()
        self.act_func = act_func
#        self.alphaS.add_persistent('alphaR_init', np.array([[1.]]))
#        self.alphaR.add_persistent('alphaS_init', np.array([[1.]]))
        
    def to_cpu(self):
        super(DRU, self).to_cpu()
        if self.y is not None:
            self.y.to_cpu()
        if self.u is not None:
            self.u.to_cpu()

    def to_gpu(self, device=None):
        super(DRU, self).to_gpu(device)
        if self.y is not None:
            self.y.to_gpu(device)
        if self.u is not None:
            self.u.to_gpu(device)

    def set_state(self, u, y):        #to be deprecated
        assert isinstance(u, chainer.Variable)
        u_ = u
        if self.xp == numpy:
            u_.to_cpu()
        else:
            u_.to_gpu(self._device_id)
        self.u = u_
        assert isinstance(y, chainer.Variable)
        y_ = y
        if self.xp == numpy:
            y_.to_cpu()
        else:
            y_.to_gpu(self._device_id)
        self.y = y_
        

    def reset_state(self):              
        self.y = None
        self.u = None
    
    def initialize_state(self, batch_size=1):          #initialize to learnable initial value
#        o = np.zeros((batch_size, self.state_size))
#        for i in range(batch_size):
#            o[i,:] = self.Oy(np.zeros((1,self.state_size)))
#        self.y = o
        self.u = self.Ou(np.zeros((batch_size,self.state_size)))
        self.y = self.act_func(self.u)
        

    def __call__(self, x1, x2=None, x3=None):
        
        z = self.U1(x1)
        if x2 is not None:
            z+= self.U2(x2)
        if x3 is not None:
            z+= self.U3(x3)
        if self.u is None:
            self.u = self.Ou(z)
#        if self.y is None:
#            self.y = self.act_func(self.u)
        
        z += self.W(self.y)           
        
#        if self.u is None:
#            self.u = self.inverse_sigmoid(self.y)
            
        u_new = self.u + self.alphaS*(z-self.u)

        self.u = u_new
        
        return self.u
    
    def firing_rate(self):
        self.y = self.y + self.alphaR*(self.act_func(self.u)-self.y)
        #self.y = self.act_func(self.u)
        return self.y
    
    def return_hidden_state_as_array(self):
        h = np.zeros([self.state_size])
        if self.y is not None:
            h = self.y
            h = h.data
        return h   
        
    def inverse_sigmoid(self, x):
        y = -F.log((1./x) -1. )
        return y

    def set_this_neural_state(self, state):
        assert isinstance(state, chainer.Variable)
        self.y = state
        _u = self.inverse_sigmoid(state)
        assert isinstance(_u, chainer.Variable)
        self.u = _u
