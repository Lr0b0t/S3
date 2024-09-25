#
# SPDX-FileCopyrightText: Copyright © 2022 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Alexandre Bittar <abittar@idiap.ch>
#
# SPDX-License-Identifier: BSD-3-Clause
#
# This file is part of the sparch package
#
"""
This is where the Spiking Neural Network (SNN) baseline is defined using the
surrogate gradient method.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import math
from einops import rearrange, repeat


class SpikeFunctionBoxcar(torch.autograd.Function):
    """
    Compute surrogate gradient of the spike step function using
    box-car function similar to DECOLLE, Kaiser et al. (2020).
    """

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x.gt(0).float()

    def backward(ctx, grad_spikes):
        (x,) = ctx.saved_tensors
        grad_x = grad_spikes.clone()
        grad_x[x <= -0.5] = 0
        grad_x[x > 0.5] = 0
        return grad_x

class SpikeFunctionSuperSpike(torch.autograd.Function):
    """
    Compute surrogate gradient of the spike step function using
    box-car function similar to DECOLLE, Kaiser et al. (2020).
    """

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x.gt(0).float()

    def backward(ctx, grad_spikes):
        (x,) = ctx.saved_tensors
        grad_x = grad_spikes.clone()
        grad_out = grad_x / (1.0 + 10.0*torch.abs(x))
        return grad_out

class SpikeFunctionSLAYER(torch.autograd.Function):
    """
    Compute surrogate gradient of the spike step function using
    box-car function similar to DECOLLE, Kaiser et al. (2020).
    """

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x.gt(0).float()

    def backward(ctx, grad_spikes):
        (x,) = ctx.saved_tensors
        grad_x = grad_spikes.clone()
        alpha=5
        c=0.4
        grad_out = grad_x * c * alpha / (2 * torch.exp(x.abs() * alpha))
        return grad_out

# def mem_reset(mem, thresh):
#     """Generates detached reset signal if mem > threshold.
#     Returns reset."""
#     mem_shift = mem - thresh
#     reset = SpikeFunctionBoxcar.apply(mem_shift).clone().detach()

#     return reset

class SNN(nn.Module):
    """
    A multi-layered Spiking Neural Network (SNN).

    It accepts input tensors formatted as (batch, time, feat). In the case of
    4d inputs like (batch, time, feat, channel) the input is flattened as
    (batch, time, feat*channel).

    The function returns the outputs of the last spiking or readout layer
    with shape (batch, time, feats) or (batch, feats) respectively, as well
    as the firing rates of all hidden neurons with shape (num_layers*feats).

    Arguments
    ---------
    input_shape : tuple
        Shape of an input example.
    layer_sizes : int list
        List of number of neurons in all hidden layers
    neuron_type : str
        Type of neuron model, either 'LIF', 'adLIF', 'RLIF' or 'RadLIF'.
    threshold : float
        Fixed threshold value for the membrane potential.
    dropout : float
        Dropout rate (must be between 0 and 1).
    normalization : str
        Type of normalization (batchnorm, layernorm). Every string different
        from batchnorm and layernorm will result in no normalization.
    use_bias : bool
        If True, additional trainable bias is used with feedforward weights.
    bidirectional : bool
        If True, a bidirectional model that scans the sequence both directions
        is used, which doubles the size of feedforward matrices in layers l>0.
    use_readout_layer : bool
        If True, the final layer is a non-spiking, non-recurrent LIF and outputs
        a cumulative sum of the membrane potential over time. The outputs have
        shape (batch, labels) with no time dimension. If False, the final layer
        is the same as the hidden layers and outputs spike trains with shape
        (batch, time, labels).
    """

    def __init__(
        self,
        input_shape,
        layer_sizes,
        neuron_type="LIF",
        threshold=1.0,
        dropout=0.0,
        normalization="batchnorm",
        use_bias=False,
        bidirectional=False,
        use_readout_layer=True,
        extra_features=None
    ):
        super().__init__()

        # Fixed parameters
        self.reshape = True if len(input_shape) > 3 else False
        self.input_size = float(torch.prod(torch.tensor(input_shape[2:])))
        self.batch_size = input_shape[0]
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes)
        self.num_outputs = layer_sizes[-1]
        self.neuron_type = neuron_type
        self.threshold = threshold
        self.dropout = dropout
        self.normalization = normalization
        self.use_bias = use_bias
        self.bidirectional = bidirectional
        self.use_readout_layer = use_readout_layer
        self.is_snn = True

        self.extra_features = extra_features

        if neuron_type not in ["LIF", "adLIF", "CadLIF", "RSEadLIF", "LIFfeature", "adLIFnoClamp","LIFfeatureDim", "adLIFclamp", "RLIF", "RadLIF", "LIFcomplex", "LIFrealcomplex","ReLULIFcomplex", "RLIFcomplex","RLIFcomplex1MinAlphaNoB","RLIFcomplex1MinAlpha", "LIFcomplex_gatedB", "LIFcomplex_gatedDt", "LIFcomplexDiscr"]:
            raise ValueError(f"Invalid neuron type {neuron_type}")

        # Init trainable parameters
        self.snn = self._init_layers()

    def _init_layers(self):

        snn = nn.ModuleList([])
        input_size = self.input_size
        snn_class = self.neuron_type + "Layer"

        if self.use_readout_layer:
            num_hidden_layers = self.num_layers - 1
        else:
            num_hidden_layers = self.num_layers
            
        for i in range(num_hidden_layers):
            snn.append(
                globals()[snn_class](
                    input_size=input_size,
                    hidden_size=self.layer_sizes[i],
                    batch_size=self.batch_size,
                    threshold=self.threshold,
                    dropout=self.dropout,
                    normalization=self.normalization,
                    use_bias=self.use_bias,
                    bidirectional=self.bidirectional,
                    extra_features = self.extra_features
                )
            )
            input_size = self.layer_sizes[i] * (1 + self.bidirectional)

        # Readout layer
        if self.use_readout_layer:
            if self.neuron_type == 'RSEadLIF':
                snn.append(
                    SEReadoutLayer(
                        input_size=input_size,
                        hidden_size=self.layer_sizes[-1],
                        batch_size=self.batch_size,
                        dropout=self.dropout,
                        normalization=self.normalization,
                        use_bias=self.use_bias,
                    )
                )
            else:
                snn.append(
                    ReadoutLayer(
                        input_size=input_size,
                        hidden_size=self.layer_sizes[-1],
                        batch_size=self.batch_size,
                        dropout=self.dropout,
                        normalization=self.normalization,
                        use_bias=self.use_bias,
                        extra_features=self.extra_features
                    )
                )

        return snn

    def forward(self, x):

        # Reshape input tensors to (batch, time, feats) for 4d inputs
        if self.reshape:
            if x.ndim == 4:
                x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3])
            else:
                raise NotImplementedError

        # Process all layers
        all_spikes = []


        if self.extra_features['residual']:
            res = 0
            for i, snn_lay in enumerate(self.snn):
                if not (self.use_readout_layer and i == self.num_layers - 1):
                    x = snn_lay(x) + res
                    res = x
                    all_spikes.append(x)  
                else:
                    x = snn_lay(x)
        else:
            for i, snn_lay in enumerate(self.snn):
                x = snn_lay(x)
                if not (self.use_readout_layer and i == self.num_layers - 1):
                    all_spikes.append(x)

        # Compute mean firing rate of each spiking neuron
        firing_rates = torch.cat(all_spikes, dim=2).mean(dim=(0, 1))

        return x, firing_rates


class LIFLayer(nn.Module):
    """
    A single layer of Leaky Integrate-and-Fire neurons without layer-wise
    recurrent connections (LIF).

    Arguments
    ---------
    input_size : int
        Number of features in the input tensors.
    hidden_size : int
        Number of output neurons.
    batch_size : int
        Batch size of the input tensors.
    threshold : float
        Value of spiking threshold (fixed)
    dropout : float
        Dropout factor (must be between 0 and 1).
    normalization : str
        Type of normalization. Every string different from 'batchnorm'
        and 'layernorm' will result in no normalization.
    use_bias : bool
        If True, additional trainable bias is used with feedforward weights.
    bidirectional : bool
        If True, a bidirectional model that scans the sequence both directions
        is used, which doubles the size of feedforward matrices in layers l>0.
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        batch_size,
        threshold=1.0,
        dropout=0.0,
        normalization="batchnorm",
        use_bias=False,
        bidirectional=False,
        extra_features=None
    ):
        super().__init__()

        # Fixed parameters
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.batch_size = batch_size
        self.threshold = threshold
        self.dropout = dropout
        self.normalization = normalization
        self.use_bias = use_bias
        self.bidirectional = bidirectional
        self.batch_size = self.batch_size * (1 + self.bidirectional)
        self.alpha_lim = [np.exp(-1 / 5), np.exp(-1 / 25)]
        self.spike_fct = SpikeFunctionBoxcar.apply

        # Trainable parameters
        self.W = nn.Linear(self.input_size, self.hidden_size, bias=use_bias)
        self.alpha = nn.Parameter(torch.Tensor(self.hidden_size))
        nn.init.uniform_(self.alpha, self.alpha_lim[0], self.alpha_lim[1])

        # Initialize normalization
        self.normalize = False
        if normalization == "batchnorm":
            self.norm = nn.BatchNorm1d(self.hidden_size, momentum=0.05)
            self.normalize = True
        elif normalization == "layernorm":
            self.norm = nn.LayerNorm(self.hidden_size)
            self.normalize = True

        # Initialize dropout
        self.drop = nn.Dropout(p=dropout)

        if extra_features['rst_detach']:
            self.rst_detach = True
        else:
            self.rst_detach = False
    def forward(self, x):

        # Concatenate flipped sequence on batch dim
        if self.bidirectional:
            x_flip = x.flip(1)
            x = torch.cat([x, x_flip], dim=0)

        # Change batch size if needed
        if self.batch_size != x.shape[0]:
            self.batch_size = x.shape[0]

        # Feed-forward affine transformations (all steps in parallel)
        Wx = self.W(x)

        # Apply normalization
        if self.normalize:
            _Wx = self.norm(Wx.reshape(Wx.shape[0] * Wx.shape[1], Wx.shape[2]))
            Wx = _Wx.reshape(Wx.shape[0], Wx.shape[1], Wx.shape[2])

        # Compute spikes via neuron dynamics
        s = self._lif_cell(Wx)

        # Concatenate forward and backward sequences on feat dim
        if self.bidirectional:
            s_f, s_b = s.chunk(2, dim=0)
            s_b = s_b.flip(1)
            s = torch.cat([s_f, s_b], dim=2)

        # Apply dropout
        s = self.drop(s)

        return s

    def _lif_cell(self, Wx):

        # Initializations
        device = Wx.device
        ut = torch.rand(Wx.shape[0], Wx.shape[2]).to(device)
        st = torch.rand(Wx.shape[0], Wx.shape[2]).to(device)
        s = []

        # Bound values of the neuron parameters to plausible ranges
        alpha = torch.clamp(self.alpha, min=self.alpha_lim[0], max=self.alpha_lim[1])

        # Loop over time axis
        for t in range(Wx.shape[1]):
            
            if self.rst_detach:
                reset = st.clone().detach()
            else: 
                reset = st

            # Compute membrane potential (LIF)
            ut = alpha * (ut - reset) + (1 - alpha) * Wx[:, t, :]

            # Compute spikes with surrogate gradient
            st = self.spike_fct(ut - self.threshold)
            s.append(st)

        return torch.stack(s, dim=1)

class LIFfeatureLayer(nn.Module):
    """
    A single layer of Leaky Integrate-and-Fire neurons without layer-wise
    recurrent connections (LIF).

    Arguments
    ---------
    input_size : int
        Number of features in the input tensors.
    hidden_size : int
        Number of output neurons.
    batch_size : int
        Batch size of the input tensors.
    threshold : float
        Value of spiking threshold (fixed)
    dropout : float
        Dropout factor (must be between 0 and 1).
    normalization : str
        Type of normalization. Every string different from 'batchnorm'
        and 'layernorm' will result in no normalization.
    use_bias : bool
        If True, additional trainable bias is used with feedforward weights.
    bidirectional : bool
        If True, a bidirectional model that scans the sequence both directions
        is used, which doubles the size of feedforward matrices in layers l>0.
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        batch_size,
        threshold=1.0,
        dropout=0.0,
        normalization="batchnorm",
        use_bias=False,
        bidirectional=False,
        extra_features="_"
    ):
        super().__init__()

        # Fixed parameters
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.batch_size = batch_size
        self.threshold = threshold
        self.dropout = dropout
        self.normalization = normalization
        self.use_bias = use_bias
        self.bidirectional = bidirectional
        self.batch_size = self.batch_size * (1 + self.bidirectional)
        
        
        self.spike_fct = SpikeFunctionBoxcar.apply

        self.b = nn.Parameter(torch.rand(self.hidden_size))

        # Trainable parameters
        self.W = nn.Linear(self.input_size, self.hidden_size, bias=use_bias)
        self.extra_features = extra_features
        device = torch.device("cuda")
        if "1-200_1-5"  in extra_features:
            self.alpha_lim = [np.exp(-1 / 5), np.exp(-1 / 200)]
        else:
            self.alpha_lim = [np.exp(-1 / 5), np.exp(-1 / 25)]
        dt = 1
        if "Dt1ms" in extra_features:
            dt = 0.001
        elif "Dt1" in extra_features:
            dt = 1
        if "dtParam" in extra_features:            
            self.register("dt", torch.ones(1)*dt, lr=0.01)
        else:
            self.dt = dt
        

        dt_min = 0.01
        dt_max = 0.4

        if "dtLog" in extra_features:
            log_dt = torch.rand(self.hidden_size)*(
            math.log(dt_max) - math.log(dt_min)
            ) + math.log(dt_min)
            self.register("log_dt", log_dt, lr=0.01)

        if  "logAlpha" in extra_features:
            self.log_alpha = nn.Parameter(torch.Tensor(self.hidden_size))
            nn.init.uniform_(self.log_alpha,torch.log(torch.tensor(self.alpha_lim[0])), torch.log(torch.tensor(self.alpha_lim[1])))

        elif "cont" in extra_features:
            if "A0_5" in extra_features:
                log_log_alpha = torch.log(0.5 * torch.ones(self.hidden_size)).to(device)
                self.register("log_log_alpha", log_log_alpha, lr=0.01)
            elif "A0_5Const" in extra_features:
                self.log_log_alpha = torch.log(0.5 * torch.ones(self.hidden_size)).to(device)
            else:
                self.log_log_alpha = nn.Parameter(torch.Tensor(self.hidden_size))
                nn.init.uniform_(self.log_log_alpha, torch.log(-torch.log(torch.tensor(self.alpha_lim[1]))/self.dt), torch.log(-torch.log(torch.tensor(self.alpha_lim[0]))/self.dt))
        
                        

        else:
            self.alpha = nn.Parameter(torch.Tensor(self.hidden_size))
            nn.init.uniform_(self.alpha, self.alpha_lim[0], self.alpha_lim[1])
            self.log_log_alpha = torch.ones(self.hidden_size).to(device)

        
        
        if "imag" in extra_features:
            alpha_img =  math.pi * torch.ones(self.hidden_size).to(device) # torch.arange(self.hidden_size)
            self.register("alpha_img", alpha_img, lr=0.01)
 

        # Initialize normalization
        self.normalize = False
        if normalization == "batchnorm":
            self.norm = nn.BatchNorm1d(self.hidden_size, momentum=0.05)
            self.normalize = True
        elif normalization == "layernorm":
            self.norm = nn.LayerNorm(self.hidden_size)
            self.normalize = True

        # Initialize dropout
        self.drop = nn.Dropout(p=dropout)

        if extra_features['rst_detach']:
            self.rst_detach = True
        else:
            self.rst_detach = False

    def forward(self, x):

        # Concatenate flipped sequence on batch dim
        if self.bidirectional:
            x_flip = x.flip(1)
            x = torch.cat([x, x_flip], dim=0)

        # Change batch size if needed
        if self.batch_size != x.shape[0]:
            self.batch_size = x.shape[0]

        # Feed-forward affine transformations (all steps in parallel)
        Wx = self.W(x)

        # Apply normalization
        if self.normalize:
            _Wx = self.norm(Wx.reshape(Wx.shape[0] * Wx.shape[1], Wx.shape[2]))
            Wx = _Wx.reshape(Wx.shape[0], Wx.shape[1], Wx.shape[2])

        # Compute spikes via neuron dynamics
        s = self._lif_cell(Wx)

        # Concatenate forward and backward sequences on feat dim
        if self.bidirectional:
            s_f, s_b = s.chunk(2, dim=0)
            s_b = s_b.flip(1)
            s = torch.cat([s_f, s_b], dim=2)

        # Apply dropout
        s = self.drop(s)

        return s

    def _lif_cell(self, Wx):

        # Initializations
        device = Wx.device
        ut = torch.rand(Wx.shape[0], Wx.shape[2]).to(device)
        st = torch.rand(Wx.shape[0], Wx.shape[2]).to(device)
        s = []
        if "imag"  in self.extra_features:
            eigenval = -torch.exp(self.log_log_alpha)+1j*self.alpha_img
        else:
            eigenval = -torch.exp(self.log_log_alpha) 

        if "dtLog"  in self.extra_features:
            self.dt = torch.exp(self.log_dt)
            
        
        if "logAlpha" in self.extra_features :
            alpha = torch.exp(self.log_alpha)
        elif "cont" in self.extra_features:
            alpha = torch.exp(self.dt*eigenval)
        else:
            alpha = self.alpha
        if "NoClamp" not in self.extra_features:
            alpha = torch.clamp(alpha, min=self.alpha_lim[0], max=self.alpha_lim[1])

        if "B" in  self.extra_features:
            b = self.b
        else:
            b = (1 - alpha)

        # Loop over time axis
        for t in range(Wx.shape[1]):
            
            if self.rst_detach:
                reset = st.clone().detach()
            else: 
                reset = st

            # Compute membrane potential (LIF)
            ut = alpha * (ut - reset) + b * Wx[:, t, :]

            # Compute spikes with surrogate gradient
            if "imag"  in self.extra_features:
                st = self.spike_fct(2*ut.real - self.threshold)
            else:
                st = self.spike_fct(ut - self.threshold)
            s.append(st)

        return torch.stack(s, dim=1)

    def register(self, name, tensor, lr=None):
        """Register a tensor with a configurable learning rate and 0 weight decay"""

        if lr == 0.0:
            self.register_buffer(name, tensor)
        else:
            self.register_parameter(name, nn.Parameter(tensor))

            optim = {"weight_decay": 0.0}
            if lr is not None: optim["lr"] = lr
            setattr(getattr(self, name), "_optim", optim)

class LIFfeatureDimLayer(nn.Module):
    """
    A single layer of Leaky Integrate-and-Fire neurons without layer-wise
    recurrent connections (LIF).

    Arguments
    ---------
    input_size : int
        Number of features in the input tensors.
    hidden_size : int
        Number of output neurons.
    batch_size : int
        Batch size of the input tensors.
    threshold : float
        Value of spiking threshold (fixed)
    dropout : float
        Dropout factor (must be between 0 and 1).
    normalization : str
        Type of normalization. Every string different from 'batchnorm'
        and 'layernorm' will result in no normalization.
    use_bias : bool
        If True, additional trainable bias is used with feedforward weights.
    bidirectional : bool
        If True, a bidirectional model that scans the sequence both directions
        is used, which doubles the size of feedforward matrices in layers l>0.
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        batch_size,
        threshold=1.0,
        dropout=0.0,
        normalization="batchnorm",
        use_bias=False,
        bidirectional=False,
        extra_features="_"
    ):
        super().__init__()

        # Fixed parameters
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.batch_size = batch_size
        self.threshold = threshold
        self.dropout = dropout
        self.normalization = normalization
        self.use_bias = use_bias
        self.bidirectional = bidirectional
        self.batch_size = self.batch_size * (1 + self.bidirectional)
        
        
        self.spike_fct = SpikeFunctionBoxcar.apply

        
        self.dim = 1
        if "dim2"  in extra_features:
            self.dim = 2
        self.b = nn.Parameter(torch.rand(self.hidden_size, self.dim)*0.5)

        # Trainable parameters
        self.W = nn.Linear(self.input_size, self.hidden_size, bias=use_bias)
        self.extra_features = extra_features
        device = torch.device("cuda")
        if "1-200_1-5"  in extra_features:
            self.alpha_lim = [np.exp(-1 / 5), np.exp(-1 / 200)]
        else:
            self.alpha_lim = [np.exp(-1 / 5), np.exp(-1 / 25)]
        dt = 1
        if "Dt1ms" in extra_features:
            dt = 0.001
        elif "Dt1" in extra_features:
            dt = 1
        if "dtParam" in extra_features:            
            self.register("dt", torch.ones(1)*dt, lr=0.01)
        else:
            self.dt = dt
        
        dt_min = 0.01
        dt_max = 0.4

        if "dtLog" in extra_features:
            log_dt = torch.rand(self.hidden_size)*(
            math.log(dt_max) - math.log(dt_min)
            ) + math.log(dt_min)
            self.register("log_dt", log_dt, lr=0.01)

        if  "logAlpha" in extra_features:
            self.log_alpha = nn.Parameter(torch.Tensor(self.hidden_size))
            nn.init.uniform_(self.log_alpha,torch.log(torch.tensor(self.alpha_lim[0])), torch.log(torch.tensor(self.alpha_lim[1])))

        elif "cont" in extra_features:
            if "A0_5" in extra_features:
                log_log_alpha = torch.log(0.5 * torch.ones(self.hidden_size)).to(device)
                self.register("log_log_alpha", log_log_alpha, lr=0.01)
            elif "A0_5Const" in extra_features:
                self.log_log_alpha = torch.log(0.5 * torch.ones(self.hidden_size)).to(device)
            else:
                self.log_log_alpha = nn.Parameter(torch.Tensor(self.hidden_size, self.dim))
                nn.init.uniform_(self.log_log_alpha, torch.log(-torch.log(torch.tensor(self.alpha_lim[1]))/self.dt), torch.log(-torch.log(torch.tensor(self.alpha_lim[0]))/self.dt))
        
                        

        else:
            self.alpha = nn.Parameter(torch.Tensor(self.hidden_size))
            nn.init.uniform_(self.alpha, self.alpha_lim[0], self.alpha_lim[1])
            self.log_log_alpha = torch.ones(self.hidden_size).to(device)

        
        
        if "imag" in extra_features:
            alpha_img =  math.pi * torch.ones(self.hidden_size).to(device) # torch.arange(self.hidden_size)
            self.register("alpha_img", alpha_img, lr=0.01)

        # Initialize normalization
        self.normalize = False
        if normalization == "batchnorm":
            self.norm = nn.BatchNorm1d(self.hidden_size, momentum=0.05)
            self.normalize = True
        elif normalization == "layernorm":
            self.norm = nn.LayerNorm(self.hidden_size)
            self.normalize = True

        # Initialize dropout
        self.drop = nn.Dropout(p=dropout)

    def forward(self, x):

        # Concatenate flipped sequence on batch dim
        if self.bidirectional:
            x_flip = x.flip(1)
            x = torch.cat([x, x_flip], dim=0)

        # Change batch size if needed
        if self.batch_size != x.shape[0]:
            self.batch_size = x.shape[0]

        # Feed-forward affine transformations (all steps in parallel)
        Wx = self.W(x)

        # Apply normalization
        if self.normalize:
            _Wx = self.norm(Wx.reshape(Wx.shape[0] * Wx.shape[1], Wx.shape[2]))
            Wx = _Wx.reshape(Wx.shape[0], Wx.shape[1], Wx.shape[2])

        # Compute spikes via neuron dynamics
        s = self._lif_cell(Wx)

        # Concatenate forward and backward sequences on feat dim
        if self.bidirectional:
            s_f, s_b = s.chunk(2, dim=0)
            s_b = s_b.flip(1)
            s = torch.cat([s_f, s_b], dim=2)

        # Apply dropout
        s = self.drop(s)

        return s

    def _lif_cell(self, Wx):

        # Initializations
        device = Wx.device
        ut = torch.rand(Wx.shape[0], Wx.shape[2], self.dim).to(device)
        st = torch.rand(Wx.shape[0], Wx.shape[2]).to(device)
        s = []
        if "imag"  in self.extra_features:
            eigenval = -torch.exp(self.log_log_alpha)+1j*self.alpha_img
        else:
            eigenval = -torch.exp(self.log_log_alpha) 

        if "dtLog"  in self.extra_features:
            self.dt = torch.exp(self.log_dt)
            
        
        if "logAlpha" in self.extra_features :
            alpha = torch.exp(self.log_alpha)
        elif "cont" in self.extra_features:
            alpha = torch.exp(self.dt*eigenval)
        else:
            alpha = self.alpha

        if "B" in  self.extra_features:
            b = self.b
        else:
            b = (1 - alpha)

        # Loop over time axis
        for t in range(Wx.shape[1]):

            # Compute membrane potential (LIF)
            ut = alpha * (ut -  st.unsqueeze(-1).expand(-1,-1, self.dim)) + self.b * Wx[:, t, :].unsqueeze(-1).expand(-1,-1, self.dim)

            # Compute spikes with surrogate gradient
            if "imag"  in self.extra_features:
                st = self.spike_fct(2*ut.real - self.threshold)
            else:
                st = self.spike_fct(0.5*torch.sum(ut, dim=-1).real - self.threshold)
            s.append(st)

        return torch.stack(s, dim=1)

    def register(self, name, tensor, lr=None):
        """Register a tensor with a configurable learning rate and 0 weight decay"""

        if lr == 0.0:
            self.register_buffer(name, tensor)
        else:
            self.register_parameter(name, nn.Parameter(tensor))

            optim = {"weight_decay": 0.0}
            if lr is not None: optim["lr"] = lr
            setattr(getattr(self, name), "_optim", optim)



class adLIFLayer(nn.Module):
    """
    A single layer of adaptive Leaky Integrate-and-Fire neurons without
    layer-wise recurrent connections (adLIF).

    Arguments
    ---------
    input_size : int
        Number of features in the input tensors.
    hidden_size : int
        Number of output neurons.
    batch_size : int
        Batch size of the input tensors.
    threshold : float
        Value of spiking threshold (fixed)
    dropout : float
        Dropout factor (must be between 0 and 1).
    normalization : str
        Type of normalization. Every string different from 'batchnorm'
        and 'layernorm' will result in no normalization.
    use_bias : bool
        If True, additional trainable bias is used with feedforward weights.
    bidirectional : bool
        If True, a bidirectional model that scans the sequence both directions
        is used, which doubles the size of feedforward matrices in layers l>0.
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        batch_size,
        threshold=1.0,
        dropout=0.0,
        normalization="batchnorm",
        use_bias=False,
        bidirectional=False,
        extra_features=None
    ):
        super().__init__()

        # Fixed parameters
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.batch_size = batch_size
        self.threshold = threshold
        self.dropout = dropout
        self.normalization = normalization
        self.use_bias = use_bias
        self.bidirectional = bidirectional
        self.batch_size = self.batch_size * (1 + self.bidirectional)
        self.alpha_lim = [np.exp(-1 / 5), np.exp(-1 / 25)]
        self.beta_lim = [np.exp(-1 / 30), np.exp(-1 / 120)]
        self.a_lim = [-1.0, 1.0]
        self.b_lim = [0.0, 2.0]
        self.spike_fct = SpikeFunctionBoxcar.apply

        # Trainable parameters
        self.W = nn.Linear(self.input_size, self.hidden_size, bias=use_bias)
        self.alpha = nn.Parameter(torch.Tensor(self.hidden_size))
        self.beta = nn.Parameter(torch.Tensor(self.hidden_size))
        self.a = nn.Parameter(torch.Tensor(self.hidden_size))
        self.b = nn.Parameter(torch.Tensor(self.hidden_size))

        nn.init.uniform_(self.alpha, self.alpha_lim[0], self.alpha_lim[1])
        nn.init.uniform_(self.beta, self.beta_lim[0], self.beta_lim[1])
        nn.init.uniform_(self.a, self.a_lim[0], self.a_lim[1])
        nn.init.uniform_(self.b, self.b_lim[0], self.b_lim[1])

        # Initialize normalization
        self.normalize = False
        if normalization == "batchnorm":
            self.norm = nn.BatchNorm1d(self.hidden_size, momentum=0.05)
            self.normalize = True
        elif normalization == "layernorm":
            self.norm = nn.LayerNorm(self.hidden_size)
            self.normalize = True

        # Initialize dropout
        self.drop = nn.Dropout(p=dropout)

        if extra_features['rst_detach']:
            self.rst_detach = True
        else:
            self.rst_detach = False

        if extra_features['no_reset']:
            self.reset_factor = 0
        else: 
            self.reset_factor = 1

    def forward(self, x):

        # Concatenate flipped sequence on batch dim
        if self.bidirectional:
            x_flip = x.flip(1)
            x = torch.cat([x, x_flip], dim=0)

        # Change batch size if needed
        if self.batch_size != x.shape[0]:
            self.batch_size = x.shape[0]

        # Feed-forward affine transformations (all steps in parallel)
        Wx = self.W(x)

        # Apply normalization
        if self.normalize:
            _Wx = self.norm(Wx.reshape(Wx.shape[0] * Wx.shape[1], Wx.shape[2]))
            Wx = _Wx.reshape(Wx.shape[0], Wx.shape[1], Wx.shape[2])

        # Compute spikes via neuron dynamics
        s = self._adlif_cell(Wx)

        # Concatenate forward and backward sequences on feat dim
        if self.bidirectional:
            s_f, s_b = s.chunk(2, dim=0)
            s_b = s_b.flip(1)
            s = torch.cat([s_f, s_b], dim=2)

        # Apply dropout
        s = self.drop(s)

        return s

    def _adlif_cell(self, Wx):

        # Initializations
        device = Wx.device
        ut = torch.rand(Wx.shape[0], Wx.shape[2]).to(device)
        wt = torch.rand(Wx.shape[0], Wx.shape[2]).to(device)
        st = torch.rand(Wx.shape[0], Wx.shape[2]).to(device)
        s = []

        # Bound values of the neuron parameters to plausible ranges
        alpha = torch.clamp(self.alpha, min=self.alpha_lim[0], max=self.alpha_lim[1])
        beta = torch.clamp(self.beta, min=self.beta_lim[0], max=self.beta_lim[1])
        a = torch.clamp(self.a, min=self.a_lim[0], max=self.a_lim[1])
        b = torch.clamp(self.b, min=self.b_lim[0], max=self.b_lim[1])

        # Loop over time axis
        for t in range(Wx.shape[1]):

            if self.rst_detach:
                reset = st.clone().detach()
            else:
                reset = st


            # Compute potential (adLIF)
            wt = beta * wt + a * ut + b * reset * self.reset_factor
            ut = alpha * (ut - reset* self.reset_factor) + (1 - alpha)* (Wx[:, t, :] - wt)

            # Compute spikes with surrogate gradient
            st = self.spike_fct(ut - self.threshold)
            s.append(st)

        return torch.stack(s, dim=1)

class CadLIFLayer(nn.Module):
    """
    A single layer of adaptive Leaky Integrate-and-Fire neurons without
    layer-wise recurrent connections (adLIF).

    Arguments
    ---------
    input_size : int
        Number of features in the input tensors.
    hidden_size : int
        Number of output neurons.
    batch_size : int
        Batch size of the input tensors.
    threshold : float
        Value of spiking threshold (fixed)
    dropout : float
        Dropout factor (must be between 0 and 1).
    normalization : str
        Type of normalization. Every string different from 'batchnorm'
        and 'layernorm' will result in no normalization.
    use_bias : bool
        If True, additional trainable bias is used with feedforward weights.
    bidirectional : bool
        If True, a bidirectional model that scans the sequence both directions
        is used, which doubles the size of feedforward matrices in layers l>0.
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        batch_size,
        threshold=1.0,
        dropout=0.0,
        normalization="batchnorm",
        use_bias=False,
        bidirectional=False,
        extra_features=None
    ):
        super().__init__()

        # Fixed parameters
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.batch_size = batch_size
        self.threshold = threshold
        self.dropout = dropout
        self.normalization = normalization
        self.use_bias = use_bias
        self.bidirectional = bidirectional
        self.batch_size = self.batch_size * (1 + self.bidirectional)
        self.alpha_lim = [0.36, 0.96]
        self.beta_lim = [0.96, 0.99]
        self.a_lim = [0.0, 1.0]
        self.b_lim = [0.0, 2.0]
        self.spike_fct = SpikeFunctionBoxcar.apply

        # Trainable parameters
        self.W = nn.Linear(self.input_size, self.hidden_size, bias=use_bias)
        init.xavier_uniform_(self.W.weight)
        self.alpha = nn.Parameter(torch.Tensor(self.hidden_size))
        self.beta = nn.Parameter(torch.Tensor(self.hidden_size))
        self.a = nn.Parameter(torch.Tensor(self.hidden_size))
        self.b = nn.Parameter(torch.Tensor(self.hidden_size))

        nn.init.uniform_(self.alpha, self.alpha_lim[0], self.alpha_lim[1])
        nn.init.uniform_(self.beta, self.beta_lim[0], self.beta_lim[1])
        nn.init.uniform_(self.a, self.a_lim[0], self.a_lim[1])
        nn.init.uniform_(self.b, self.b_lim[0], self.b_lim[1])

        # Initialize normalization
        self.normalize = False
        if normalization == "batchnorm":
            self.norm = nn.BatchNorm1d(self.hidden_size, momentum=0.05)
            self.normalize = True
        elif normalization == "layernorm":
            self.norm = nn.LayerNorm(self.hidden_size)
            self.normalize = True

        # Initialize dropout
        self.drop = nn.Dropout(p=dropout)

        if extra_features['rst_detach']:
            self.rst_detach = True
        else:
            self.rst_detach = False

    def forward(self, x):

        # Concatenate flipped sequence on batch dim
        if self.bidirectional:
            x_flip = x.flip(1)
            x = torch.cat([x, x_flip], dim=0)

        # Change batch size if needed
        if self.batch_size != x.shape[0]:
            self.batch_size = x.shape[0]

        # Feed-forward affine transformations (all steps in parallel)
        Wx = self.W(x)

        # Apply normalization
        if self.normalize:
            _Wx = self.norm(Wx.reshape(Wx.shape[0] * Wx.shape[1], Wx.shape[2]))
            Wx = _Wx.reshape(Wx.shape[0], Wx.shape[1], Wx.shape[2])

        # Compute spikes via neuron dynamics
        s = self._cadlif_cell(Wx)

        # Concatenate forward and backward sequences on feat dim
        if self.bidirectional:
            s_f, s_b = s.chunk(2, dim=0)
            s_b = s_b.flip(1)
            s = torch.cat([s_f, s_b], dim=2)

        # Apply dropout
        s = self.drop(s)

        return s

    def _cadlif_cell(self, Wx):

        # Initializations
        device = Wx.device
        ut = torch.rand(Wx.shape[0], Wx.shape[2]).to(device)
        wt = torch.rand(Wx.shape[0], Wx.shape[2]).to(device)
        st = torch.rand(Wx.shape[0], Wx.shape[2]).to(device)
        s = []

        # Bound values of the neuron parameters to plausible ranges
        alpha = torch.clamp(self.alpha, min=self.alpha_lim[0], max=self.alpha_lim[1])
        beta = torch.clamp(self.beta, min=self.beta_lim[0], max=self.beta_lim[1])
        a = torch.clamp(self.a, min=self.a_lim[0], max=self.a_lim[1])
        b = torch.clamp(self.b, min=self.b_lim[0], max=self.b_lim[1])

        # Loop over time axis
        for t in range(Wx.shape[1]):

            # if self.rst_detach:
            #     reset = st.clone().detach()
            # else:
            #     reset = st

            # Compute potential (adLIF)
            wt = beta * wt + a * ut + b * st
            ut = alpha * (ut - st) + (1 - alpha)* (Wx[:, t, :] - wt)

            # Compute spikes with surrogate gradient
            st = self.spike_fct(ut - self.threshold)
            s.append(st)

        return torch.stack(s, dim=1)

class RSEadLIFLayer(nn.Module):
    """
    A single layer of adaptive Leaky Integrate-and-Fire neurons without
    layer-wise recurrent connections (adLIF).

    Arguments
    ---------
    input_size : int
        Number of features in the input tensors.
    hidden_size : int
        Number of output neurons.
    batch_size : int
        Batch size of the input tensors.
    threshold : float
        Value of spiking threshold (fixed)
    dropout : float
        Dropout factor (must be between 0 and 1).
    normalization : str
        Type of normalization. Every string different from 'batchnorm'
        and 'layernorm' will result in no normalization.
    use_bias : bool
        If True, additional trainable bias is used with feedforward weights.
    bidirectional : bool
        If True, a bidirectional model that scans the sequence both directions
        is used, which doubles the size of feedforward matrices in layers l>0.
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        batch_size,
        threshold=1.0,
        dropout=0.0,
        normalization="batchnorm",
        use_bias=False,
        bidirectional=False,
        extra_features=None
    ):
        super().__init__()

        # Fixed parameters
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.batch_size = batch_size
        self.threshold = threshold
        self.dropout = dropout
        self.normalization = normalization
        self.use_bias = use_bias
        self.bidirectional = bidirectional
        self.batch_size = self.batch_size * (1 + self.bidirectional)
        self.dt = 1.0
        self.tau_u_lim = [5, 25]
        self.tau_w_lim = [60, 300]
        self.a_lim = [0.0, 1.0]
        self.b_lim = [0.0, 2.0]
        self.q = 120

        # Trainable parameters
        self.W = nn.Linear(self.input_size, self.hidden_size, bias=use_bias)
        self.V = nn.Parameter(torch.empty(self.hidden_size, self.hidden_size))
        self.alpha = nn.Parameter(torch.Tensor(self.hidden_size))
        self.beta = nn.Parameter(torch.Tensor(self.hidden_size))
        self.a = nn.Parameter(torch.Tensor(self.hidden_size))
        self.b = nn.Parameter(torch.Tensor(self.hidden_size))
        self.theta = nn.Parameter(torch.Tensor(self.hidden_size))
        
        nn.init.uniform_(self.theta)
        nn.init.uniform_(self.a, self.a_lim[0], self.a_lim[1])
        nn.init.uniform_(self.b, self.b_lim[0], self.b_lim[1])
        nn.init.orthogonal_(self.V)

        # Initialize normalization
        self.normalize = False
        if normalization == "batchnorm":
            self.norm = nn.BatchNorm1d(self.hidden_size, momentum=0.05)
            self.normalize = True
        elif normalization == "layernorm":
            self.norm = nn.LayerNorm(self.hidden_size)
            self.normalize = True

        # Initialize dropout
        self.drop = nn.Dropout(p=dropout)

        if extra_features['rst_detach']:
            self.rst_detach = True
        else:
            self.rst_detach = False

    def forward(self, x):

        # Concatenate flipped sequence on batch dim
        if self.bidirectional:
            x_flip = x.flip(1)
            x = torch.cat([x, x_flip], dim=0)

        # Change batch size if needed
        if self.batch_size != x.shape[0]:
            self.batch_size = x.shape[0]

        # Feed-forward affine transformations (all steps in parallel)
        Wx = self.W(x)

        # Apply normalization
        if self.normalize:
            _Wx = self.norm(Wx.reshape(Wx.shape[0] * Wx.shape[1], Wx.shape[2]))
            Wx = _Wx.reshape(Wx.shape[0], Wx.shape[1], Wx.shape[2])

        # Compute spikes via neuron dynamics
        s = self._seadlif_cell(Wx)

        # Concatenate forward and backward sequences on feat dim
        if self.bidirectional:
            s_f, s_b = s.chunk(2, dim=0)
            s_b = s_b.flip(1)
            s = torch.cat([s_f, s_b], dim=2)

        # Apply dropout
        s = self.drop(s)

        return s

    def SLAYER(self, x, alpha=5, c=0.4):
        return c * alpha / (2 * torch.exp(x.abs() * alpha))

    def _seadlif_cell(self, Wx):

        # Initializations
        device = Wx.device
        utm1 = torch.rand(Wx.shape[0], Wx.shape[2]).to(device)
        ut = torch.zeros(Wx.shape[0], Wx.shape[2]).to(device)
        wt = torch.rand(Wx.shape[0], Wx.shape[2]).to(device)
        st = torch.rand(Wx.shape[0], Wx.shape[2]).to(device)
        s = []

        # Bound values of the neuron parameters to plausible ranges
        tau_u = self.tau_u_lim[0] + self.theta * (self.tau_u_lim[1]- self.tau_u_lim[0])
        tau_w = self.tau_w_lim[0] + self.theta * (self.tau_w_lim[1]- self.tau_w_lim[0])
        alpha = torch.exp(-self.dt / tau_u)
        beta = torch.exp(-self.dt / tau_w)
        a = torch.clamp(self.a, min=self.a_lim[0], max=self.a_lim[1])
        b = torch.clamp(self.b, min=self.b_lim[0], max=self.b_lim[1])

        # Loop over time axis
        for t in range(Wx.shape[1]):

            # Compute potential (adLIF)
            
            ut = alpha * utm1 + (1 - alpha)* (Wx[:, t, :] + F.linear(st, self.V, None) - wt)
            u_thr = ut - self.threshold
            st = torch.heaviside(u_thr, torch.as_tensor(0.0).type(u_thr.dtype)).detach() + (u_thr - u_thr.detach()) * self.SLAYER(u_thr).detach()
            ut = ut * (1 - st.detach())
            
            wt = beta * wt + (1 - beta)* (a * ut + b * st) * self.q


            s.append(st)

        return torch.stack(s, dim=1)

class adLIFclampLayer(nn.Module):
    """
    A single layer of adaptive Leaky Integrate-and-Fire neurons without
    layer-wise recurrent connections (adLIF).

    Arguments
    ---------
    input_size : int
        Number of features in the input tensors.
    hidden_size : int
        Number of output neurons.
    batch_size : int
        Batch size of the input tensors.
    threshold : float
        Value of spiking threshold (fixed)
    dropout : float
        Dropout factor (must be between 0 and 1).
    normalization : str
        Type of normalization. Every string different from 'batchnorm'
        and 'layernorm' will result in no normalization.
    use_bias : bool
        If True, additional trainable bias is used with feedforward weights.
    bidirectional : bool
        If True, a bidirectional model that scans the sequence both directions
        is used, which doubles the size of feedforward matrices in layers l>0.
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        batch_size,
        threshold=1.0,
        dropout=0.0,
        normalization="batchnorm",
        use_bias=False,
        bidirectional=False,
        extra_features=None
    ):
        super().__init__()

        # Fixed parameters
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.batch_size = batch_size
        self.threshold = threshold
        self.dropout = dropout
        self.normalization = normalization
        self.use_bias = use_bias
        self.bidirectional = bidirectional
        self.batch_size = self.batch_size * (1 + self.bidirectional)
        self.alpha_lim = [np.exp(-1 / 5), np.exp(-1 / 25)]
        self.beta_lim = [np.exp(-1 / 30), np.exp(-1 / 120)]
        self.a_lim = [-1.0, 1.0]
        self.b_lim = [0.0, 2.0]
        self.spike_fct = SpikeFunctionBoxcar.apply

        # Trainable parameters
        self.W = nn.Linear(self.input_size, self.hidden_size, bias=use_bias)
        self.alpha = nn.Parameter(torch.Tensor(self.hidden_size))
        self.beta = nn.Parameter(torch.Tensor(self.hidden_size))
        self.a = nn.Parameter(torch.Tensor(self.hidden_size))
        self.b = nn.Parameter(torch.Tensor(self.hidden_size))

        nn.init.uniform_(self.alpha, self.alpha_lim[0], self.alpha_lim[1])
        nn.init.uniform_(self.beta, self.beta_lim[0], self.beta_lim[1])
        nn.init.uniform_(self.a, self.a_lim[0], self.a_lim[1])
        nn.init.uniform_(self.b, self.b_lim[0], self.b_lim[1])

        # Initialize normalization
        self.normalize = False
        if normalization == "batchnorm":
            self.norm = nn.BatchNorm1d(self.hidden_size, momentum=0.05)
            self.normalize = True
        elif normalization == "layernorm":
            self.norm = nn.LayerNorm(self.hidden_size)
            self.normalize = True

        # Initialize dropout
        self.drop = nn.Dropout(p=dropout)

        if extra_features['rst_detach']:
            self.rst_detach = True
        else:
            self.rst_detach = False

    def forward(self, x):

        # Concatenate flipped sequence on batch dim
        if self.bidirectional:
            x_flip = x.flip(1)
            x = torch.cat([x, x_flip], dim=0)

        # Change batch size if needed
        if self.batch_size != x.shape[0]:
            self.batch_size = x.shape[0]

        self.alpha.data.clamp_(self.alpha_lim[0], self.alpha_lim[1])
        self.beta.data.clamp_(self.beta_lim[0], self.beta_lim[1])
        self.a.data.clamp_(self.a_lim[0], self.a_lim[1])
        self.b.data.clamp_(self.b_lim[0], self.b_lim[1])

        # Feed-forward affine transformations (all steps in parallel)
        Wx = self.W(x)

        # Apply normalization
        if self.normalize:
            _Wx = self.norm(Wx.reshape(Wx.shape[0] * Wx.shape[1], Wx.shape[2]))
            Wx = _Wx.reshape(Wx.shape[0], Wx.shape[1], Wx.shape[2])

        # Compute spikes via neuron dynamics
        s = self._adlif_cell(Wx)

        # Concatenate forward and backward sequences on feat dim
        if self.bidirectional:
            s_f, s_b = s.chunk(2, dim=0)
            s_b = s_b.flip(1)
            s = torch.cat([s_f, s_b], dim=2)

        # Apply dropout
        s = self.drop(s)

        return s

    def _adlif_cell(self, Wx):

        # Initializations
        device = Wx.device
        ut = torch.rand(Wx.shape[0], Wx.shape[2]).to(device)
        wt = torch.rand(Wx.shape[0], Wx.shape[2]).to(device)
        st = torch.rand(Wx.shape[0], Wx.shape[2]).to(device)
        s = []

        # Bound values of the neuron parameters to plausible ranges
        alpha = torch.clamp(self.alpha, min=self.alpha_lim[0], max=self.alpha_lim[1])
        beta = torch.clamp(self.beta, min=self.beta_lim[0], max=self.beta_lim[1])
        a = torch.clamp(self.a, min=self.a_lim[0], max=self.a_lim[1])
        b = torch.clamp(self.b, min=self.b_lim[0], max=self.b_lim[1])

        # Loop over time axis
        for t in range(Wx.shape[1]):

            if self.rst_detach:
                reset = st.clone().detach()
            else:
                reset = st

            # Compute potential (adLIF)
            wt = beta * wt + a * ut + b * reset
            ut = alpha * (ut - reset) + (1 - alpha)* (Wx[:, t, :] - wt)

            # Compute spikes with surrogate gradient
            st = self.spike_fct(ut - self.threshold)
            s.append(st)

        return torch.stack(s, dim=1)
    


class adLIFnoClampLayer(nn.Module):
    """
    A single layer of adaptive Leaky Integrate-and-Fire neurons without
    layer-wise recurrent connections (adLIF).

    Arguments
    ---------
    input_size : int
        Number of features in the input tensors.
    hidden_size : int
        Number of output neurons.
    batch_size : int
        Batch size of the input tensors.
    threshold : float
        Value of spiking threshold (fixed)
    dropout : float
        Dropout factor (must be between 0 and 1).
    normalization : str
        Type of normalization. Every string different from 'batchnorm'
        and 'layernorm' will result in no normalization.
    use_bias : bool
        If True, additional trainable bias is used with feedforward weights.
    bidirectional : bool
        If True, a bidirectional model that scans the sequence both directions
        is used, which doubles the size of feedforward matrices in layers l>0.
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        batch_size,
        threshold=1.0,
        dropout=0.0,
        normalization="batchnorm",
        use_bias=False,
        bidirectional=False,
        extra_features=None
    ):
        super().__init__()

        # Fixed parameters
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.batch_size = batch_size
        self.threshold = threshold
        self.dropout = dropout
        self.normalization = normalization
        self.use_bias = use_bias
        self.bidirectional = bidirectional
        self.batch_size = self.batch_size * (1 + self.bidirectional)
        self.alpha_lim = [np.exp(-1 / 5), np.exp(-1 / 25)]
        self.beta_lim = [np.exp(-1 / 30), np.exp(-1 / 120)]
        self.a_lim = [-1.0, 1.0]
        self.b_lim = [0.0, 2.0]
        self.spike_fct = SpikeFunctionBoxcar.apply

        # Trainable parameters
        self.W = nn.Linear(self.input_size, self.hidden_size, bias=use_bias)
        self.alpha = nn.Parameter(torch.Tensor(self.hidden_size))
        self.beta = nn.Parameter(torch.Tensor(self.hidden_size))
        self.a = nn.Parameter(torch.Tensor(self.hidden_size))
        self.b = nn.Parameter(torch.Tensor(self.hidden_size))

        nn.init.uniform_(self.alpha, self.alpha_lim[0], self.alpha_lim[1])
        nn.init.uniform_(self.beta, self.beta_lim[0], self.beta_lim[1])
        nn.init.uniform_(self.a, self.a_lim[0], self.a_lim[1])
        nn.init.uniform_(self.b, self.b_lim[0], self.b_lim[1])

        # Initialize normalization
        self.normalize = False
        if normalization == "batchnorm":
            self.norm = nn.BatchNorm1d(self.hidden_size, momentum=0.05)
            self.normalize = True
        elif normalization == "layernorm":
            self.norm = nn.LayerNorm(self.hidden_size)
            self.normalize = True

        # Initialize dropout
        self.drop = nn.Dropout(p=dropout)

        if extra_features['rst_detach']:
            self.rst_detach = True
        else:
            self.rst_detach = False

    def forward(self, x):

        # Concatenate flipped sequence on batch dim
        if self.bidirectional:
            x_flip = x.flip(1)
            x = torch.cat([x, x_flip], dim=0)

        # Change batch size if needed
        if self.batch_size != x.shape[0]:
            self.batch_size = x.shape[0]

        # Feed-forward affine transformations (all steps in parallel)
        Wx = self.W(x)

        # Apply normalization
        if self.normalize:
            _Wx = self.norm(Wx.reshape(Wx.shape[0] * Wx.shape[1], Wx.shape[2]))
            Wx = _Wx.reshape(Wx.shape[0], Wx.shape[1], Wx.shape[2])

        # Compute spikes via neuron dynamics
        s = self._adlif_cell(Wx)

        # Concatenate forward and backward sequences on feat dim
        if self.bidirectional:
            s_f, s_b = s.chunk(2, dim=0)
            s_b = s_b.flip(1)
            s = torch.cat([s_f, s_b], dim=2)

        # Apply dropout
        s = self.drop(s)

        return s

    def _adlif_cell(self, Wx):

        # Initializations
        device = Wx.device
        ut = torch.rand(Wx.shape[0], Wx.shape[2]).to(device)
        wt = torch.rand(Wx.shape[0], Wx.shape[2]).to(device)
        st = torch.rand(Wx.shape[0], Wx.shape[2]).to(device)
        s = []

        # Bound values of the neuron parameters to plausible ranges
        alpha = self.alpha
        beta = self.beta 
        a = self.a 
        b = self.b 

        # Loop over time axis
        for t in range(Wx.shape[1]):

            if self.rst_detach:
                reset = st.clone().detach()
            else:
                reset = st

            # Compute potential (adLIF)
            wt = beta * wt + a * ut + b * reset
            ut = alpha * (ut - reset) + (1 - alpha)* (Wx[:, t, :] - wt)

            # Compute spikes with surrogate gradient
            st = self.spike_fct(ut - self.threshold)
            s.append(st)

        return torch.stack(s, dim=1)
    

class LIFcomplexLayer(nn.Module):
    """
    A single layer of Leaky Integrate-and-Fire neurons without layer-wise
    recurrent connections (LIF).

    Arguments
    ---------
    input_size : int
        Number of features in the input tensors.
    hidden_size : int
        Number of output neurons.
    batch_size : int
        Batch size of the input tensors.
    threshold : float
        Value of spiking threshold (fixed)
    dropout : float
        Dropout factor (must be between 0 and 1).
    normalization : str
        Type of normalization. Every string different from 'batchnorm'
        and 'layernorm' will result in no normalization.
    use_bias : bool
        If True, additional trainable bias is used with feedforward weights.
    bidirectional : bool
        If True, a bidirectional model that scans the sequence both directions
        is used, which doubles the size of feedforward matrices in layers l>0.
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        batch_size,
        threshold=1.0,
        dropout=0.0,
        normalization="batchnorm",
        use_bias=False,
        bidirectional=False,
        extra_features=None
    ):
        super().__init__()
        self.device = torch.device("cuda")

        # Fixed parameters
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.batch_size = batch_size
        self.threshold = extra_features["thr"]
        alpha_imag = extra_features["alpha_imag"]
        alpha_range = extra_features["alpha_range"]
        self.dropout = dropout
        self.normalization = normalization
        self.use_bias = use_bias
        self.bidirectional = bidirectional
        self.batch_size = self.batch_size * (1 + self.bidirectional)

        self.bRand = extra_features["bRand"]
        
        if extra_features['superspike']:
            self.spike_fct = SpikeFunctionSuperSpike.apply
        elif extra_features['slayer']:
            self.spike_fct = SpikeFunctionSLAYER.apply
        else:
            self.spike_fct = SpikeFunctionBoxcar.apply

        # Trainable parameters
        
        if extra_features['xavier_init']:
            init.xavier_uniform_(self.W.weight)
        if extra_features.get("alpha", False)!=False:
            alph = extra_features.get("alpha", False)
        else:
            alph = 0.5

        if extra_features.get("alphaRe_rand", False)  =="Rand":
            alph =  alph * torch.rand(self.hidden_size)
        elif extra_features.get("alphaRe_rand", False)  =="RandN":
            alph =  alph * torch.randn(self.hidden_size)
        else:
            alph = alph * torch.ones(self.hidden_size)
        log_log_alpha = torch.log(alph)
        #self.log_log_alpha_lim = [math.log(1 / 200), math.log(1 / 5)]
        dt_min = extra_features["dt_min"]
        dt_max = extra_features["dt_max"]
        self.c_discr = extra_features["c_discr"]
        self.c_param = extra_features["c_param"]
        
            
        
        log_dt = torch.rand(self.hidden_size)*(
            math.log(dt_max) - math.log(dt_min)
        ) + math.log(dt_min)
        #nn.init.uniform_(log_log_alpha, self.log_log_alpha_lim[0], self.log_log_alpha_lim[1])
        
        if alpha_imag == 3.14:
            alpha_img =  math.pi  ##* torch.arange(self.hidden_size)  
        elif alpha_imag == 1.57:
            alpha_img =  math.pi / 2 
        elif alpha_imag == 0.785:
            alpha_img =  math.pi / 4 
        elif alpha_imag == 0.523:
            alpha_img =  math.pi / 6 
        elif alpha_imag == 2.09:
            alpha_img =  math.pi / 1.5 
        elif alpha_imag == 6.28:
            alpha_img =  math.pi * 2 
        else:
            alpha_img = alpha_imag 

        if alpha_range:
            alpha_img =  alpha_img /  torch.arange(self.hidden_size)  
        else:
            alpha_img =  alpha_img *  torch.ones(self.hidden_size)
        
        if extra_features.get("alpha_rand", False)  =="Rand":
            alpha_img =  alpha_img * torch.rand(self.hidden_size)
        elif extra_features.get("alpha_rand", False)  =="RandN":
            alpha_img =  alpha_img * torch.randn(self.hidden_size)
           

        self.register("log_log_alpha", log_log_alpha, lr=0.01)
        self.register("log_dt", log_dt, lr=0.01)
        self.register("alpha_img", alpha_img, lr=0.01)

        if self.bRand =="RandN":
            self.b = nn.Parameter(torch.randn(self.hidden_size))
        elif self.bRand =="Rand":
            self.b = nn.Parameter(torch.rand(self.hidden_size))
        if self.c_param:
            self.c = nn.Parameter(torch.randn(self.hidden_size,  dtype=torch.cfloat))
        else:
            self.c = torch.ones(self.hidden_size)
        if extra_features['reset'] == 'no_reset':
            self.reset_factor = 0
        elif extra_features['reset'] == 'half_reset':
            self.reset_factor = 0.5
        else:
            self.reset_factor = 1.0
        

        if extra_features['rst_detach']:
            self.rst_detach = True
        else:
            self.rst_detach = False

        # Initialize normalinzation
        self.normalize = False
        if normalization == "batchnorm":
            self.norm = nn.BatchNorm1d(self.hidden_size, momentum=0.05)
            self.normalize = True
        elif normalization == "layernorm":
            self.norm = nn.LayerNorm(self.hidden_size)
            self.normalize = True

        # Initialize dropout
        self.s_GLU = extra_features['s_GLU']
        self.drop = nn.Dropout(p=dropout)
        if self.s_GLU:
            self.output_linear = nn.Sequential(
                nn.Conv1d(self.input_size, 2*self.hidden_size, kernel_size=1),
                nn.GLU(dim=-2),
            )
        else:
            self.W = nn.Linear(self.input_size, self.hidden_size, bias=use_bias)
        self.to(self.device)
        



    def forward(self, x):

        # Concatenate flipped sequence on batch dim
        if self.bidirectional:
            x_flip = x.flip(1)
            x = torch.cat([x, x_flip], dim=0)

        # Change batch size if needed
        if self.batch_size != x.shape[0]:
            self.batch_size = x.shape[0]

        # Feed-forward affine transformations (all steps in parallel)
        if not self.s_GLU:
            Wx = self.W(x)
        else:
            Wx = self.output_linear(x.reshape(x.shape[0], x.shape[2], x.shape[1]))
            Wx = Wx.reshape(x.shape[0], x.shape[1], self.hidden_size)


        #Wx = self.output_linear(Wx.reshape(Wx.shape[0], Wx.shape[2], Wx.shape[1])).reshape(Wx.shape[0], Wx.shape[1], Wx.shape[2])

        # Apply normalization
        if self.normalize:
            _Wx = self.norm(Wx.reshape(Wx.shape[0] * Wx.shape[1], Wx.shape[2]))
            Wx = _Wx.reshape(Wx.shape[0], Wx.shape[1], Wx.shape[2])

        # Compute spikes via neuron dynamics
        s = self._lif_cell(Wx)
            

        # Concatenate forward and backward sequences on feat dim
        if self.bidirectional:
            s_f, s_b = s.chunk(2, dim=0)
            s_b = s_b.flip(1)
            s = torch.cat([s_f, s_b], dim=2)

        # Apply dropout
        s = self.drop(s)

        return s
    def register(self, name, tensor, lr=None):
        """Register a tensor with a configurable learning rate and 0 weight decay"""

        if lr == 0.0:
            self.register_buffer(name, tensor)
        else:
            self.register_parameter(name, nn.Parameter(tensor))

            optim = {"weight_decay": 0.0}
            if lr is not None: optim["lr"] = lr
            setattr(getattr(self, name), "_optim", optim)


    def _lif_cell(self, Wx):

        # Initializations
        device = Wx.device
        ut = torch.rand(Wx.shape[0], Wx.shape[2], dtype=torch.cfloat).to(device)
        st = torch.rand(Wx.shape[0], Wx.shape[2]).to(device)
        s = []

        # Bound values of the neuron parameters to plausible ranges
        #log_log__alpha = torch.clamp(self.log_log_alpha, min=self.log_log_alpha_lim[0], max=self.log_log_alpha_lim[1])
        alpha = torch.exp((-torch.exp(self.log_log_alpha)+1j*self.alpha_img)*torch.exp(self.log_dt))
        if self.c_discr:
            C = (self.c.to(device) * (alpha-1.) / (-torch.exp(self.log_log_alpha)+1j*self.alpha_img)).to(device)
        else:
            C = (self.c).to(device)
        # Loop over time axis
        for t in range(Wx.shape[1]):

            if self.rst_detach:
                reset = st.clone().detach()
            else: 
                reset = st

            # Compute membrane potential (LIF)
            ut = alpha * (ut - self.reset_factor*reset) + self.b * Wx[:, t, :]

            # Compute spikes with surrogate gradient
            st = self.spike_fct((2*C*ut).real - self.threshold)
            s.append(st)

        return torch.stack(s, dim=1)

class LIFrealcomplexLayer(nn.Module):
    """
    A single layer of Leaky Integrate-and-Fire neurons without layer-wise
    recurrent connections (LIF).

    Arguments
    ---------
    input_size : int
        Number of features in the input tensors.
    hidden_size : int
        Number of output neurons.
    batch_size : int
        Batch size of the input tensors.
    threshold : float
        Value of spiking threshold (fixed)
    dropout : float
        Dropout factor (must be between 0 and 1).
    normalization : str
        Type of normalization. Every string different from 'batchnorm'
        and 'layernorm' will result in no normalization.
    use_bias : bool
        If True, additional trainable bias is used with feedforward weights.
    bidirectional : bool
        If True, a bidirectional model that scans the sequence both directions
        is used, which doubles the size of feedforward matrices in layers l>0.
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        batch_size,
        threshold=1.0,
        dropout=0.0,
        normalization="batchnorm",
        use_bias=False,
        bidirectional=False,
        extra_features=None
    ):
        super().__init__()

        # Fixed parameters
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.batch_size = batch_size
        self.threshold = threshold
        self.dropout = dropout
        self.normalization = normalization
        self.use_bias = use_bias
        self.bidirectional = bidirectional
        self.batch_size = self.batch_size * (1 + self.bidirectional)
        
        self.spike_fct = SpikeFunctionBoxcar.apply

        # Trainable parameters
        self.W = nn.Linear(self.input_size, self.hidden_size, bias=use_bias)
        log_log_alpha = torch.log(0.5 * torch.ones(self.hidden_size))
        #self.log_log_alpha_lim = [math.log(1 / 200), math.log(1 / 5)]
        dt_min = extra_features["dt_min"]
        dt_max = extra_features["dt_max"]
        log_dt = torch.rand(self.hidden_size)*(
            math.log(dt_max) - math.log(dt_min)
        ) + math.log(dt_min)
        #nn.init.uniform_(log_log_alpha, self.log_log_alpha_lim[0], self.log_log_alpha_lim[1])
        alpha_img =  math.pi * torch.ones(self.hidden_size) # torch.arange(self.hidden_size)

        self.register("log_log_alpha", log_log_alpha, lr=0.01)
        self.register("log_dt", log_dt, lr=0.01)
        self.register("alpha_img", alpha_img, lr=0.01)

        self.b = nn.Parameter(torch.rand(self.hidden_size))

        if extra_features['half_reset']:
            self.reset_factor = 0.5
        else:
            self.reset_factor = 1.0

        if extra_features['rst_detach']:
            self.rst_detach = True
        else:
            self.rst_detach = False

        # Initialize normalinzation
        self.normalize = False
        if normalization == "batchnorm":
            self.norm = nn.BatchNorm1d(self.hidden_size, momentum=0.05)
            self.normalize = True
        elif normalization == "layernorm":
            self.norm = nn.LayerNorm(self.hidden_size)
            self.normalize = True

        # Initialize dropout
        self.drop = nn.Dropout(p=dropout)



    def forward(self, x):

        # Concatenate flipped sequence on batch dim
        if self.bidirectional:
            x_flip = x.flip(1)
            x = torch.cat([x, x_flip], dim=0)

        # Change batch size if needed
        if self.batch_size != x.shape[0]:
            self.batch_size = x.shape[0]

        # Feed-forward affine transformations (all steps in parallel)
        Wx = self.W(x)

        #Wx = self.output_linear(Wx.reshape(Wx.shape[0], Wx.shape[2], Wx.shape[1])).reshape(Wx.shape[0], Wx.shape[1], Wx.shape[2])

        # Apply normalization
        if self.normalize:
            _Wx = self.norm(Wx.reshape(Wx.shape[0] * Wx.shape[1], Wx.shape[2]))
            Wx = _Wx.reshape(Wx.shape[0], Wx.shape[1], Wx.shape[2])

        # Compute spikes via neuron dynamics
        s = self._lif_cell(Wx)

        # Concatenate forward and backward sequences on feat dim
        if self.bidirectional:
            s_f, s_b = s.chunk(2, dim=0)
            s_b = s_b.flip(1)
            s = torch.cat([s_f, s_b], dim=2)

        # Apply dropout
        s = self.drop(s)

        return s
    def register(self, name, tensor, lr=None):
        """Register a tensor with a configurable learning rate and 0 weight decay"""

        if lr == 0.0:
            self.register_buffer(name, tensor)
        else:
            self.register_parameter(name, nn.Parameter(tensor))

            optim = {"weight_decay": 0.0}
            if lr is not None: optim["lr"] = lr
            setattr(getattr(self, name), "_optim", optim)


    def _lif_cell(self, Wx):

        # Initializations
        device = Wx.device
        ut = torch.rand(Wx.shape[0], Wx.shape[2]).to(device)
        wt = torch.rand(Wx.shape[0], Wx.shape[2]).to(device)
        st = torch.rand(Wx.shape[0], Wx.shape[2]).to(device)
        s = []

        # Bound values of the neuron parameters to plausible ranges
        #log_log__alpha = torch.clamp(self.log_log_alpha, min=self.log_log_alpha_lim[0], max=self.log_log_alpha_lim[1])
        dt = torch.exp(self.log_dt)
        real_part = -torch.exp(self.log_log_alpha) * dt
        imaginary_part = self.alpha_img * dt

        # Compute the separate components
        exp_real = torch.exp(real_part)
        cos_imag = torch.cos(imaginary_part)
        sin_imag = torch.sin(imaginary_part)

        alpha_real = exp_real * cos_imag
        alpha_imag = exp_real * sin_imag

        # Loop over time axis
        for t in range(Wx.shape[1]):

            if self.rst_detach:
                reset = st.clone().detach()
            else: 
                reset = st

            # Compute membrane potential (LIF)
            wt = alpha_real * wt + alpha_imag * (ut - self.reset_factor*reset)
            ut = alpha_real * (ut - self.reset_factor*reset) - alpha_imag*wt + self.b * Wx[:, t, :]

            # Compute spikes with surrogate gradient
            st = self.spike_fct(2*ut - self.threshold)
            s.append(st)

        return torch.stack(s, dim=1)

class ReLULIFcomplexLayer(nn.Module):
    """
    A single layer of Leaky Integrate-and-Fire neurons without layer-wise
    recurrent connections (LIF).

    Arguments
    ---------
    input_size : int
        Number of features in the input tensors.
    hidden_size : int
        Number of output neurons.
    batch_size : int
        Batch size of the input tensors.
    threshold : float
        Value of spiking threshold (fixed)
    dropout : float
        Dropout factor (must be between 0 and 1).
    normalization : str
        Type of normalization. Every string different from 'batchnorm'
        and 'layernorm' will result in no normalization.
    use_bias : bool
        If True, additional trainable bias is used with feedforward weights.
    bidirectional : bool
        If True, a bidirectional model that scans the sequence both directions
        is used, which doubles the size of feedforward matrices in layers l>0.
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        batch_size,
        threshold=1.0,
        dropout=0.0,
        normalization="batchnorm",
        use_bias=False,
        bidirectional=False,
        extra_features=None
    ):
        super().__init__()

        # Fixed parameters
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.batch_size = batch_size
        self.threshold = threshold
        self.dropout = dropout
        self.normalization = normalization
        self.use_bias = use_bias
        self.bidirectional = bidirectional
        self.batch_size = self.batch_size * (1 + self.bidirectional)
        
        self.spike_fct = SpikeFunctionBoxcar.apply

        # Trainable parameters
        self.W = nn.Linear(self.input_size, self.hidden_size, bias=use_bias)
        log_log_alpha = torch.log(0.5 * torch.ones(self.hidden_size))
        #self.log_log_alpha_lim = [math.log(1 / 200), math.log(1 / 5)]
        dt_min = extra_features["dt_min"]
        dt_max = extra_features["dt_max"]
        log_dt = torch.rand(self.hidden_size)*(
            math.log(dt_max) - math.log(dt_min)
        ) + math.log(dt_min)
        #nn.init.uniform_(log_log_alpha, self.log_log_alpha_lim[0], self.log_log_alpha_lim[1])
        alpha_img =  math.pi * torch.ones(self.hidden_size) # torch.arange(self.hidden_size)

        self.register("log_log_alpha", log_log_alpha, lr=0.01)
        self.register("log_dt", log_dt, lr=0.01)
        self.register("alpha_img", alpha_img, lr=0.01)

        self.b = nn.Parameter(torch.rand(self.hidden_size))

        if extra_features['half_reset']:
            self.reset_factor = 0.5
        else:
            self.reset_factor = 1.0

        if extra_features['rst_detach']:
            self.rst_detach = True
        else:
            self.rst_detach = False

        # Initialize normalinzation
        self.normalize = False
        if normalization == "batchnorm":
            self.norm = nn.BatchNorm1d(self.hidden_size, momentum=0.05)
            self.normalize = True
        elif normalization == "layernorm":
            self.norm = nn.LayerNorm(self.hidden_size)
            self.normalize = True

        # Initialize dropout
        self.drop = nn.Dropout(p=dropout)

        self.shifted_relu = extra_features['shifted_relu']

    def forward(self, x):

        # Concatenate flipped sequence on batch dim
        if self.bidirectional:
            x_flip = x.flip(1)
            x = torch.cat([x, x_flip], dim=0)

        # Change batch size if needed
        if self.batch_size != x.shape[0]:
            self.batch_size = x.shape[0]

        # Feed-forward affine transformations (all steps in parallel)
        Wx = self.W(x)

        #Wx = self.output_linear(Wx.reshape(Wx.shape[0], Wx.shape[2], Wx.shape[1])).reshape(Wx.shape[0], Wx.shape[1], Wx.shape[2])

        # Apply normalization
        if self.normalize:
            _Wx = self.norm(Wx.reshape(Wx.shape[0] * Wx.shape[1], Wx.shape[2]))
            Wx = _Wx.reshape(Wx.shape[0], Wx.shape[1], Wx.shape[2])

        # Compute spikes via neuron dynamics
        s = self._lif_cell(Wx)

        # Concatenate forward and backward sequences on feat dim
        if self.bidirectional:
            s_f, s_b = s.chunk(2, dim=0)
            s_b = s_b.flip(1)
            s = torch.cat([s_f, s_b], dim=2)

        # Apply dropout
        s = self.drop(s)

        return s
    def register(self, name, tensor, lr=None):
        """Register a tensor with a configurable learning rate and 0 weight decay"""

        if lr == 0.0:
            self.register_buffer(name, tensor)
        else:
            self.register_parameter(name, nn.Parameter(tensor))

            optim = {"weight_decay": 0.0}
            if lr is not None: optim["lr"] = lr
            setattr(getattr(self, name), "_optim", optim)


    def _lif_cell(self, Wx):

        # Initializations
        device = Wx.device
        ut = torch.rand(Wx.shape[0], Wx.shape[2], dtype=torch.cfloat).to(device)
        st = torch.rand(Wx.shape[0], Wx.shape[2]).to(device)
        s = []

        # Bound values of the neuron parameters to plausible ranges
        #log_log__alpha = torch.clamp(self.log_log_alpha, min=self.log_log_alpha_lim[0], max=self.log_log_alpha_lim[1])
        alpha = torch.exp((-torch.exp(self.log_log_alpha)+1j*self.alpha_img)*torch.exp(self.log_dt))
        # Loop over time axis
        for t in range(Wx.shape[1]):

            # Compute membrane potential (LIF)
            ut = alpha * ut + self.b * Wx[:, t, :]

            # Compute spikes with surrogate gradient
            if self.shifted_relu:
                st = F.relu(2*ut.real - self.threshold)
            else:
                st = F.relu(2*ut.real)
            s.append(st)

        return torch.stack(s, dim=1)

class RLIFcomplexLayer(nn.Module):
    """
    A single layer of Leaky Integrate-and-Fire neurons without layer-wise
    recurrent connections (LIF).

    Arguments
    ---------
    input_size : int
        Number of features in the input tensors.
    hidden_size : int
        Number of output neurons.
    batch_size : int
        Batch size of the input tensors.
    threshold : float
        Value of spiking threshold (fixed)
    dropout : float
        Dropout factor (must be between 0 and 1).
    normalization : str
        Type of normalization. Every string different from 'batchnorm'
        and 'layernorm' will result in no normalization.
    use_bias : bool
        If True, additional trainable bias is used with feedforward weights.
    bidirectional : bool
        If True, a bidirectional model that scans the sequence both directions
        is used, which doubles the size of feedforward matrices in layers l>0.
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        batch_size,
        threshold=1.0,
        dropout=0.0,
        normalization="batchnorm",
        use_bias=False,
        bidirectional=False,
        extra_features=None
    ):
        super().__init__()

        # Fixed parameters
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.batch_size = batch_size
        self.threshold = threshold
        self.dropout = dropout
        self.normalization = normalization
        self.use_bias = use_bias
        self.bidirectional = bidirectional
        self.batch_size = self.batch_size * (1 + self.bidirectional)
        
        self.spike_fct = SpikeFunctionBoxcar.apply

        # Trainable parameters
        self.W = nn.Linear(self.input_size, self.hidden_size, bias=use_bias)
        self.V = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        log_log_alpha = torch.log(0.5 * torch.ones(self.hidden_size))
        #self.log_log_alpha_lim = [math.log(1 / 200), math.log(1 / 5)]
        dt_min = extra_features["dt_min"]
        dt_max = extra_features["dt_max"]
        log_dt = torch.rand(self.hidden_size)*(
            math.log(dt_max) - math.log(dt_min)
        ) + math.log(dt_min)
        #nn.init.uniform_(log_log_alpha, self.log_log_alpha_lim[0], self.log_log_alpha_lim[1])
        alpha_img =  math.pi * torch.ones(self.hidden_size) # torch.arange(self.hidden_size)

        self.register("log_log_alpha", log_log_alpha, lr=0.01)
        self.register("log_dt", log_dt, lr=0.01)
        self.register("alpha_img", alpha_img, lr=0.01)

        self.b = nn.Parameter(torch.rand(self.hidden_size))

        if extra_features['half_reset']:
            self.reset_factor = 0.5
        else:
            self.reset_factor = 1.0

        if extra_features['rst_detach']:
            self.rst_detach = True
        else:
            self.rst_detach = False


        # Initialize normalinzation
        self.normalize = False
        if normalization == "batchnorm":
            self.norm = nn.BatchNorm1d(self.hidden_size, momentum=0.05)
            self.normalize = True
        elif normalization == "layernorm":
            self.norm = nn.LayerNorm(self.hidden_size)
            self.normalize = True

        # Initialize dropout
        self.drop = nn.Dropout(p=dropout)

        self.output_linear = nn.Sequential(
            nn.Conv1d(self.hidden_size, 2*self.hidden_size, kernel_size=1),
            nn.GLU(dim=-2),
        )

    def forward(self, x):

        # Concatenate flipped sequence on batch dim
        if self.bidirectional:
            x_flip = x.flip(1)
            x = torch.cat([x, x_flip], dim=0)

        # Change batch size if needed
        if self.batch_size != x.shape[0]:
            self.batch_size = x.shape[0]

        # Feed-forward affine transformations (all steps in parallel)
        Wx = self.W(x)

        #Wx = self.output_linear(Wx.reshape(Wx.shape[0], Wx.shape[2], Wx.shape[1])).reshape(Wx.shape[0], Wx.shape[1], Wx.shape[2])

        # Apply normalization
        if self.normalize:
            _Wx = self.norm(Wx.reshape(Wx.shape[0] * Wx.shape[1], Wx.shape[2]))
            Wx = _Wx.reshape(Wx.shape[0], Wx.shape[1], Wx.shape[2])

        # Compute spikes via neuron dynamics
        s = self._lif_cell(Wx)

        # Concatenate forward and backward sequences on feat dim
        if self.bidirectional:
            s_f, s_b = s.chunk(2, dim=0)
            s_b = s_b.flip(1)
            s = torch.cat([s_f, s_b], dim=2)

        # Apply dropout
        s = self.drop(s)

        return s
    def register(self, name, tensor, lr=None):
        """Register a tensor with a configurable learning rate and 0 weight decay"""

        if lr == 0.0:
            self.register_buffer(name, tensor)
        else:
            self.register_parameter(name, nn.Parameter(tensor))

            optim = {"weight_decay": 0.0}
            if lr is not None: optim["lr"] = lr
            setattr(getattr(self, name), "_optim", optim)


    def _lif_cell(self, Wx):

        # Initializations
        device = Wx.device
        ut = torch.rand(Wx.shape[0], Wx.shape[2], dtype=torch.cfloat).to(device)
        st = torch.rand(Wx.shape[0], Wx.shape[2]).to(device)
        s = []

        V = self.V.weight.clone().fill_diagonal_(0)

        # Bound values of the neuron parameters to plausible ranges
        #log_log__alpha = torch.clamp(self.log_log_alpha, min=self.log_log_alpha_lim[0], max=self.log_log_alpha_lim[1])
        alpha = torch.exp((-torch.exp(self.log_log_alpha)+1j*self.alpha_img)*torch.exp(self.log_dt))
        # Loop over time axis
        for t in range(Wx.shape[1]):

            if self.rst_detach:
                reset = st.clone().detach()
            else: 
                reset = st

            # Compute membrane potential (LIF)
            ut = alpha * (ut - self.reset_factor*reset) + self.b * (Wx[:, t, :] + torch.matmul(st, V))

            # Compute spikes with surrogate gradient
            st = self.spike_fct(2*ut.real - self.threshold)
            s.append(st)

        return torch.stack(s, dim=1)

class RLIFcomplex1MinAlphaLayer(nn.Module):
    """
    A single layer of Leaky Integrate-and-Fire neurons without layer-wise
    recurrent connections (LIF).

    Arguments
    ---------
    input_size : int
        Number of features in the input tensors.
    hidden_size : int
        Number of output neurons.
    batch_size : int
        Batch size of the input tensors.
    threshold : float
        Value of spiking threshold (fixed)
    dropout : float
        Dropout factor (must be between 0 and 1).
    normalization : str
        Type of normalization. Every string different from 'batchnorm'
        and 'layernorm' will result in no normalization.
    use_bias : bool
        If True, additional trainable bias is used with feedforward weights.
    bidirectional : bool
        If True, a bidirectional model that scans the sequence both directions
        is used, which doubles the size of feedforward matrices in layers l>0.
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        batch_size,
        threshold=1.0,
        dropout=0.0,
        normalization="batchnorm",
        use_bias=False,
        bidirectional=False,
        extra_features=None
    ):
        super().__init__()

        # Fixed parameters
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.batch_size = batch_size
        self.threshold = threshold
        self.dropout = dropout
        self.normalization = normalization
        self.use_bias = use_bias
        self.bidirectional = bidirectional
        self.batch_size = self.batch_size * (1 + self.bidirectional)
        
        self.spike_fct = SpikeFunctionBoxcar.apply

        # Trainable parameters
        self.W = nn.Linear(self.input_size, self.hidden_size, bias=use_bias)
        self.V = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        log_log_alpha = torch.log(0.5 * torch.ones(self.hidden_size))
        #self.log_log_alpha_lim = [math.log(1 / 200), math.log(1 / 5)]
        dt_min = 0.01
        dt_max = 0.4
        log_dt = torch.rand(self.hidden_size)*(
            math.log(dt_max) - math.log(dt_min)
        ) + math.log(dt_min)
        #nn.init.uniform_(log_log_alpha, self.log_log_alpha_lim[0], self.log_log_alpha_lim[1])
        alpha_img =  math.pi * torch.ones(self.hidden_size) # torch.arange(self.hidden_size)

        self.register("log_log_alpha", log_log_alpha, lr=0.01)
        self.register("log_dt", log_dt, lr=0.01)
        self.register("alpha_img", alpha_img, lr=0.01)

        self.b = nn.Parameter(torch.rand(self.hidden_size))

        # Initialize normalinzation
        self.normalize = False
        if normalization == "batchnorm":
            self.norm = nn.BatchNorm1d(self.hidden_size, momentum=0.05)
            self.normalize = True
        elif normalization == "layernorm":
            self.norm = nn.LayerNorm(self.hidden_size)
            self.normalize = True

        # Initialize dropout
        self.drop = nn.Dropout(p=dropout)

    def forward(self, x):

        # Concatenate flipped sequence on batch dim
        if self.bidirectional:
            x_flip = x.flip(1)
            x = torch.cat([x, x_flip], dim=0)

        # Change batch size if needed
        if self.batch_size != x.shape[0]:
            self.batch_size = x.shape[0]

        # Feed-forward affine transformations (all steps in parallel)
        Wx = self.W(x)

        #Wx = self.output_linear(Wx.reshape(Wx.shape[0], Wx.shape[2], Wx.shape[1])).reshape(Wx.shape[0], Wx.shape[1], Wx.shape[2])

        # Apply normalization
        if self.normalize:
            _Wx = self.norm(Wx.reshape(Wx.shape[0] * Wx.shape[1], Wx.shape[2]))
            Wx = _Wx.reshape(Wx.shape[0], Wx.shape[1], Wx.shape[2])

        # Compute spikes via neuron dynamics
        s = self._lif_cell(Wx)

        # Concatenate forward and backward sequences on feat dim
        if self.bidirectional:
            s_f, s_b = s.chunk(2, dim=0)
            s_b = s_b.flip(1)
            s = torch.cat([s_f, s_b], dim=2)

        # Apply dropout
        s = self.drop(s)

        return s
    def register(self, name, tensor, lr=None):
        """Register a tensor with a configurable learning rate and 0 weight decay"""

        if lr == 0.0:
            self.register_buffer(name, tensor)
        else:
            self.register_parameter(name, nn.Parameter(tensor))

            optim = {"weight_decay": 0.0}
            if lr is not None: optim["lr"] = lr
            setattr(getattr(self, name), "_optim", optim)


    def _lif_cell(self, Wx):

        # Initializations
        device = Wx.device
        ut = torch.rand(Wx.shape[0], Wx.shape[2], dtype=torch.cfloat).to(device)
        st = torch.rand(Wx.shape[0], Wx.shape[2]).to(device)
        s = []

        V = self.V.weight.clone().fill_diagonal_(0)

        # Bound values of the neuron parameters to plausible ranges
        #log_log__alpha = torch.clamp(self.log_log_alpha, min=self.log_log_alpha_lim[0], max=self.log_log_alpha_lim[1])
        alpha = torch.exp((-torch.exp(self.log_log_alpha)+1j*self.alpha_img)*torch.exp(self.log_dt))
        # Loop over time axis
        for t in range(Wx.shape[1]):

            # Compute membrane potential (LIF)
            ut = alpha * (ut -st) + self.b * (Wx[:, t, :]) + (1-alpha)*(torch.matmul(st, V))

            # Compute spikes with surrogate gradient
            st = self.spike_fct(2*ut.real - self.threshold)
            s.append(st)

        return torch.stack(s, dim=1)
    
class RLIFcomplex1MinAlphaNoBLayer(nn.Module):
    """
    A single layer of Leaky Integrate-and-Fire neurons without layer-wise
    recurrent connections (LIF).

    Arguments
    ---------
    input_size : int
        Number of features in the input tensors.
    hidden_size : int
        Number of output neurons.
    batch_size : int
        Batch size of the input tensors.
    threshold : float
        Value of spiking threshold (fixed)
    dropout : float
        Dropout factor (must be between 0 and 1).
    normalization : str
        Type of normalization. Every string different from 'batchnorm'
        and 'layernorm' will result in no normalization.
    use_bias : bool
        If True, additional trainable bias is used with feedforward weights.
    bidirectional : bool
        If True, a bidirectional model that scans the sequence both directions
        is used, which doubles the size of feedforward matrices in layers l>0.
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        batch_size,
        threshold=1.0,
        dropout=0.0,
        normalization="batchnorm",
        use_bias=False,
        bidirectional=False,
        extra_features=None
    ):
        super().__init__()

        # Fixed parameters
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.batch_size = batch_size
        self.threshold = threshold
        self.dropout = dropout
        self.normalization = normalization
        self.use_bias = use_bias
        self.bidirectional = bidirectional
        self.batch_size = self.batch_size * (1 + self.bidirectional)
        
        self.spike_fct = SpikeFunctionBoxcar.apply

        # Trainable parameters
        self.W = nn.Linear(self.input_size, self.hidden_size, bias=use_bias)
        self.V = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        log_log_alpha = torch.log(0.5 * torch.ones(self.hidden_size))
        #self.log_log_alpha_lim = [math.log(1 / 200), math.log(1 / 5)]
        dt_min = 0.01
        dt_max = 0.4
        log_dt = torch.rand(self.hidden_size)*(
            math.log(dt_max) - math.log(dt_min)
        ) + math.log(dt_min)
        #nn.init.uniform_(log_log_alpha, self.log_log_alpha_lim[0], self.log_log_alpha_lim[1])
        alpha_img =  math.pi * torch.ones(self.hidden_size) # torch.arange(self.hidden_size)

        self.register("log_log_alpha", log_log_alpha, lr=0.01)
        self.register("log_dt", log_dt, lr=0.01)
        self.register("alpha_img", alpha_img, lr=0.01)

        self.b = nn.Parameter(torch.rand(self.hidden_size))

        # Initialize normalinzation
        self.normalize = False
        if normalization == "batchnorm":
            self.norm = nn.BatchNorm1d(self.hidden_size, momentum=0.05)
            self.normalize = True
        elif normalization == "layernorm":
            self.norm = nn.LayerNorm(self.hidden_size)
            self.normalize = True

        # Initialize dropout
        self.drop = nn.Dropout(p=dropout)

    def forward(self, x):

        # Concatenate flipped sequence on batch dim
        if self.bidirectional:
            x_flip = x.flip(1)
            x = torch.cat([x, x_flip], dim=0)

        # Change batch size if needed
        if self.batch_size != x.shape[0]:
            self.batch_size = x.shape[0]

        # Feed-forward affine transformations (all steps in parallel)
        Wx = self.W(x)

        #Wx = self.output_linear(Wx.reshape(Wx.shape[0], Wx.shape[2], Wx.shape[1])).reshape(Wx.shape[0], Wx.shape[1], Wx.shape[2])

        # Apply normalization
        if self.normalize:
            _Wx = self.norm(Wx.reshape(Wx.shape[0] * Wx.shape[1], Wx.shape[2]))
            Wx = _Wx.reshape(Wx.shape[0], Wx.shape[1], Wx.shape[2])

        # Compute spikes via neuron dynamics
        s = self._lif_cell(Wx)

        # Concatenate forward and backward sequences on feat dim
        if self.bidirectional:
            s_f, s_b = s.chunk(2, dim=0)
            s_b = s_b.flip(1)
            s = torch.cat([s_f, s_b], dim=2)

        # Apply dropout
        s = self.drop(s)

        return s
    def register(self, name, tensor, lr=None):
        """Register a tensor with a configurable learning rate and 0 weight decay"""

        if lr == 0.0:
            self.register_buffer(name, tensor)
        else:
            self.register_parameter(name, nn.Parameter(tensor))

            optim = {"weight_decay": 0.0}
            if lr is not None: optim["lr"] = lr
            setattr(getattr(self, name), "_optim", optim)


    def _lif_cell(self, Wx):

        # Initializations
        device = Wx.device
        ut = torch.rand(Wx.shape[0], Wx.shape[2], dtype=torch.cfloat).to(device)
        st = torch.rand(Wx.shape[0], Wx.shape[2]).to(device)
        s = []

        V = self.V.weight.clone().fill_diagonal_(0)

        # Bound values of the neuron parameters to plausible ranges
        #log_log__alpha = torch.clamp(self.log_log_alpha, min=self.log_log_alpha_lim[0], max=self.log_log_alpha_lim[1])
        alpha = torch.exp((-torch.exp(self.log_log_alpha)+1j*self.alpha_img)*torch.exp(self.log_dt))
        # Loop over time axis
        for t in range(Wx.shape[1]):

            # Compute membrane potential (LIF)
            ut = alpha * (ut -st) + (1-alpha) * (Wx[:, t, :] + torch.matmul(st, V))

            # Compute spikes with surrogate gradient
            st = self.spike_fct(2*ut.real - self.threshold)
            s.append(st)

        return torch.stack(s, dim=1)

class LIFcomplexDiscrLayer(nn.Module):
    """
    A single layer of Leaky Integrate-and-Fire neurons without layer-wise
    recurrent connections (LIF).

    Arguments
    ---------
    input_size : int
        Number of features in the input tensors.
    hidden_size : int
        Number of output neurons.
    batch_size : int
        Batch size of the input tensors.
    threshold : float
        Value of spiking threshold (fixed)
    dropout : float
        Dropout factor (must be between 0 and 1).
    normalization : str
        Type of normalization. Every string different from 'batchnorm'
        and 'layernorm' will result in no normalization.
    use_bias : bool
        If True, additional trainable bias is used with feedforward weights.
    bidirectional : bool
        If True, a bidirectional model that scans the sequence both directions
        is used, which doubles the size of feedforward matrices in layers l>0.
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        batch_size,
        threshold=1.0,
        dropout=0.0,
        normalization="batchnorm",
        use_bias=False,
        bidirectional=False,
        extra_features=None
    ):
        super().__init__()

        # Fixed parameters
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.batch_size = batch_size
        self.threshold = threshold
        self.dropout = dropout
        self.normalization = normalization
        self.use_bias = use_bias
        self.bidirectional = bidirectional
        self.batch_size = self.batch_size * (1 + self.bidirectional)
        
        self.spike_fct = SpikeFunctionBoxcar.apply

        # Trainable parameters
        self.W = nn.Linear(self.input_size, self.hidden_size, bias=use_bias)
        self.D = nn.Parameter(torch.randn(self.hidden_size))
        log_log_alpha = torch.log(0.5 * torch.ones(self.hidden_size))
        #self.log_log_alpha_lim = [math.log(1 / 200), math.log(1 / 5)]
        dt_min = 0.01
        dt_max = 0.4
        log_dt = torch.rand(self.hidden_size)*(
            math.log(dt_max) - math.log(dt_min)
        ) + math.log(dt_min)
        #nn.init.uniform_(log_log_alpha, self.log_log_alpha_lim[0], self.log_log_alpha_lim[1])
        alpha_img =  math.pi * torch.ones(self.hidden_size) # torch.arange(self.hidden_size)

        self.register("log_log_alpha", log_log_alpha, lr=0.01)
        self.register("log_dt", log_dt, lr=0.01)
        self.register("alpha_img", alpha_img, lr=0.01)

        self.b = nn.Parameter(torch.rand(self.hidden_size))

        # position-wise output transform to mix features
        self.output_linear = nn.Sequential(
            nn.Conv1d(self.hidden_size, 2*self.hidden_size, kernel_size=1),
            nn.GLU(dim=-2),
        )

        # Initialize normalinzation
        self.normalize = False
        if normalization == "batchnorm":
            self.norm = nn.BatchNorm1d(self.hidden_size, momentum=0.05)
            self.normalize = True
        elif normalization == "layernorm":
            self.norm = nn.LayerNorm(self.hidden_size)
            self.normalize = True

        # Initialize dropout
        self.drop = nn.Dropout(p=dropout)

        self.output_linear = nn.Sequential(
            nn.Conv1d(self.hidden_size, 2*self.hidden_size, kernel_size=1),
            nn.GLU(dim=-2),
        )

    def forward(self, x):

        # Concatenate flipped sequence on batch dim
        if self.bidirectional:
            x_flip = x.flip(1)
            x = torch.cat([x, x_flip], dim=0)

        # Change batch size if needed
        if self.batch_size != x.shape[0]:
            self.batch_size = x.shape[0]

        # Feed-forward affine transformations (all steps in parallel)
        Wx = self.W(x)

        #Wx = self.output_linear(Wx.reshape(Wx.shape[0], Wx.shape[2], Wx.shape[1])).reshape(Wx.shape[0], Wx.shape[1], Wx.shape[2])

        # Apply normalization
        if self.normalize:
            _Wx = self.norm(Wx.reshape(Wx.shape[0] * Wx.shape[1], Wx.shape[2]))
            Wx = _Wx.reshape(Wx.shape[0], Wx.shape[1], Wx.shape[2])

        # Compute spikes via neuron dynamics
        s = self._lif_cell(Wx)

        Wx = self.output_linear(Wx.reshape(Wx.shape[0], Wx.shape[2], Wx.shape[1])).reshape(Wx.shape[0], Wx.shape[1], Wx.shape[2])


        # Concatenate forward and backward sequences on feat dim
        if self.bidirectional:
            s_f, s_b = s.chunk(2, dim=0)
            s_b = s_b.flip(1)
            s = torch.cat([s_f, s_b], dim=2)

        # Apply dropout
        s = self.drop(s)

        return s
    def register(self, name, tensor, lr=None):
        """Register a tensor with a configurable learning rate and 0 weight decay"""

        if lr == 0.0:
            self.register_buffer(name, tensor)
        else:
            self.register_parameter(name, nn.Parameter(tensor))

            optim = {"weight_decay": 0.0}
            if lr is not None: optim["lr"] = lr
            setattr(getattr(self, name), "_optim", optim)


    def _lif_cell(self, Wx):

        # Initializations
        device = Wx.device
        ut = torch.rand(Wx.shape[0], Wx.shape[2], dtype=torch.cfloat).to(device)
        st = torch.rand(Wx.shape[0], Wx.shape[2]).to(device)
        s = []

        # Bound values of the neuron parameters to plausible ranges
        #log_log__alpha = torch.clamp(self.log_log_alpha, min=self.log_log_alpha_lim[0], max=self.log_log_alpha_lim[1])
        alpha = torch.exp((-torch.exp(self.log_log_alpha)+1j*self.alpha_img)*torch.exp(self.log_dt))
        b_disc = self.b * (alpha-1.0)/(-torch.exp(self.log_log_alpha)+1j*self.alpha_img)
        # Loop over time axis
        for t in range(Wx.shape[1]):

            # Compute membrane potential (LIF)
            ut = alpha * (ut -st) + b_disc * Wx[:, t, :]

            # Compute spikes with surrogate gradient
            st = self.spike_fct(2*ut.real - self.threshold)
            s.append(st)

        return torch.stack(s, dim=1)


class LIFcomplex_gatedBLayer(nn.Module):
    """
    A single layer of Leaky Integrate-and-Fire neurons without layer-wise
    recurrent connections (LIF).

    Arguments
    ---------
    input_size : int
        Number of features in the input tensors.
    hidden_size : int
        Number of output neurons.
    batch_size : int
        Batch size of the input tensors.
    threshold : float
        Value of spiking threshold (fixed)
    dropout : float
        Dropout factor (must be between 0 and 1).
    normalization : str
        Type of normalization. Every string different from 'batchnorm'
        and 'layernorm' will result in no normalization.
    use_bias : bool
        If True, additional trainable bias is used with feedforward weights.
    bidirectional : bool
        If True, a bidirectional model that scans the sequence both directions
        is used, which doubles the size of feedforward matrices in layers l>0.
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        batch_size,
        threshold=1.0,
        dropout=0.0,
        normalization="batchnorm",
        use_bias=False,
        bidirectional=False,
        extra_features=None
    ):
        super().__init__()

        # Fixed parameters
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.batch_size = batch_size
        self.threshold = threshold
        self.dropout = dropout
        self.normalization = normalization
        self.use_bias = use_bias
        self.bidirectional = bidirectional
        self.batch_size = self.batch_size * (1 + self.bidirectional)
        
        self.spike_fct = SpikeFunctionBoxcar.apply

        # Trainable parameters
        self.W = nn.Linear(self.input_size, self.hidden_size, bias=use_bias)

        log_log_alpha = torch.log(0.5 * torch.ones(self.hidden_size))
        #self.log_log_alpha_lim = [math.log(1 / 200), math.log(1 / 5)]
        dt_min = 0.01 #0.001 #0.01
        dt_max = 0.4 #0.1 #0.4
        #log_dt = torch.rand(self.hidden_size)*(
        ##    math.log(dt_max) - math.log(dt_min)
        #) + math.log(dt_min)

        dt_init = "random"
        dt_scale = 1.0
        dt_rank = "auto"
        self.dt_rank = math.ceil(self.hidden_size / 1) if dt_rank == "auto" else dt_rank

        #self.dt_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.x_proj = nn.Linear(
            self.hidden_size,1, bias=False
        )
        dt_min = 0.01
        dt_max = 0.4
        log_dt = torch.rand(self.hidden_size)*(
            math.log(dt_max) - math.log(dt_min)
        ) + math.log(dt_min)
        self.register("log_dt", log_dt, lr=0.01)
        

        #nn.init.uniform_(log_log_alpha, self.log_log_alpha_lim[0], self.log_log_alpha_lim[1])
        alpha_img =  math.pi * torch.ones(self.hidden_size) # torch.arange(self.hidden_size)



        self.register("log_log_alpha", log_log_alpha, lr=0.01)
        #self.register("log_dt", log_dt, lr=0.01)
        self.register("alpha_img", alpha_img, lr=0.01)

        self.b = nn.Parameter(torch.rand(self.hidden_size))

        # Initialize normalinzation
        self.normalize = False
        if normalization == "batchnorm":
            self.norm = nn.BatchNorm1d(self.hidden_size, momentum=0.05)
            self.normalize = True
        elif normalization == "layernorm":
            self.norm = nn.LayerNorm(self.hidden_size)
            self.normalize = True

        # Initialize dropout
        self.drop = nn.Dropout(p=dropout)
        self.sigm = nn.Sigmoid()
        self.normB = nn.BatchNorm1d(1, momentum=0.05)

    def forward(self, x):

        # Concatenate flipped sequence on batch dim
        if self.bidirectional:
            x_flip = x.flip(1)
            x = torch.cat([x, x_flip], dim=0)

        # Change batch size if needed
        if self.batch_size != x.shape[0]:
            self.batch_size = x.shape[0]

        # Feed-forward affine transformations (all steps in parallel)
        Wx = self.W(x)

        #Wx = self.output_linear(Wx.reshape(Wx.shape[0], Wx.shape[2], Wx.shape[1])).reshape(Wx.shape[0], Wx.shape[1], Wx.shape[2])

        # Apply normalization
        if self.normalize:
            _Wx = self.norm(Wx.reshape(Wx.shape[0] * Wx.shape[1], Wx.shape[2]))
            Wx = _Wx.reshape(Wx.shape[0], Wx.shape[1], Wx.shape[2])

        # Compute spikes via neuron dynamics
        s = self._lif_cell(Wx)

        # Concatenate forward and backward sequences on feat dim
        if self.bidirectional:
            s_f, s_b = s.chunk(2, dim=0)
            s_b = s_b.flip(1)
            s = torch.cat([s_f, s_b], dim=2)

        # Apply dropout
        s = self.drop(s)

        return s
    def register(self, name, tensor, lr=None):
        """Register a tensor with a configurable learning rate and 0 weight decay"""

        if lr == 0.0:
            self.register_buffer(name, tensor)
        else:
            self.register_parameter(name, nn.Parameter(tensor))

            optim = {"weight_decay": 0.0}
            if lr is not None: optim["lr"] = lr
            setattr(getattr(self, name), "_optim", optim)


    def _lif_cell(self, Wx):

        # Initializations
        device = Wx.device
        ut = torch.rand(Wx.shape[0], Wx.shape[2], dtype=torch.cfloat).to(device)
        st = torch.rand(Wx.shape[0], Wx.shape[2]).to(device)
        s = []


        b = self.x_proj(rearrange(Wx, "b l d -> (b l) d"))  # (bl dt_rank)
        b = rearrange(b, "(b l) d -> b d l", l=Wx.shape[1])
        '''
        min_d = b.min(dim=2, keepdim=True)[0]
        max_d = b.max(dim=2, keepdim=True)[0]
        range_d = max_d - min_d
        epsilon = 1e-8
        range_d = range_d + epsilon
        b = (b - min_d) / range_d
        '''

        #b = self.normB(b)
        #b = self.sigm(b)
        dt = torch.exp(self.log_dt)
        b = torch.transpose((dt * torch.transpose(b, 1,2)) , 1,2)

        # Bound values of the neuron parameters to plausible ranges
        #log_log__alpha = torch.clamp(self.log_log_alpha, min=self.log_log_alpha_lim[0], max=self.log_log_alpha_lim[1])

        alpha = torch.exp((-torch.exp(self.log_log_alpha)+1j*self.alpha_img)*dt)
        # Loop over time axis
        for t in range(Wx.shape[1]):

            # Compute membrane potential (LIF)
            ut = alpha * (ut -st) + b[:,:,t] * Wx[:, t, :]

            # Compute spikes with surrogate gradient
            st = self.spike_fct(2*ut.real - self.threshold)
            s.append(st)

        return torch.stack(s, dim=1)


class LIFcomplex_gatedDtLayer(nn.Module):
    """
    A single layer of Leaky Integrate-and-Fire neurons without layer-wise
    recurrent connections (LIF).

    Arguments
    ---------
    input_size : int
        Number of features in the input tensors.
    hidden_size : int
        Number of output neurons.
    batch_size : int
        Batch size of the input tensors.
    threshold : float
        Value of spiking threshold (fixed)
    dropout : float
        Dropout factor (must be between 0 and 1).
    normalization : str
        Type of normalization. Every string different from 'batchnorm'
        and 'layernorm' will result in no normalization.
    use_bias : bool
        If True, additional trainable bias is used with feedforward weights.
    bidirectional : bool
        If True, a bidirectional model that scans the sequence both directions
        is used, which doubles the size of feedforward matrices in layers l>0.
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        batch_size,
        threshold=1.0,
        dropout=0.0,
        normalization="batchnorm",
        use_bias=False,
        bidirectional=False,
        extra_features=None
    ):
        super().__init__()

        # Fixed parameters
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.batch_size = batch_size
        self.threshold = threshold
        self.dropout = dropout
        self.normalization = normalization
        self.use_bias = use_bias
        self.bidirectional = bidirectional
        self.batch_size = self.batch_size * (1 + self.bidirectional)
        
        self.spike_fct = SpikeFunctionBoxcar.apply

        # Trainable parameters
        self.W = nn.Linear(self.input_size, self.hidden_size, bias=use_bias)

        log_log_alpha = torch.log(0.5 * torch.ones(self.hidden_size))
        #self.log_log_alpha_lim = [math.log(1 / 200), math.log(1 / 5)]
        self.dt_min = 0.01 #0.001 #0.01
        self.dt_max = 0.4 #0.1 #0.4
        #log_dt = torch.rand(self.hidden_size)*(
        ##    math.log(dt_max) - math.log(dt_min)
        #) + math.log(dt_min)

        dt_init = "random"
        dt_scale = 1.0
        dt_rank = "auto"
        self.dt_rank = math.ceil(self.hidden_size / 1) if dt_rank == "auto" else dt_rank

        #self.dt_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.x_proj = nn.Linear(
            self.hidden_size, self.dt_rank, bias=False
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.hidden_size, bias=True)

        dt_init_std = (self.dt_rank)**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        
        
        dt = torch.exp(
            torch.rand(self.hidden_size) * (math.log(self.dt_max) - math.log(self.dt_min))
            + math.log(self.dt_min)
        ).clamp(min=1e-4)

        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True

        #nn.init.uniform_(log_log_alpha, self.log_log_alpha_lim[0], self.log_log_alpha_lim[1])
        alpha_img =  math.pi * torch.ones(self.hidden_size) # torch.arange(self.hidden_size)



        self.register("log_log_alpha", log_log_alpha, lr=0.01)
        #self.register("log_dt", log_dt, lr=0.01)
        self.register("alpha_img", alpha_img, lr=0.01)

        self.b = nn.Parameter(torch.rand(self.hidden_size))

        # Initialize normalinzation
        self.normalize = False
        if normalization == "batchnorm":
            self.norm = nn.BatchNorm1d(self.hidden_size, momentum=0.05)
            self.normalize = True
        elif normalization == "layernorm":
            self.norm = nn.LayerNorm(self.hidden_size)
            self.normalize = True

        # Initialize dropout
        self.drop = nn.Dropout(p=dropout)

    def forward(self, x):

        # Concatenate flipped sequence on batch dim
        if self.bidirectional:
            x_flip = x.flip(1)
            x = torch.cat([x, x_flip], dim=0)

        # Change batch size if needed
        if self.batch_size != x.shape[0]:
            self.batch_size = x.shape[0]

        # Feed-forward affine transformations (all steps in parallel)
        Wx = self.W(x)

        #Wx = self.output_linear(Wx.reshape(Wx.shape[0], Wx.shape[2], Wx.shape[1])).reshape(Wx.shape[0], Wx.shape[1], Wx.shape[2])

        # Apply normalization
        if self.normalize:
            _Wx = self.norm(Wx.reshape(Wx.shape[0] * Wx.shape[1], Wx.shape[2]))
            Wx = _Wx.reshape(Wx.shape[0], Wx.shape[1], Wx.shape[2])

        # Compute spikes via neuron dynamics
        s = self._lif_cell(Wx)

        # Concatenate forward and backward sequences on feat dim
        if self.bidirectional:
            s_f, s_b = s.chunk(2, dim=0)
            s_b = s_b.flip(1)
            s = torch.cat([s_f, s_b], dim=2)

        # Apply dropout
        s = self.drop(s)

        return s
    def register(self, name, tensor, lr=None):
        """Register a tensor with a configurable learning rate and 0 weight decay"""

        if lr == 0.0:
            self.register_buffer(name, tensor)
        else:
            self.register_parameter(name, nn.Parameter(tensor))

            optim = {"weight_decay": 0.0}
            if lr is not None: optim["lr"] = lr
            setattr(getattr(self, name), "_optim", optim)


    def _lif_cell(self, Wx):

        # Initializations
        device = Wx.device
        ut = torch.rand(Wx.shape[0], Wx.shape[2], dtype=torch.cfloat).to(device)
        st = torch.rand(Wx.shape[0], Wx.shape[2]).to(device)
        s = []


        dt1 = self.x_proj(rearrange(Wx, "b l d -> (b l) d"))  # (bl dt_rank)
        bias = repeat(
            self.dt_proj.bias,
            "n -> n d",
            d=Wx.shape[0]*Wx.shape[1],
        )
        dt = F.softplus( self.dt_proj.weight @ dt1.t() + bias)
        dt = rearrange(dt, "d (b l) -> b d l", l=Wx.shape[1])
        dt = torch.clamp(dt, min = self.dt_min, max = self.dt_max)

        

        #dt = torch.sigmoid(dt)

        # Scale and shift to get values between 0.001 and 0.4
        #dt = 0.001 + dt * (0.4 - 0.001)

        # Bound values of the neuron parameters to plausible ranges
        #log_log__alpha = torch.clamp(self.log_log_alpha, min=self.log_log_alpha_lim[0], max=self.log_log_alpha_lim[1])

        alpha = torch.exp((-torch.exp(self.log_log_alpha)+1j*self.alpha_img).unsqueeze(0).unsqueeze(2).repeat(Wx.shape[0], 1, Wx.shape[1])*dt) # B H L 
        # Loop over time axis
        for t in range(Wx.shape[1]):

            # Compute membrane potential (LIF)
            ut = alpha[:,:,t] * (ut -st) + self.b * Wx[:, t, :]

            # Compute spikes with surrogate gradient
            st = self.spike_fct(2*ut.real - self.threshold)
            s.append(st)

        return torch.stack(s, dim=1)

class RLIFLayer(nn.Module):
    """
    A single layer of Leaky Integrate-and-Fire neurons with layer-wise
    recurrent connections (RLIF).

    Arguments
    ---------
    input_size : int
        Number of features in the input tensors.
    hidden_size : int
        Number of output neurons.
    batch_size : int
        Batch size of the input tensors.
    threshold : float
        Value of spiking threshold (fixed)
    dropout : float
        Dropout factor (must be between 0 and 1).
    normalization : str
        Type of normalization. Every string different from 'batchnorm'
        and 'layernorm' will result in no normalization.
    use_bias : bool
        If True, additional trainable bias is used with feedforward weights.
    bidirectional : bool
        If True, a bidirectional model that scans the sequence both directions
        is used, which doubles the size of feedforward matrices in layers l>0.
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        batch_size,
        threshold=1.0,
        dropout=0.0,
        normalization="batchnorm",
        use_bias=False,
        bidirectional=False,
        extra_features=None
    ):
        super().__init__()

        # Fixed parameters
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.batch_size = batch_size
        self.threshold = threshold
        self.dropout = dropout
        self.normalization = normalization
        self.use_bias = use_bias
        self.bidirectional = bidirectional
        self.batch_size = self.batch_size * (1 + self.bidirectional)
        self.alpha_lim = [np.exp(-1 / 5), np.exp(-1 / 25)]
        self.spike_fct = SpikeFunctionBoxcar.apply

        # Trainable parameters
        self.W = nn.Linear(self.input_size, self.hidden_size, bias=use_bias)
        self.V = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.alpha = nn.Parameter(torch.Tensor(self.hidden_size))

        nn.init.uniform_(self.alpha, self.alpha_lim[0], self.alpha_lim[1])
        nn.init.orthogonal_(self.V.weight)

        # Initialize normalization
        self.normalize = False
        if normalization == "batchnorm":
            self.norm = nn.BatchNorm1d(self.hidden_size, momentum=0.05)
            self.normalize = True
        elif normalization == "layernorm":
            self.norm = nn.LayerNorm(self.hidden_size)
            self.normalize = True

        # Initialize dropout
        self.drop = nn.Dropout(p=dropout)

        if extra_features['rst_detach']:
            self.rst_detach = True
        else:
            self.rst_detach = False

    def forward(self, x):

        # Concatenate flipped sequence on batch dim
        if self.bidirectional:
            x_flip = x.flip(1)
            x = torch.cat([x, x_flip], dim=0)

        # Change batch size if needed
        if self.batch_size != x.shape[0]:
            self.batch_size = x.shape[0]

        # Feed-forward affine transformations (all steps in parallel)
        Wx = self.W(x)

        # Apply normalization
        if self.normalize:
            _Wx = self.norm(Wx.reshape(Wx.shape[0] * Wx.shape[1], Wx.shape[2]))
            Wx = _Wx.reshape(Wx.shape[0], Wx.shape[1], Wx.shape[2])

        # Compute spikes via neuron dynamics
        s = self._rlif_cell(Wx)

        # Concatenate forward and backward sequences on feat dim
        if self.bidirectional:
            s_f, s_b = s.chunk(2, dim=0)
            s_b = s_b.flip(1)
            s = torch.cat([s_f, s_b], dim=2)

        # Apply dropout
        s = self.drop(s)

        return s

    def _rlif_cell(self, Wx):

        # Initializations
        device = Wx.device
        ut = torch.rand(Wx.shape[0], Wx.shape[2]).to(device)
        st = torch.rand(Wx.shape[0], Wx.shape[2]).to(device)
        s = []

        # Bound values of the neuron parameters to plausible ranges
        alpha = torch.clamp(self.alpha, min=self.alpha_lim[0], max=self.alpha_lim[1])

        # Set diagonal elements of recurrent matrix to zero
        V = self.V.weight.clone().fill_diagonal_(0)

        # Loop over time axis
        for t in range(Wx.shape[1]):
            
            if self.rst_detach:
                reset = st.clone().detach()
            else: 
                reset = st

            # Compute membrane potential (RLIF)
            ut = alpha * (ut - reset) + (1 - alpha) * (Wx[:, t, :] + torch.matmul(st, V))

            # Compute spikes with surrogate gradient
            st = self.spike_fct(ut - self.threshold)
            s.append(st)

        return torch.stack(s, dim=1)


class RadLIFLayer(nn.Module):
    """
    A single layer of adaptive Leaky Integrate-and-Fire neurons with layer-wise
    recurrent connections (RadLIF).

    Arguments
    ---------
    input_size : int
        Number of features in the input tensors.
    hidden_size : int
        Number of output neurons.
    batch_size : int
        Batch size of the input tensors.
    threshold : float
        Value of spiking threshold (fixed)
    dropout : float
        Dropout factor (must be between 0 and 1).
    normalization : str
        Type of normalization. Every string different from 'batchnorm'
        and 'layernorm' will result in no normalization.
    use_bias : bool
        If True, additional trainable bias is used with feedforward weights.
    bidirectional : bool
        If True, a bidirectional model that scans the sequence both directions
        is used, which doubles the size of feedforward matrices in layers l>0.
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        batch_size,
        threshold=1.0,
        dropout=0.0,
        normalization="batchnorm",
        use_bias=False,
        bidirectional=False,
        extra_features=None
    ):
        super().__init__()

        # Fixed parameters
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.batch_size = batch_size
        self.threshold = threshold
        self.dropout = dropout
        self.normalization = normalization
        self.use_bias = use_bias
        self.bidirectional = bidirectional
        self.batch_size = self.batch_size * (1 + self.bidirectional)
        self.alpha_lim = [np.exp(-1 / 5), np.exp(-1 / 25)]
        self.beta_lim = [np.exp(-1 / 30), np.exp(-1 / 120)]
        self.a_lim = [-1.0, 1.0]
        self.b_lim = [0.0, 2.0]
        self.spike_fct = SpikeFunctionBoxcar.apply

        # Trainable parameters
        self.W = nn.Linear(self.input_size, self.hidden_size, bias=use_bias)
        self.V = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.alpha = nn.Parameter(torch.Tensor(self.hidden_size))
        self.beta = nn.Parameter(torch.Tensor(self.hidden_size))
        self.a = nn.Parameter(torch.Tensor(self.hidden_size))
        self.b = nn.Parameter(torch.Tensor(self.hidden_size))

        nn.init.uniform_(self.alpha, self.alpha_lim[0], self.alpha_lim[1])
        nn.init.uniform_(self.beta, self.beta_lim[0], self.beta_lim[1])
        nn.init.uniform_(self.a, self.a_lim[0], self.a_lim[1])
        nn.init.uniform_(self.b, self.b_lim[0], self.b_lim[1])
        nn.init.orthogonal_(self.V.weight)

        # Initialize normalization
        self.normalize = False
        if normalization == "batchnorm":
            self.norm = nn.BatchNorm1d(self.hidden_size, momentum=0.05)
            self.normalize = True
        elif normalization == "layernorm":
            self.norm = nn.LayerNorm(self.hidden_size)
            self.normalize = True

        # Initialize dropout
        self.drop = nn.Dropout(p=dropout)

        if extra_features['rst_detach']:
            self.rst_detach = True
        else:
            self.rst_detach = False

    def forward(self, x):

        # Concatenate flipped sequence on batch dim
        if self.bidirectional:
            x_flip = x.flip(1)
            x = torch.cat([x, x_flip], dim=0)

        # Change batch size if needed
        if self.batch_size != x.shape[0]:
            self.batch_size = x.shape[0]

        # Feed-forward affine transformations (all steps in parallel)
        Wx = self.W(x)

        # Apply normalization
        if self.normalize:
            _Wx = self.norm(Wx.reshape(Wx.shape[0] * Wx.shape[1], Wx.shape[2]))
            Wx = _Wx.reshape(Wx.shape[0], Wx.shape[1], Wx.shape[2])

        # Compute spikes via neuron dynamics
        s = self._radlif_cell(Wx)

        # Concatenate forward and backward sequences on feat dim
        if self.bidirectional:
            s_f, s_b = s.chunk(2, dim=0)
            s_b = s_b.flip(1)
            s = torch.cat([s_f, s_b], dim=2)

        # Apply dropout
        s = self.drop(s)

        return s

    def _radlif_cell(self, Wx):

        # Initializations
        device = Wx.device
        ut = torch.rand(Wx.shape[0], Wx.shape[2]).to(device)
        wt = torch.rand(Wx.shape[0], Wx.shape[2]).to(device)
        st = torch.rand(Wx.shape[0], Wx.shape[2]).to(device)
        s = []

        # Bound values of the neuron parameters to plausible ranges
        alpha = torch.clamp(self.alpha, min=self.alpha_lim[0], max=self.alpha_lim[1])
        beta = torch.clamp(self.beta, min=self.beta_lim[0], max=self.beta_lim[1])
        a = torch.clamp(self.a, min=self.a_lim[0], max=self.a_lim[1])
        b = torch.clamp(self.b, min=self.b_lim[0], max=self.b_lim[1])

        # Set diagonal elements of recurrent matrix to zero
        V = self.V.weight.clone().fill_diagonal_(0)

        # Loop over time axis
        for t in range(Wx.shape[1]):

            if self.rst_detach:
                reset = st.clone().detach()
            else:
                reset = st

            # Compute potential (RadLIF)
            wt = beta * wt + a * ut + b * reset
            ut = alpha * (ut - reset) + (1 - alpha) * (
                Wx[:, t, :] + torch.matmul(st, V) - wt
            )

            # Compute spikes with surrogate gradient
            st = self.spike_fct(ut - self.threshold)
            s.append(st)

        return torch.stack(s, dim=1)


class ReadoutLayer(nn.Module):
    """
    This function implements a single layer of non-spiking Leaky Integrate and
    Fire (LIF) neurons, where the output consists of a cumulative sum of the
    membrane potential using a softmax function, instead of spikes.

    Arguments
    ---------
    input_size : int
        Feature dimensionality of the input tensors.
    hidden_size : int
        Number of output neurons.
    batch_size : int
        Batch size of the input tensors.
    dropout : float
        Dropout factor (must be between 0 and 1).
    normalization : str
        Type of normalization. Every string different from 'batchnorm'
        and 'layernorm' will result in no normalization.
    use_bias : bool
        If True, additional trainable bias is used with feedforward weights.
    bidirectional : bool
        If True, a bidirectional model that scans the sequence both directions
        is used, which doubles the size of feedforward matrices in layers l>0.
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        batch_size,
        dropout=0.0,
        normalization="batchnorm",
        use_bias=False,
        extra_features=None
    ):
        super().__init__()

        # Fixed parameters
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.batch_size = batch_size
        self.dropout = dropout
        self.normalization = normalization
        self.use_bias = use_bias
        self.alpha_lim = [np.exp(-1 / 5), np.exp(-1 / 25)]

        # Trainable parameters
        self.W = nn.Linear(self.input_size, self.hidden_size, bias=use_bias)
        self.alpha = nn.Parameter(torch.Tensor(self.hidden_size))
        nn.init.uniform_(self.alpha, self.alpha_lim[0], self.alpha_lim[1])

        # Initialize normalization
        self.normalize = False
        if normalization == "batchnorm":
            self.norm = nn.BatchNorm1d(self.hidden_size, momentum=0.05)
            self.normalize = True
        elif normalization == "layernorm":
            self.norm = nn.LayerNorm(self.hidden_size)
            self.normalize = True

        # Initialize dropout
        self.drop = nn.Dropout(p=dropout)

        self.time_offset = extra_features['time_offset']

    def forward(self, x):

        # Feed-forward affine transformations (all steps in parallel)
        Wx = self.W(x)

        # Apply normalization
        if self.normalize:
            _Wx = self.norm(Wx.reshape(Wx.shape[0] * Wx.shape[1], Wx.shape[2]))
            Wx = _Wx.reshape(Wx.shape[0], Wx.shape[1], Wx.shape[2])

        # Compute membrane potential via non-spiking neuron dynamics
        out = self._readout_cell(Wx)

        return out

    def _readout_cell(self, Wx):

        # Initializations
        device = Wx.device
        ut = torch.rand(Wx.shape[0], Wx.shape[2]).to(device)
        out = torch.zeros(Wx.shape[0], Wx.shape[2]).to(device)

        # Bound values of the neuron parameters to plausible ranges
        alpha = torch.clamp(self.alpha, min=self.alpha_lim[0], max=self.alpha_lim[1])

        # Loop over time axis
        for t in range(self.time_offset, Wx.shape[1]):

            # Compute potential (LIF)
            ut = alpha * ut + (1 - alpha) * Wx[:, t, :]
            out = out + F.softmax(ut, dim=1)

        return out

class SEReadoutLayer(nn.Module):
    """
    This function implements a single layer of non-spiking Leaky Integrate and
    Fire (LIF) neurons, where the output consists of a cumulative sum of the
    membrane potential using a softmax function, instead of spikes.

    Arguments
    ---------
    input_size : int
        Feature dimensionality of the input tensors.
    hidden_size : int
        Number of output neurons.
    batch_size : int
        Batch size of the input tensors.
    dropout : float
        Dropout factor (must be between 0 and 1).
    normalization : str
        Type of normalization. Every string different from 'batchnorm'
        and 'layernorm' will result in no normalization.
    use_bias : bool
        If True, additional trainable bias is used with feedforward weights.
    bidirectional : bool
        If True, a bidirectional model that scans the sequence both directions
        is used, which doubles the size of feedforward matrices in layers l>0.
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        batch_size,
        dropout=0.0,
        normalization="batchnorm",
        use_bias=False,
    ):
        super().__init__()

        # Fixed parameters
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.batch_size = batch_size
        self.dropout = dropout
        self.normalization = normalization
        self.use_bias = use_bias
        self.alpha = np.exp(-1.0 / 15.0)

        # Trainable parameters
        self.W = nn.Linear(self.input_size, self.hidden_size, bias=use_bias)

        # Initialize normalization
        self.normalize = False
        if normalization == "batchnorm":
            self.norm = nn.BatchNorm1d(self.hidden_size, momentum=0.05)
            self.normalize = True
        elif normalization == "layernorm":
            self.norm = nn.LayerNorm(self.hidden_size)
            self.normalize = True

        # Initialize dropout
        self.drop = nn.Dropout(p=dropout)

        

    def forward(self, x):

        # Feed-forward affine transformations (all steps in parallel)
        Wx = self.W(x)

        # Apply normalization
        if self.normalize:
            _Wx = self.norm(Wx.reshape(Wx.shape[0] * Wx.shape[1], Wx.shape[2]))
            Wx = _Wx.reshape(Wx.shape[0], Wx.shape[1], Wx.shape[2])

        # Compute membrane potential via non-spiking neuron dynamics
        out = self._readout_cell(Wx)

        return out

    def _readout_cell(self, Wx):

        # Initializations
        device = Wx.device
        ut = torch.rand(Wx.shape[0], Wx.shape[2]).to(device)
        out = torch.zeros(Wx.shape[0], Wx.shape[2]).to(device)

        # Bound values of the neuron parameters to plausible ranges
        alpha = self.alpha

        # Loop over time axis
        for t in range(10, Wx.shape[1]):

            # Compute potential (LIF)
            ut = alpha * ut + (1 - alpha) * Wx[:, t, :]
            out = out + F.softmax(ut, dim=1)

        return out

class Network_S4(nn.Module):
    #chnages to Maximes implementation: initialization, alpha clampling 
    def __init__(
        self,
        input_shape,
        input_size,
        layer_size,
        output_size,
        state_size, 
        block_num,  
        dropout=0.0,  
        lr=0.01
        ):
        super().__init__() 

        # Fixed parameters
        self.input_size = input_size
        self.batch_size = input_shape[0]
        self.h = layer_size
        self.output_size = output_size
        self.n = state_size
        self.block_num = block_num
         
        self.dropout = dropout 


        self.encoder = nn.Linear(input_size, layer_size)
        
        self.log_A_reals = nn.ParameterList()
        self.log_dts = nn.ParameterList()
        self.Cs = nn.ParameterList() 
        self.Ds = nn.ParameterList() 

        self.dropouts1 = nn.ModuleList()
        self.dropouts2 = nn.ModuleList()

        self.activation = nn.GELU()

        self.glu_module = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(self.h, 2 * self.h, kernel_size=1), #(batch_size, layer_size*2, 1)
                nn.GLU(dim=-2) #(batch_size, layer_size, 1)
            ) for _ in range(self.block_num)
        ])

        self.decoder = nn.Linear(layer_size, output_size)
        self.norm_block = nn.ModuleList()

        dt_min=0.001
        dt_max=0.1
        for i in range(self.block_num):

            weight = torch.randn(self.h, self.n)
            self.Cs.append(nn.Parameter(weight))

            log_A_real = torch.log(0.5 * torch.ones(self.h, self.n))
            self.register("log_A_real"+str(i), log_A_real, lr)            
            self.log_A_reals.append(getattr(self, "log_A_real"+str(i)))

            log_dt = torch.rand(self.h) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
            self.register("log_dt"+str(i), log_dt, lr)
            self.log_dts.append(getattr(self, "log_dt"+str(i)))

            self.Ds.append(nn.Parameter(torch.randn(self.h, 1)))

            self.norm_block.append(nn.LayerNorm(self.h))

            self.dropouts1.append(DropoutNd(dropout))
            self.dropouts2.append(DropoutNd(dropout))
        self.is_snn = False
        

            

    def forward(self, x):
        batch_size, seq_length, _ = x.shape

        # Initialize states
        #u = [torch.zeros(batch_size,  self.h, self.n, seq_length).to(x.device) for _ in range(self.block_num)]
        #y = torch.zeros(batch_size, self.h, self.output_size).to(x.device)
        

        x = self.encoder(x) #(batch_size, L, input_size) --> (b, L, h) 
        x = x.transpose(1,2) #blh-->bhl

        for l in range(self.block_num):
            x_in = x.transpose(1,2).unsqueeze(-1).expand(batch_size, seq_length, self.h, self.n) #  b h l --> (b l h) --> b l h 1 --> b l h n 
            dt = torch.exp(self.log_dts[l]).unsqueeze(-1) #h 1
            A = torch.exp(dt * -torch.exp(self.log_A_reals[l]))  #h n 
            u_l = torch.zeros(batch_size,  self.h, self.n, seq_length).to(x.device)
            for t in range(seq_length-1):
                u_l[:,:,:,t+1] = A.unsqueeze(0) * u_l[:,:,:,t].clone() + x_in[:, t, :, :] 
            y = torch.einsum('bhnl,hn->bhl', u_l, self.Cs[l])  + x * self.Ds[l]
            y = self.activation(y)
            y = self.dropouts1[l](y)
            y = self.glu_module[l](y)
            x = self.dropouts2[l](y) + x
            x = self.norm_block[l](x.permute(2, 0, 1)) #lbh
            x = x.permute(1, 2, 0) #lbh --> bhl

        x = x.transpose(-1, -2) #bhl -> blh 
        out = x.mean(dim=1) # bh 
        out = self.decoder(out)  # (B, h) -> (B, d_output)
        
        '''
        x = x.transpose(1,2) #b l h --> b h l

        for l in range(self.block_num):
            x_in = x.transpose(1,2).unsqueeze(-1).expand(batch_size, seq_length, self.h, self.n) #  b h l --> (b l h) --> b l h 1 --> b l h n 
            dt = torch.exp(self.log_dts[l]).unsqueeze(-1) #h 1
            A = torch.exp(dt * -torch.exp(self.log_A_reals[l]))  #h n 
            u_l = torch.zeros(batch_size,  self.h, self.n, seq_length).to(x.device)
            for t in range(seq_length-1):
                u_l[:,:,:,t+1] = A.unsqueeze(0) * u_l[:,:,:,t] + x_in[:, t, :, :] 
            x = torch.einsum('bhnl,hn->bhl', u_l, self.Cs[l])  + x * self.Ds[l] # ?? x: bhl
            x = self.activation(x)
            x = self.dropouts1[l](x)

            x = self.glu_module[l](x)
            x = self.dropouts2[l](x) #+ x

            x = self.norm_block[l](x.permute(2, 0, 1)) #lbh

            x = x.permute(1, 2, 0) #lbh --> bhl
        
        z = x.transpose(-1, -2) #bhl -> blh 
        out = z.mean(dim=1)

        out = self.decoder(out)  # (B, h) -> (B, d_output)
'''
        return out, 0
    
    def register(self, name, tensor, lr=None):
        """Register a tensor with a configurable learning rate and 0 weight decay"""

        if lr == 0.0:
            self.register_buffer(name, tensor)
        else:
            self.register_parameter(name, nn.Parameter(tensor))

            optim = {"weight_decay": 0.0}
            if lr is not None: optim["lr"] = lr
            setattr(getattr(self, name), "_optim", optim)


class DropoutNd(nn.Module):
    def __init__(self, p: float = 0.5, tie=True, transposed=True):
        """
        tie: tie dropout mask across sequence lengths (Dropout1d/2d/3d)
        """
        super().__init__()
        if p < 0 or p >= 1:
            raise ValueError("dropout probability has to be in [0, 1), " "but got {}".format(p))
        self.p = p
        self.tie = tie
        self.transposed = transposed
        self.binomial = torch.distributions.binomial.Binomial(probs=1-self.p)

    def forward(self, X):
        """X: (batch, dim, lengths...)."""
        if self.training:
            if not self.transposed: X = rearrange(X, 'b ... d -> b d ...')
            # binomial = torch.distributions.binomial.Binomial(probs=1-self.p) # This is incredibly slow because of CPU -> GPU copying
            mask_shape = X.shape[:2] + (1,)*(X.ndim-2) if self.tie else X.shape
            # mask = self.binomial.sample(mask_shape)
            mask = torch.rand(*mask_shape, device=X.device) < 1.-self.p
            X = X * mask * (1.0/(1-self.p))
            if not self.transposed: X = rearrange(X, 'b d ... -> b ... d')
            return X
        return X



class S4Model(nn.Module):

    def __init__(
        self,
        d_input,
        d_output=10,
        d_model=256,
        d_state = 128,
        n_layers=4,
        dropout=0.2,
        prenorm=False,
        lr = 0.01,
        batch_size = 0,
        normalization="batchnorm",
        extra_features = None
    ):
        super().__init__()

        self.device = torch.device("cuda")
        self.pure_complex = extra_features["pure_complex"]
        self.extra_features = extra_features
        self.normalization = normalization
        dt_min = extra_features["dt_min"]
        dt_max = extra_features["dt_max"]
        activation = extra_features["activation"]
        self.premix = extra_features["premix"]
        self.mix = extra_features["mix"]
        self.residual1 = extra_features["residual1"]
        self.residual2 = extra_features["residual2"]
        self.drop2 = extra_features["drop2"]

        self.prenorm = prenorm

        # Linear encoder (d_input = 1 for grayscale and 3 for RGB)
        self.encoder = nn.Linear(d_input, d_model)

        # Stack S4 layers as residual blocks
        self.s4_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        for _ in range(n_layers):

            self.s4_layers.append(
                S4D_orig(d_model, d_state = d_state, dropout=dropout, transposed=True, lr = lr, pure_complex = self.pure_complex, dt_max = dt_max, dt_min = dt_min, activation = activation, premix = self.premix, mix = self.mix, residual1 = self.residual1)
            )
            if normalization == "batchnorm":
                self.norms.append(nn.BatchNorm1d(d_model, momentum=0.05))
            elif normalization == "layernorm":
                self.norms.append(nn.LayerNorm(d_model))

            
            self.dropouts.append(DropoutNd(dropout))

        # Linear decoder
        if extra_features["use_readout_layer"]:
            self.decoder = ReadoutLayer(
                input_size=d_model,
                hidden_size=d_output,
                batch_size=batch_size,
                dropout=dropout,
                normalization=True,
                use_bias=False,
                extra_features=extra_features
            )
        else:
            self.decoder = nn.Linear(d_model, d_output)
        self.is_snn = False

        if self.mix == "GLU":
            self.output_linear = nn.Sequential(
                nn.Conv1d(d_model, 2*d_model, kernel_size=1),
                nn.GLU(dim=-2),
            )
        elif self.mix == "Linear":
            self.output_linear = nn.Sequential(
                nn.Linear(d_model, d_model)
            )

        self.to(self.device)
        


    def forward(self, x):
        """
        Input x is shape (B, L, d_input)
        """
        x = x.to(self.device)
        x = self.encoder(x)  # (B, L, d_input) -> (B, L, d_model)

        x = x.transpose(-1, -2)  # (B, L, d_model) -> (B, d_model, L)
        for i, (layer, norm, dropout) in enumerate(zip(self.s4_layers, self.norms, self.dropouts)):
            # Each iteration of this loop will map (B, d_model, L) -> (B, d_model, L)

            z = x
            if not self.premix:
                if self.mix == "GLU":
                    z = self.output_linear(z)
                elif self.mix == "Linear":
                    z = self.output_linear(z.transpose(1, 2)).transpose(1, 2)

            if self.prenorm:
                # Prenorm
                if self.normalization == "batchnorm":
                    z = z.transpose(-1, -2) # (B, d_model, L) -> (B, L, d_model)
                    _z = norm(z.reshape(z.shape[0] * z.shape[1], z.shape[2]))
                    z = _z.reshape(z.shape[0], z.shape[1], z.shape[2]).transpose(-1, -2)
                elif self.normalization == "layernorm":
                    z = norm(z.transpose(-1, -2)).transpose(-1, -2)

            # Apply S4 block: we ignore the state input and output
            z, _ = layer(z)

            # Dropout on the output of the S4 block
            if self.drop2:
                z = dropout(z)

            # Residual connection
            if self.residual2:
                x = z + x
            else:
                x = z

            if not self.prenorm:
                # Postnorm
                if self.normalization == "batchnorm":
                    x = x.transpose(-1, -2) # (B, d_model, L) -> (B, L, d_model)
                    _x = norm(x.reshape(x.shape[0] * x.shape[1], x.shape[2]))
                    x = _x.reshape(x.shape[0], x.shape[1], x.shape[2]).transpose(-1, -2)
                elif self.normalization == "layernorm":
                    x = norm(x.transpose(-1, -2)).transpose(-1, -2)

        x = x.transpose(-1, -2)

        if self.extra_features["use_readout_layer"] == False:
            # Pooling: average pooling over the sequence length
            x = x.mean(dim=1)

        # Decode the outputs
        self.x = self.decoder(x)  # (B, d_model) -> (B, d_output)

        return self.x, 0


class S4DKernel(nn.Module):
    """Generate convolution kernel from diagonal SSM parameters."""

    def __init__(self, d_model, N=64, dt_min=0.001, dt_max=0.1, lr=None, pure_complex = None):
        super().__init__()
        self.device = torch.device("cuda")
        # Generate dt
        H = d_model
        log_dt = torch.rand(H).to(self.device) * (
            math.log(dt_max) - math.log(dt_min)
        ) + math.log(dt_min)

        C = torch.randn(H, N // 2, dtype=torch.cfloat).to(self.device)

        self.C = nn.Parameter(torch.view_as_real(C))
        self.register("log_dt", log_dt, lr)

        log_A_real = torch.log(0.5 * torch.ones(H, N//2)).to(self.device)
        if pure_complex:
            A_imag = math.pi * repeat(torch.arange(1,N//2+1), 'n -> h n', h=H).to(self.device)
        else:
            A_imag = math.pi * repeat(torch.arange(N//2), 'n -> h n', h=H).to(self.device)
        self.register("log_A_real", log_A_real, lr)
        self.register("A_imag", A_imag, lr)

    def forward(self, L):
        """
        returns: (..., c, L) where c is number of channels (default 1)
        """

        # Materialize parameters
        dt = torch.exp(self.log_dt) # (H)
        C = torch.view_as_complex(self.C) # (H N)
        A = -torch.exp(self.log_A_real) + 1j * self.A_imag # (H N)

        # Vandermonde multiplication
        dtA = A * dt.unsqueeze(-1)  # (H N)
        K = dtA.unsqueeze(-1) * torch.arange(L, device=self.device) # (H N L)
        C = C * (torch.exp(dtA)-1.) / A
        K = 2 * torch.einsum('hn, hnl -> hl', C, torch.exp(K)).real

        return K

    def register(self, name, tensor, lr=None):
        """Register a tensor with a configurable learning rate and 0 weight decay"""

        if lr == 0.0:
            self.register_buffer(name, tensor)
        else:
            self.register_parameter(name, nn.Parameter(tensor))

            optim = {"weight_decay": 0.0}
            if lr is not None: optim["lr"] = lr
            setattr(getattr(self, name), "_optim", optim)


class S4D_orig(nn.Module):
    def __init__(self, d_model, d_state=64, dropout=0.0, transposed=True, lr = None, dt_min = 0, dt_max = 0, activation = "GELU", premix = False, mix = "GLU", residual1 = True, **kernel_args):
        super().__init__()

        self.h = d_model
        self.n = d_state
        self.d_output = self.h
        self.transposed = transposed

        self.D = nn.Parameter(torch.randn(self.h))

        # SSM Kernel
        self.kernel = S4DKernel(self.h, N=self.n, dt_min=dt_min, dt_max=dt_max, **kernel_args)
        self.spike_fct = SpikeFunctionBoxcar.apply
        self.activation = activation
        # Pointwise
        if activation == "GELU":
            self.activation = nn.GELU()
        elif activation == "step":
            self.activation = self.spike_fct
        # dropout_fn = nn.Dropout2d # NOTE: bugged in PyTorch 1.11
        dropout_fn = DropoutNd
        self.dropout = dropout_fn(dropout) if dropout > 0.0 else nn.Identity()

        # position-wise output transform to mix features
        if mix == "GLU":
            self.output_linear = nn.Sequential(
                nn.Conv1d(self.h, 2*self.h, kernel_size=1),
                nn.GLU(dim=-2),
            )
        elif mix == "Linear":
            self.output_linear = nn.Sequential(
                nn.Linear(self.h, self.h)
            )

        self.premix = premix
        self.mix = mix
        self.residual1 = residual1

        self.device = torch.device("cuda")

    def forward(self, u, **kwargs): # absorbs return_output and transformer src mask
        """ Input and output shape (B, H, L) """
        if not self.transposed: u = u.transpose(-1, -2)
        L = u.size(-1)

        # Compute SSM Kernel
        k = self.kernel(L=L) # (H L)
        

        # Convolution
        k_f = torch.fft.rfft(k, n=2*L) # (H L)
        u_f = torch.fft.rfft(u, n=2*L) # (B H L)
        y = torch.fft.irfft(u_f*k_f, n=2*L)[..., :L] # (B H L)

        # Compute D term in state space equation - essentially a skip connection
        
        if self.residual1:
            y = y + u * self.D.unsqueeze(-1)
        if self.activation == "step":
            y = self.activation(y-self.thr)
        else:
            y = self.activation(y)
        y = self.dropout(y)
        if not self.premix:
            if self.mix == "GLU":
                y = self.output_linear(y)
            elif self.mix == "Linear":
                y = self.output_linear(y.transpose(1, 2)).transpose(1, 2)
        if not self.transposed: y = y.transpose(-1, -2)
        return y, None # Return a dummy state to satisfy this repo's interface, but this can be modified
