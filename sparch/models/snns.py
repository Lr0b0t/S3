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
        lif_feature = "_"
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

        self.lif_feature = lif_feature

        if neuron_type not in ["LIF", "adLIF", "LIFfeature", "adLIFnoClamp","LIFfeatureDim", "adLIFclamp", "RLIF", "RadLIF", "LIFcomplex","RLIFcomplex","RLIFcomplex1MinAlphaNoB","RLIFcomplex1MinAlpha", "LIFcomplex_gatedB", "LIFcomplex_gatedDt", "LIFcomplexDiscr"]:
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
        if self.neuron_type == "LIFfeature":
            # Hidden layers
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
                        lif_feature = self.lif_feature
                    )
                )
                input_size = self.layer_sizes[i] * (1 + self.bidirectional)
        else:
            # Hidden layers
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
                    )
                )
                input_size = self.layer_sizes[i] * (1 + self.bidirectional)

        # Readout layer
        if self.use_readout_layer:
            snn.append(
                ReadoutLayer(
                    input_size=input_size,
                    hidden_size=self.layer_sizes[-1],
                    batch_size=self.batch_size,
                    dropout=self.dropout,
                    normalization=self.normalization,
                    use_bias=self.use_bias,
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

            # Compute membrane potential (LIF)
            ut = alpha * (ut - st) + (1 - alpha) * Wx[:, t, :]

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
        lif_feature="_"
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
        self.lif_feature = lif_feature
        device = torch.device("cuda")
        if "1-200_1-5"  in lif_feature:
            self.alpha_lim = [np.exp(-1 / 5), np.exp(-1 / 200)]
        else:
            self.alpha_lim = [np.exp(-1 / 5), np.exp(-1 / 25)]
        dt = 1
        if "Dt1ms" in lif_feature:
            dt = 0.001
        elif "Dt1" in lif_feature:
            dt = 1
        if "dtParam" in lif_feature:            
            self.register("dt", torch.ones(1)*dt, lr=0.01)
        else:
            self.dt = dt
        
        dt_min = 0.01
        dt_max = 0.4

        if "dtLog" in lif_feature:
            log_dt = torch.rand(self.hidden_size)*(
            math.log(dt_max) - math.log(dt_min)
            ) + math.log(dt_min)
            self.register("log_dt", log_dt, lr=0.01)

        if  "logAlpha" in lif_feature:
            self.log_alpha = nn.Parameter(torch.Tensor(self.hidden_size))
            nn.init.uniform_(self.log_alpha,torch.log(torch.tensor(self.alpha_lim[0])), torch.log(torch.tensor(self.alpha_lim[1])))

        elif "cont" in lif_feature:
            if "A0_5" in lif_feature:
                log_log_alpha = torch.log(0.5 * torch.ones(self.hidden_size)).to(device)
                self.register("log_log_alpha", log_log_alpha, lr=0.01)
            elif "A0_5Const" in lif_feature:
                self.log_log_alpha = torch.log(0.5 * torch.ones(self.hidden_size)).to(device)
            else:
                self.log_log_alpha = nn.Parameter(torch.Tensor(self.hidden_size))
                nn.init.uniform_(self.log_log_alpha, torch.log(-torch.log(torch.tensor(self.alpha_lim[1]))/self.dt), torch.log(-torch.log(torch.tensor(self.alpha_lim[0]))/self.dt))
        
                        

        else:
            self.alpha = nn.Parameter(torch.Tensor(self.hidden_size))
            nn.init.uniform_(self.alpha, self.alpha_lim[0], self.alpha_lim[1])
            self.log_log_alpha = torch.ones(self.hidden_size).to(device)

        
        
        if "imag" in lif_feature:
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
        ut = torch.rand(Wx.shape[0], Wx.shape[2]).to(device)
        st = torch.rand(Wx.shape[0], Wx.shape[2]).to(device)
        s = []
        if "imag"  in self.lif_feature:
            eigenval = -torch.exp(self.log_log_alpha)+1j*self.alpha_img
        else:
            eigenval = -torch.exp(self.log_log_alpha) 

        if "dtLog"  in self.lif_feature:
            self.dt = torch.exp(self.log_dt)
            
        
        if "logAlpha" in self.lif_feature :
            alpha = torch.exp(self.log_alpha)
        elif "cont" in self.lif_feature:
            alpha = torch.exp(self.dt*eigenval)
        else:
            alpha = self.alpha
        if "NoClamp" not in self.lif_feature:
            alpha = torch.clamp(alpha, min=self.alpha_lim[0], max=self.alpha_lim[1])

        if "B" in  self.lif_feature:
            b = self.b
        else:
            b = (1 - alpha)

        # Loop over time axis
        for t in range(Wx.shape[1]):

            # Compute membrane potential (LIF)
            ut = alpha * (ut - st) + b * Wx[:, t, :]

            # Compute spikes with surrogate gradient
            if "imag"  in self.lif_feature:
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
        lif_feature="_"
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
        if "dim2"  in lif_feature:
            self.dim = 2
        self.b = nn.Parameter(torch.rand(self.hidden_size, self.dim)*0.5)

        # Trainable parameters
        self.W = nn.Linear(self.input_size, self.hidden_size, bias=use_bias)
        self.lif_feature = lif_feature
        device = torch.device("cuda")
        if "1-200_1-5"  in lif_feature:
            self.alpha_lim = [np.exp(-1 / 5), np.exp(-1 / 200)]
        else:
            self.alpha_lim = [np.exp(-1 / 5), np.exp(-1 / 25)]
        dt = 1
        if "Dt1ms" in lif_feature:
            dt = 0.001
        elif "Dt1" in lif_feature:
            dt = 1
        if "dtParam" in lif_feature:            
            self.register("dt", torch.ones(1)*dt, lr=0.01)
        else:
            self.dt = dt
        
        dt_min = 0.01
        dt_max = 0.4

        if "dtLog" in lif_feature:
            log_dt = torch.rand(self.hidden_size)*(
            math.log(dt_max) - math.log(dt_min)
            ) + math.log(dt_min)
            self.register("log_dt", log_dt, lr=0.01)

        if  "logAlpha" in lif_feature:
            self.log_alpha = nn.Parameter(torch.Tensor(self.hidden_size))
            nn.init.uniform_(self.log_alpha,torch.log(torch.tensor(self.alpha_lim[0])), torch.log(torch.tensor(self.alpha_lim[1])))

        elif "cont" in lif_feature:
            if "A0_5" in lif_feature:
                log_log_alpha = torch.log(0.5 * torch.ones(self.hidden_size)).to(device)
                self.register("log_log_alpha", log_log_alpha, lr=0.01)
            elif "A0_5Const" in lif_feature:
                self.log_log_alpha = torch.log(0.5 * torch.ones(self.hidden_size)).to(device)
            else:
                self.log_log_alpha = nn.Parameter(torch.Tensor(self.hidden_size, self.dim))
                nn.init.uniform_(self.log_log_alpha, torch.log(-torch.log(torch.tensor(self.alpha_lim[1]))/self.dt), torch.log(-torch.log(torch.tensor(self.alpha_lim[0]))/self.dt))
        
                        

        else:
            self.alpha = nn.Parameter(torch.Tensor(self.hidden_size))
            nn.init.uniform_(self.alpha, self.alpha_lim[0], self.alpha_lim[1])
            self.log_log_alpha = torch.ones(self.hidden_size).to(device)

        
        
        if "imag" in lif_feature:
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
        if "imag"  in self.lif_feature:
            eigenval = -torch.exp(self.log_log_alpha)+1j*self.alpha_img
        else:
            eigenval = -torch.exp(self.log_log_alpha) 

        if "dtLog"  in self.lif_feature:
            self.dt = torch.exp(self.log_dt)
            
        
        if "logAlpha" in self.lif_feature :
            alpha = torch.exp(self.log_alpha)
        elif "cont" in self.lif_feature:
            alpha = torch.exp(self.dt*eigenval)
        else:
            alpha = self.alpha

        if "B" in  self.lif_feature:
            b = self.b
        else:
            b = (1 - alpha)

        # Loop over time axis
        for t in range(Wx.shape[1]):

            # Compute membrane potential (LIF)
            ut = alpha * (ut -  st.unsqueeze(-1).expand(-1,-1, self.dim)) + self.b * Wx[:, t, :].unsqueeze(-1).expand(-1,-1, self.dim)

            # Compute spikes with surrogate gradient
            if "imag"  in self.lif_feature:
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

            # Compute potential (adLIF)
            wt = beta * wt + a * ut + b * st
            ut = alpha * (ut - st) + (1 - alpha)* (Wx[:, t, :] - wt)

            # Compute spikes with surrogate gradient
            st = self.spike_fct(ut - self.threshold)
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

            # Compute potential (adLIF)
            wt = beta * wt + a * ut + b * st
            ut = alpha * (ut - st) + (1 - alpha)* (Wx[:, t, :] - wt)

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

            # Compute potential (adLIF)
            wt = beta * wt + a * ut + b * st
            ut = alpha * (ut - st) + (1 - alpha)* (Wx[:, t, :] - wt)

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

        # Bound values of the neuron parameters to plausible ranges
        #log_log__alpha = torch.clamp(self.log_log_alpha, min=self.log_log_alpha_lim[0], max=self.log_log_alpha_lim[1])
        alpha = torch.exp((-torch.exp(self.log_log_alpha)+1j*self.alpha_img)*torch.exp(self.log_dt))
        # Loop over time axis
        for t in range(Wx.shape[1]):

            # Compute membrane potential (LIF)
            ut = alpha * (ut -st) + self.b * Wx[:, t, :]

            # Compute spikes with surrogate gradient
            st = self.spike_fct(2*ut.real - self.threshold)
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
            ut = alpha * (ut -st) + self.b * (Wx[:, t, :] + torch.matmul(st, V))

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

            # Compute membrane potential (RLIF)
            ut = alpha * (ut - st) + (1 - alpha) * (Wx[:, t, :] + torch.matmul(st, V))

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

            # Compute potential (RadLIF)
            wt = beta * wt + a * ut + b * st
            ut = alpha * (ut - st) + (1 - alpha) * (
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
        for t in range(Wx.shape[1]):

            # Compute potential (LIF)
            ut = alpha * ut + (1 - alpha) * Wx[:, t, :]
            out = out + F.softmax(ut, dim=1)

        return out