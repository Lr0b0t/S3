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
This is where the parser for the model configuration is defined.
"""
import logging
from distutils.util import strtobool

logger = logging.getLogger(__name__)


def add_model_options(parser):
    parser.add_argument(
        "--model_type",
        nargs='+',
        type=str,
        choices=["LIF", "LIFfeature", "adLIFnoClamp", "LIFfeatureDim", "adLIF", "CadLIF", "RSEadLIF", "RLIF", "RadLIF", "MLP", "RNN", "LiGRU", "GRU", "LIFcomplex", "LIFrealcomplex", "ReLULIFcomplex", "RLIFcomplex","RLIFcomplex1MinAlpha", "adLIFclamp", "RLIFcomplex1MinAlphaNoB","LIFcomplex_gatedB", "LIFcomplex_gatedDt", "LIFcomplexDiscr", "BRF", "ResonateFire"],
        default=["LIF"],
        help="Type of ANN or SNN model.",
    )
    parser.add_argument(
        "--s4",
        type=lambda x: bool(strtobool(str(x))),
        default=False,
        help="Whether to include trainable bias with feedforward weights.",
    )
    parser.add_argument(
        "--recurrent",
        type=lambda x: bool(strtobool(str(x))),
        default=False,
        help="Whether to include trainable bias with feedforward weights.",
    )

    parser.add_argument(
        "--pure_complex",
        nargs='+',
        type=lambda x: bool(strtobool(str(x))),
        default=[False],
        help="Whether to include trainable bias with feedforward weights.",
    )
    
    parser.add_argument(
        "--nb_state",
        nargs='+',
        type=int,
        default=[64],
        help="Number of neurons in all hidden layers.",
    )
    
    parser.add_argument(
        "--lif_feature",
        type=str,
        choices=["logAlpha", "cont", "1-200_1-5", "A0_5", "dtParam", "A0_5Const", "dtLog", "Dt1ms", "Dt1", "alphaConst", "imag", "NoClamp", "B", "dim2"],
        default=None,
        nargs='+',
        help="Feature of LIF",
    )
    parser.add_argument(
        "--half_reset",
        type=lambda x: bool(strtobool(str(x))),
        default=True,
        help="Whether to include trainable bias with feedforward weights.",
    )
    parser.add_argument(
        "--no_reset",
        nargs='+',
        type=lambda x: bool(strtobool(str(x))),
        default=[False],
        help="Whether to include trainable bias with feedforward weights.",
    )
    parser.add_argument(
        "--activation",
        nargs='+',
        type=str,
        choices=["step", "GELU"],
        default=["GELU"],
        help="activation",
    )
    parser.add_argument(
        "--mix",
        nargs='+',
        type=str,
        choices=["GLU", "Linear"],
        default=["GLU"],
        help="mix",
    )
    parser.add_argument(
        "--reset",
        nargs='+',
        type=str,
        choices=["no_reset", "half_reset"],
        default=["half_reset"],
        help="mix",
    )
    parser.add_argument(
        "--bRand",
        nargs='+',
        type=str,
        choices=["Rand", "RandN"],
        default=["Rand"],
        help="mix",
    )
    parser.add_argument(
        "--s_GLU",
        nargs='+',
        type=lambda x: bool(strtobool(str(x))),
        default=[False],
        help="Whether to include trainable bias with feedforward weights.",
    )
    parser.add_argument(
        "--superspike",
        nargs='+',
        type=bool,
        default=[False],
        help="Use Superspike surrogate gradient. False by default",
    )
    parser.add_argument(
        "--slayer",
        nargs='+',
        type=bool,
        default=[False],
        help="Use SLAYER surrogate gradient. False by default",
    )
    parser.add_argument(
        "--xavier_init",
        nargs='+',
        type=bool,
        default=[False],
        help="Use Xavier init as initialization for weights. False by default",
    )
    parser.add_argument(
        "--shifted_relu",
        action= 'store_true',
        help="Use threshold shift for ReLULIFcomp model",
    )
    parser.add_argument(
        "--residual",
        nargs='+',
        type=bool,
        default=[False],
        help="Use residual connections in all SNNs. False by default",
    )
    parser.add_argument(
        "--rst_detach",
        nargs='+',
        type=bool,
        default=[False],
        help="Detach reset signal specifically for autograd. True by default",
    )
    parser.add_argument(
        "--dt_min",
        type=float,
        default=[0.01],
        nargs='+',
        help="Min dt initialization for LIFcomplex",
    )
    parser.add_argument(
        "--dt_max",
        type=float,
        default=[0.7],
        nargs='+',
        help="Max dt initializationfor LIFcomplex ",
    )
    parser.add_argument(
        "--thr",
        type=float,
        default=[1.0],
        nargs='+',
        help="Max dt initializationfor LIFcomplex ",
    )
    parser.add_argument(
        "--c_discr",
        nargs='+',
        type=lambda x: bool(strtobool(str(x))),
        default=[False],
        help="Whether to include trainable bias with feedforward weights.",
    )
    parser.add_argument(
        "--c_param",
        nargs='+',
        type=lambda x: bool(strtobool(str(x))),
        default=[False],
        help="Whether to include trainable bias with feedforward weights.",
    )
    parser.add_argument(
        "--nb_layers",
        nargs='+',
        type=int,
        default=[3],
        help="Number of layers (including readout layer).",
    )

    parser.add_argument(
        "--nb_hiddens",
        nargs='+',
        type=int,
        default=[128],
        help="Number of neurons in all hidden layers.",
    )
    parser.add_argument(
        "--alpha_imag",
        nargs='+',
        type=float,
        default=[3.14],
        help="Number of neurons in all hidden layers.",
    )
    parser.add_argument(
        "--alpha_range",
        type=lambda x: bool(strtobool(str(x))),
        default=False,
        help="Whether to include trainable bias with feedforward weights.",
    )
    parser.add_argument(
        "--alpha_rand",
        nargs='+',
        type=str,
        choices=["Rand", "RandN", "False"],
        default=["False"],
        help="mix",
    )
    parser.add_argument(
        "--alphaRe_rand",
        nargs='+',
        type=str,
        choices=["Rand", "RandN", "False"],
        default=["False"],
        help="mix",
    )
    parser.add_argument(
        "--alpha",
        nargs='+',
        type=float,
        default=[0.5],
        help="mix",
    )
    parser.add_argument(
        "--pdrop",
        nargs='+',
        type=float,
        default=[0.1],
        help="Dropout rate, must be between 0 and 1.",
    )
    parser.add_argument(
        "--normalization",
        type=str,
        nargs='+',
        default=["batchnorm"],
        help="Type of normalization, Every string different from batchnorm "
        "and layernorm will result in no normalization.",
    )
    parser.add_argument(
        "--use_bias",
        type=lambda x: bool(strtobool(str(x))),
        default=False,
        help="Whether to include trainable bias with feedforward weights.",
    )
    parser.add_argument(
        "--drop2",
        nargs='+',
        type=lambda x: bool(strtobool(str(x))),
        default=[False],
        help="Whether to include trainable bias with feedforward weights.",
    )
    parser.add_argument(
        "--use_readout_layer",
        nargs='+',
        type=lambda x: bool(strtobool(str(x))),
        default=[False],
        help="Whether to include trainable bias with feedforward weights.",
    )
    parser.add_argument(
        "--prenorm",
        nargs='+',
        type=lambda x: bool(strtobool(str(x))),
        default=[False],
        help="Whether to include trainable bias with feedforward weights.",
    )
    parser.add_argument(
        "--premix",
        nargs='+',
        type=lambda x: bool(strtobool(str(x))),
        default=[False],
        help="Whether to include trainable bias with feedforward weights.",
    )
    parser.add_argument(
        "--residual1",
        nargs='+',
        type=lambda x: bool(strtobool(str(x))),
        default=[True],
        help="Whether to include trainable bias with feedforward weights.",
    )
    parser.add_argument(
        "--residual2",
        nargs='+',
        type=lambda x: bool(strtobool(str(x))),
        default=[True],
        help="Whether to include trainable bias with feedforward weights.",
    )
    parser.add_argument(
        "--bidirectional",
        type=lambda x: bool(strtobool(str(x))),
        default=False,
        help="If True, a bidirectional model that scans the sequence in both "
        "directions is used, which doubles the size of feedforward matrices. ",
    )
    return parser


def print_model_options(config):
    logging.info(
        """
        Model Config
        ------------
        Model Type: {model_type}
        Number of layers: {nb_layers}
        Number of hidden neurons: {nb_hiddens}
        Dropout rate: {pdrop}
        Normalization: {normalization}
        Use bias: {use_bias}
        Bidirectional: {bidirectional}
    """.format(
            **config
        )
    )
