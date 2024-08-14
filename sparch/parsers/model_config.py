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
        type=str,
        choices=["LIF", "LIFfeature", "adLIFnoClamp", "LIFfeatureDim", "adLIF", "RLIF", "RadLIF", "MLP", "RNN", "LiGRU", "GRU", "LIFcomplex", "RLIFcomplex","RLIFcomplex1MinAlpha", "adLIFclamp", "RLIFcomplex1MinAlphaNoB","LIFcomplex_gatedB", "LIFcomplex_gatedDt", "LIFcomplexDiscr"],
        default="LIF",
        help="Type of ANN or SNN model.",
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
        type=bool,
        default=True,
        help="Use half reset for LIFcomplex and RLIFcomplex models. True by default",
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
        default=[0.4],
        nargs='+',
        help="Max dt initializationfor LIFcomplex ",
    )
    parser.add_argument(
        "--nb_layers",
        type=int,
        default=3,
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
        "--pdrop",
        type=float,
        default=0.1,
        help="Dropout rate, must be between 0 and 1.",
    )
    parser.add_argument(
        "--normalization",
        type=str,
        default="batchnorm",
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
