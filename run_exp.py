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
This is the script used to run experiments.
"""
import argparse
import logging

from sparch.exp import Experiment
from sparch.parsers.model_config import add_model_options
from sparch.parsers.training_config import add_training_options

import wandb

logger = logging.getLogger(__name__)


def parse_args():

    parser = argparse.ArgumentParser(
        description="Model training on spiking speech commands datasets."
    )
    parser = add_model_options(parser)
    parser = add_training_options(parser)
    args = parser.parse_args()

    return args


args = parse_args()

DEBUG = args.debug

LIF_FEAT_DICT = {}
if args.lif_feature:
    for feat in args.lif_feature:
        LIF_FEAT_DICT[feat] = True

delattr(args, 'lif_feature')

def main(debug_config=None):
    """
    Runs model training/testing using the configuration specified
    by the parser arguments. Run `python run_exp.py -h` for details.
    """

    if DEBUG:
        config = debug_config
        config['lif_feature'] = LIF_FEAT_DICT
    else:
        wandb.init()
        config = wandb.config
        config['lif_feature'] = LIF_FEAT_DICT

    # Instantiate class for the desired experiment
    experiment = Experiment(config)

    # Run experiment
    experiment.forward()


if __name__ == "__main__":

    # Get experiment configuration from parser


    sweep_config = {
        'method': 'grid',
        'metric': {
            'name': "valid acc",
            'goal': 'minimize'
        },
        'parameters': {
            },
    }

    if args.sweep:
        sweep_name = args.sweep
        sweep_config['name'] = sweep_name
    delattr(args, 'sweep')

    debug_config = {'seed':42}
    for arg, value in vars(args).items():
        if isinstance(value, list):
            sweep_config['parameters'][arg] = {'values': value}
            debug_config[arg] = value[0]
        else:
            sweep_config['parameters'][arg] = {'values': [value]}
            debug_config[arg] = value

    # Add LIF features to wandb config for tracking
    for feat in LIF_FEAT_DICT:
        sweep_config['parameters'][feat] = {'values': [True]}

    if DEBUG:
        main(debug_config)
    else:
        if args.dataset_name == "shd":
            project_name="S3_SHD_runs"
        elif args.dataset_name == "ssc":
            project_name="S3_SSC_runs"
        elif args.dataset_name == "sc":
            project_name="S3_SC_runs"
        elif args.dataset_name == "hd":
            project_name="S3_HD_runs"


        sweep_id = wandb.sweep(sweep_config,
                            entity="maximes_crew", 
                            project=project_name) 
        # Run the sweep
        wandb.agent(sweep_id, function=main)
