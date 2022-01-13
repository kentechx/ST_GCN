import argparse
import yaml
import os

def get_parser():
    parser = argparse.ArgumentParser(description='Tooth Classification')
    parser.add_argument("--config", type=str, default="config/teethgnn_dental_data.yaml", help="path to config file")
    parser.add_argument("--gpu", type=str, default="0", help="the index of gpus, e.x. `0` or `0,1`")

    args_cfg = parser.parse_args()
    with open(args_cfg.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    for key in config:
        for k, v in config[key].items():
            setattr(args_cfg, k, v)

    return args_cfg

def get_args(config):
    parser = argparse.ArgumentParser(description='Tooth Classification')

    args_cfg = parser.parse_args()
    with open(config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    for key in config:
        for k, v in config[key].items():
            setattr(args_cfg, k, v)

    return args_cfg
