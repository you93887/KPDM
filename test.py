import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import numpy as np
import time
import os.path as op
import argparse

from datasets import build_dataloader
from processor.processor import do_inference
from utils.checkpoint import Checkpointer
from utils.logger import setup_logger
from model import build_model
from utils.iotools import load_train_configs


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="RDE Text-Image ReID Inference")
    parser.add_argument("--config_file", type=str, required=True,
                        help="Path to config YAML file")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model checkpoint (.pth). Default: <config_dir>/best.pth")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for logs. Default: same as config dir")
    args = parser.parse_args()

    # Load config
    cfg = load_train_configs(args.config_file)
    cfg.training = False

    # Determine output directory
    if args.output_dir:
        cfg.output_dir = args.output_dir
    else:
        cfg.output_dir = op.dirname(op.abspath(args.config_file))

    # Determine checkpoint path
    if args.checkpoint:
        checkpoint_path = args.checkpoint
    else:
        checkpoint_path = op.join(cfg.output_dir, 'best.pth')

    logger = setup_logger('RDE', save_dir=cfg.output_dir, if_train=False)
    logger.info(cfg)
    logger.info(f"Checkpoint: {checkpoint_path}")

    device = "cuda"
    test_img_loader, test_txt_loader, num_classes = build_dataloader(cfg)

    if not op.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = build_model(cfg, num_classes)
    checkpointer = Checkpointer(model)
    checkpointer.load(f=checkpoint_path)
    model = model.cuda()
    do_inference(model, test_img_loader, test_txt_loader)
