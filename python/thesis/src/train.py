import argparse
import numpy as np
import random
import torch
import torch.backends

from builders.data import make_datasets, build_vocab, make_data_loaders
from builders.model import build_model
from builders.trainer import build_trainer
from config.loader import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        required=True,
        help='Path to YAML config file.'
    )
    args = parser.parse_args()

    cfg = load_config('configs/config.yaml', args.config)
    s = cfg.get('seed', None)

    if s is not None:
        random.seed(s)
        np.random.seed(s)
        torch.manual_seed(s)
        torch.cuda.manual_seed(s)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    cuda_device = cfg['cuda_device'] if torch.cuda.is_available() else -1

    train_ds, dev_ds = make_datasets(cfg)
    vocab = build_vocab(train_ds, dev_ds)
    train_loader, dev_loader = make_data_loaders(train_ds, dev_ds, vocab, cfg)

    model = build_model(cfg, vocab, cuda_device=cuda_device)
    trainer = build_trainer(model, train_loader, dev_loader, cfg)
    trainer.train()

if __name__ == "__main__":
    main()
