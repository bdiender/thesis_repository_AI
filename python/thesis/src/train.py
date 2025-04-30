import argparse
import numpy as np
import os
import random
import torch
import torch.backends

from allennlp.common.params import Params

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
    if cuda_device >= 0:
        model = model.cuda(cuda_device)

    trainer = build_trainer(model, train_loader, dev_loader, cfg, cuda_device)
    trainer.train()

    # torch.save(model.state_dict(), os.path.join(cfg['training']['output_dir'], 'weights.th'))
    # vocab.save_to_files(os.path.join(cfg['training']['output_dir'], 'vocabulary'))
    # Params.from_file(args.config)

if __name__ == "__main__":
    main()
