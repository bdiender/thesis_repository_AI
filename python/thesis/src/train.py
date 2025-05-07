import argparse
import json
import os
import random
import dill

import numpy as np
import torch
import torch.backends

from builders import (
    build_datasets,
    build_vocab,
    build_data_loaders,
    build_model,
    build_trainer
)
from config import GLOBAL_CONFIG


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        required=True,
        help='Key of configuration settings for run.'
    )
    args = parser.parse_args()

    cfg = GLOBAL_CONFIG.get(args.config)

    # Set random seed
    s = cfg.get('seed', None)
    if s is not None:
        random.seed(s)
        np.random.seed(s)
        torch.manual_seed(s)
        torch.cuda.manual_seed(s)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    cuda_device = cfg.cuda_device if torch.cuda.is_available() else -1

    # Load data, vocab, data loaders
    train_ds, dev_ds = build_datasets(cfg)
    vocab = build_vocab(train_ds, dev_ds)
    train_loader, dev_loader = build_data_loaders(train_ds, dev_ds, vocab, cfg)

    # Build model
    model = build_model(cfg, vocab, cuda_device=cuda_device)
    if cuda_device >= 0:
        model = model.cuda(cuda_device)

    # Make output dir
    os.makedirs(cfg.output_dir, exist_ok=True)

    # Build and apply trainer
    trainer = build_trainer(model, train_loader, dev_loader, cfg, cuda_device)
    metrics = trainer.train()

    # Save model weights, vocabulary, model, and metrics
    torch.save(model.state_dict(), os.path.join(cfg.output_dir, 'weights.th'))
    vocab.save_to_files(os.path.join(cfg.output_dir, 'vocabulary'))

    with open(os.path.join(cfg.output_dir, 'model.pkl'), 'wb') as f:
        dill.dump(model, f)

    with open(os.path.join(cfg.output_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    
if __name__ == "__main__":
    main()
