import argparse
import dill
import json
import math
import os
import random

import numpy as np
import torch
import torch.backends

from builders import (
    build_vocab,
    build_data_loaders,
    build_model,
    build_trainer
)
from config import GLOBAL_CONFIG

from datetime import datetime as dt

def main():
    now = lambda: dt.now().strftime('%H:%M:%S')
    
    def update(update: str):
        print(f'[{now()}] {update}')

    update('Starting!')
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        required=True,
        help='Key of configuration settings for run.'
    )
    args, unknown = parser.parse_known_args()
    update('Got arguments')

    cfg = GLOBAL_CONFIG.get(args.config)

    for override in unknown:
        if override.startswith('--') and '=' in override:
            cfg.set(*override[2:].split('=', 1))
    
    update('Updated config')

    # Make output dir
    os.makedirs(cfg.output_dir, exist_ok=True)

    # Set random seed
    s = cfg.get('seed', None)
    if s is not None:
        random.seed(s)
        np.random.seed(s)
        torch.manual_seed(s)
        torch.cuda.manual_seed(s)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    update('Set random seed')

    cuda_device = cfg.cuda_device if torch.cuda.is_available() else -1
    update('Set device')

    # Load data, vocab, data loaders
    vocab = build_vocab()
    update('Built vocab')
    train_loader, dev_loader = build_data_loaders(vocab, cfg)
    update('Built loaders')
    for loader in (train_loader, dev_loader):
         loader.index_with(vocab)
         update('Indexed a loader with vocab')
         if cuda_device >= 0:
             loader.set_target_device(torch.device(f'cuda:{cuda_device}'))

    with open(os.path.join(cfg.output_dir, 'config.json'), 'w') as f:
        cfg_dict = cfg.to_dict()
        cfg_dict['training']['epochs'] = math.ceil(cfg.training.total_updates / len(train_loader))
        json.dump(cfg_dict, f, indent=2)

    # Build model
    model = build_model(cfg, vocab, cuda_device=cuda_device)
    if cuda_device >= 0:
        model = model.cuda(cuda_device)

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
