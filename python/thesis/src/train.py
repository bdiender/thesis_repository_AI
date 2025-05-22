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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        required=True,
        help='Key of configuration settings for run.'
    )
    args, unknown = parser.parse_known_args()

    cfg = GLOBAL_CONFIG.get(args.config)

    for override in unknown:
        if override.startswith('--') and '=' in override:
            cfg.set(*override[2:].split('=', 1))

    in_sweep = False

    if cfg.enable_wandb:
        import wandb

        run = wandb.init(
            project=cfg.wb.project,
            entity=cfg.wb.entity,
            config=cfg.to_dict(),
            name=f'{args.config}-{dt.now():%Y%m%d-%H%M%S}'
        )

        in_sweep = (run.sweep_id is not None)

        if in_sweep:
            for k, v in run.config.items():
                if k in ('project', 'entity'):
                    continue
                cfg.set(k, str(v))

    # Make output dir
    if not in_sweep:
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

    cuda_device = cfg.cuda_device if torch.cuda.is_available() else -1

    # Load data, vocab, data loaders
    vocab = build_vocab()
    train_loader, dev_loader = build_data_loaders(vocab, cfg)
    for loader in (train_loader, dev_loader):
         loader.index_with(vocab)
         if cuda_device >= 0:
             loader.set_target_device(torch.device(f'cuda:{cuda_device}'))

    if not in_sweep:
        with open(os.path.join(cfg.output_dir, 'config.json'), 'w') as f:
            cfg_dict = cfg.to_dict()
            updates_per_epoch = math.ceil(len(train_loader) / cfg.training.gradient_accumulation_steps)
            cfg_dict['training']['epochs'] = math.ceil(cfg.training.total_updates / updates_per_epoch)
            json.dump(cfg_dict, f, indent=2)

    # Build model
    model = build_model(cfg, vocab, cuda_device=cuda_device)
    if cuda_device >= 0:
        model = model.cuda(cuda_device)

    # Build and apply trainer
    trainer = build_trainer(model, train_loader, dev_loader, cfg, cuda_device, in_sweep)
    if in_sweep:
        trainer._load_model_state = lambda *args, **kwargs: None

    metrics = trainer.train()

    if cfg.enable_wandb:
        wandb.log(metrics)
        run.finish()

    # Save model weights, vocabulary, model, and metrics
    if not in_sweep:
        torch.save(model.state_dict(), os.path.join(cfg.output_dir, 'weights.th'))
        vocab.save_to_files(os.path.join(cfg.output_dir, 'vocabulary'))

        with open(os.path.join(cfg.output_dir, 'model.pkl'), 'wb') as f:
            dill.dump(model, f)

        with open(os.path.join(cfg.output_dir, 'metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=2)

    
if __name__ == "__main__":
    main()
