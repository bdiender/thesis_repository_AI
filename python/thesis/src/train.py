import argparse
import torch

from builders.data import make_datasets, build_vocab, make_data_loaders
from builders.model import build_model
from builders.optimizer import build_optimizer
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

    cfg = load_config(args.config)

    cuda_device = cfg['cuda_device'] if torch.cuda.is_available() else -1

    train_ds, dev_ds = make_datasets(cfg)
    vocab = build_vocab(train_ds, dev_ds)
    train_loader, dev_loader = make_data_loaders(train_ds, dev_ds, vocab, cfg)

    model = build_model(vocab, cfg, cuda_device=cuda_device)
    trainer = build_trainer(model, train_loader, dev_loader, cfg)
    trainer.train()

if __name__ == "__main__":
    main()
