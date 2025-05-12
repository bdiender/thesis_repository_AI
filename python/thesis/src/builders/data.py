import random
from typing import Any, Dict, List, Tuple

import torch

from allennlp.data import Vocabulary
from allennlp.data.dataset_readers.dataset_reader import Instance
from allennlp.data.data_loaders import SimpleDataLoader

from dataset_readers import UniversalDependenciesReader


# def build_datasets(cfg: Dict[str, Any]) -> Tuple[List[Instance], List[Instance]]:
#     tb_name = cfg.dataset.name
#     train = list(UniversalDependenciesReader(split=cfg.dataset.splits.train).read(tb_name))
#     dev = list(UniversalDependenciesReader(split=cfg.dataset.splits.dev).read(tb_name))

#     return train, dev


def build_vocab() -> Vocabulary:
    return Vocabulary.from_files('master_vocab/')


def _reservoir_sample(stream, k, seed):
    random.seed(seed)
    it = iter(stream)
    reservoir = [next(it) for _ in range(k)]
    for i, inst in enumerate(it, start=k+1):
        j = random.randint(1, i)
        if j <= k:
            reservoir[j-1] = inst
    
    return reservoir


def build_data_loaders(vocab: Vocabulary, cfg: Dict[str, Any]) -> Tuple[SimpleDataLoader, SimpleDataLoader]:
    batch_size = cfg.training.batch_size

    reader = UniversalDependenciesReader(split=cfg.dataset.splits.train)
    stream = _reservoir_sample(reader.read(cfg.dataset.name), cfg.dataset.samples_train, cfg.seed)
    train_loader = SimpleDataLoader(stream, batch_size=batch_size, shuffle=True)

    reader = UniversalDependenciesReader(split=cfg.dataset.splits.dev)
    stream = _reservoir_sample(reader.read(cfg.dataset.name), cfg.dataset.samples_dev, cfg.seed)
    dev_loader = SimpleDataLoader(stream, batch_size=batch_size, shuffle=True)

    cuda_device = cfg.cuda_device if torch.cuda.is_available() else -1

    for loader in (train_loader, dev_loader):
        loader.index_with(vocab)
        if cuda_device >= 0:
            loader.set_target_device(torch.device(f'cuda:{cuda_device}'))
    
    return train_loader, dev_loader
