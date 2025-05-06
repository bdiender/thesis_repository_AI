from typing import Any, Dict, List, Tuple

from allennlp.data import Vocabulary
from allennlp.data.dataset_readers.dataset_reader import Instance
from allennlp.data.data_loaders import SimpleDataLoader

from dataset_readers import UniversalDependenciesReader


def build_datasets(cfg: Dict[str, Any]) -> Tuple[List[Instance], List[Instance]]:
    tb_name = cfg.dataset.name
    train = list(UniversalDependenciesReader(split=cfg.dataset.splits.train).read(tb_name))
    dev = list(UniversalDependenciesReader(split=cfg.dataset.splits.dev).read(tb_name))

    return train, dev


def build_vocab(train: List[Instance], dev: List[Instance]) -> Vocabulary:
    return Vocabulary.from_instances(train + dev)


def build_data_loaders(train: List[Instance], dev: List[Instance],
                      vocab: Vocabulary, cfg: Dict[str, Any]) -> Tuple[SimpleDataLoader, SimpleDataLoader]:
    batch_size = cfg.dataset.load_batch_size
    train_loader = SimpleDataLoader(train, batch_size=batch_size, shuffle=True)
    dev_loader = SimpleDataLoader(dev, batch_size=batch_size, shuffle=False)

    for loader in (train_loader, dev_loader):
        loader.index_with(vocab)
    
    return train_loader, dev_loader
