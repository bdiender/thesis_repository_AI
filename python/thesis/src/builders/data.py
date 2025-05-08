from typing import Any, Dict, List, Tuple

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



def build_data_loaders(vocab: Vocabulary, cfg: Dict[str, Any]) -> Tuple[SimpleDataLoader, SimpleDataLoader]:
    batch_size = cfg.training.batch_size
    train_loader = SimpleDataLoader.from_dataset_reader(
        UniversalDependenciesReader(split=cfg.dataset.splits.train),
        batch_size=batch_size,
        shuffle=True
    )
    dev_loader = SimpleDataLoader.from_dataset_reader(
        UniversalDependenciesReader(split=cfg.dataset.splits.train),
        batch_size=batch_size,
        shuffle=True
    )

    for loader in (train_loader, dev_loader):
        loader.index_with(vocab)
    
    return train_loader, dev_loader
