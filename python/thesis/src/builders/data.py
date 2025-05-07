from typing import Any, Dict, List, Tuple

from allennlp.data import Vocabulary
from allennlp.data.dataset_readers.dataset_reader import Instance
from allennlp.data.data_loaders import SimpleDataLoader

from dataset_readers import UniversalDependenciesReader


UD_RELATION_LIST = [  # https://universaldependencies.org/u/dep/
    "acl", "acl:relcl",
    "advcl", "advcl:relcl",
    "advmod", "advmod:emph", "advmod:lmod",
    "amod",
    "appos",
    "aux", "aux:pass",
    "case",
    "cc", "cc:preconj",
    "ccomp",
    "clf",
    "compound", "compound:lvc", "compound:prt", "compound:redup", "compound:svc",
    "conj",
    "cop",
    "csubj", "csubj:outer", "csubj:pass",
    "dep",
    "det", "det:numgov", "det:nummod", "det:poss",
    "discourse",
    "dislocated",
    "expl", "expl:impers", "expl:pass", "expl:pv",
    "fixed",
    "flat", "flat:foreign", "flat:name",
    "goeswith",
    "iobj",
    "list",
    "mark",
    "nmod", "nmod:poss", "nmod:tmod",
    "nsubj", "nsubj:outer", "nsubj:pass",
    "nummod", "nummod:gov",
    "obj",
    "obl", "obl:agent", "obl:arg", "obl:lmod", "obl:tmod",
    "orphan",
    "parataxis",
    "punct",
    "reparandum",
    "root",
    "vocative",
    "xcomp",
    "@@UNKNOWN@@"
]


def build_datasets(cfg: Dict[str, Any]) -> Tuple[List[Instance], List[Instance]]:
    tb_name = cfg.dataset.name
    train = list(UniversalDependenciesReader(split=cfg.dataset.splits.train).read(tb_name))
    dev = list(UniversalDependenciesReader(split=cfg.dataset.splits.dev).read(tb_name))

    return train, dev


def build_vocab(train: List[Instance], dev: List[Instance]) -> Vocabulary:
    vocab = Vocabulary()

    for rel in UD_RELATION_LIST:
        vocab.add_token_to_namespace(rel, namespace='head_tags')
    
    vocab.extend_from_instances(train + dev)

    return vocab



def build_data_loaders(train: List[Instance], dev: List[Instance],
                      vocab: Vocabulary, cfg: Dict[str, Any]) -> Tuple[SimpleDataLoader, SimpleDataLoader]:
    batch_size = cfg.training.batch_size
    train_loader = SimpleDataLoader(train, batch_size=batch_size, shuffle=True)
    dev_loader = SimpleDataLoader(dev, batch_size=batch_size, shuffle=False)

    for loader in (train_loader, dev_loader):
        loader.index_with(vocab)
    
    return train_loader, dev_loader
