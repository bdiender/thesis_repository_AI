"""
Adapted from: https://github.com/Hyperparticle/udify/blob/master/udify/dataset_readers/universal_dependencies.py
"""

from allennlp.data.dataset_readers.dataset_reader import DatasetReader, Instance
from allennlp.data.fields import Field, TextField, SequenceLabelField, MetadataField
from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.token_indexers.pretrained_transformer_indexer import PretrainedTransformerIndexer
from allennlp.data.tokenizers import Token
from allennlp.data.tokenizers.tokenizer import Tokenizer
from allennlp.data.tokenizers.pretrained_transformer_tokenizer import PretrainedTransformerTokenizer

from conllu import parse_incr
from datasets import load_dataset
from overrides import overrides
from typing import Any, Callable, Dict, Iterable, List, Tuple

from dataset_readers.reader_utils import process_multiword_tokens, gen_lemma_rule

import os

@DatasetReader.register("ud_reader")
class UniversalDependenciesReader(DatasetReader):
    def __init__(
        self,
        tokenizer: Tokenizer = None,
        token_indexers: Dict[str, TokenIndexer] = None,
        split: str = "train"
    ):
        super().__init__()
        self.split = split
        self.tokenizer = tokenizer or PretrainedTransformerTokenizer(
            model_name="bert-base-multilingual-cased",
        )

        self._token_indexers = token_indexers or {
            "tokens": PretrainedTransformerIndexer(
                model_name="bert-base-multilingual-cased"
            )
        }
    
    @overrides(check_signature=False)
    def _read(self, file_path: str) -> Iterable[Instance]:
        if os.path.isfile(file_path) and file_path.endswith(".conllu"):
            tb = parse_incr(open(file_path, encoding="utf-8"))
            local = True
        else:
            tb = load_dataset("universal_dependencies", file_path, split=self.split)
            local = False

        for example in tb:
            annotation = self._restructure(example, local)
            annotation = process_multiword_tokens(annotation)
            annotation = [t for t in annotation if t["id"] is not None]
            if not annotation:
                continue

            heads = [t["head"] for t in annotation]
            rels = [t["deprel"] for t in annotation]
            deps = list(zip(rels, heads))
            words = [t["form"] for t in annotation]
            ids = [t["id"] for t in annotation]
            mw_ids = [t["multi_id"] for t in annotation if t["multi_id"]]
            mw_forms = [t["form"] for t in annotation if t["multi_id"]]

            yield self.text_to_instance({
                "words": words,
                "dependencies": deps,
                "ids": ids,
                "multiword_ids": mw_ids,
                "multiword_forms": mw_forms
            })


    @overrides
    def text_to_instance(self, *inputs) -> Instance:
        if len(inputs) != 1 or not isinstance(inputs[0], dict):
            raise ValueError("text_to_instance expects a single dictionary input")
        
        inputs = inputs[0]
        words: List[str] = inputs['words']
        # lemmas = inputs.get('lemmas')
        # lemma_rules = inputs.get('lemma_rules')
        # upos_tags = inputs.get('upos_tags')
        # xpos_tags = inputs.get('xpos_tags')
        # feats = inputs.get('feats')
        dependencies: List[Tuple[str, int]] = inputs.get('dependencies')
        ids: List[str] = inputs.get('ids')
        multiword_ids: List[str] = inputs.get('multiword_ids')
        multiword_forms: List[str] = inputs.get('multiword_forms')
        
        fields: Dict[str, Field] = {}

        tokens = TextField([Token(w) for w in words], self._token_indexers)
        fields["tokens"] = tokens

        if dependencies is not None:
            fields["head_tags"] = SequenceLabelField([x[0] for x in dependencies],
                                                     tokens,
                                                     label_namespace="head_tags")
            fields["head_indices"] = SequenceLabelField([int(x[1]) for x in dependencies],
                                                        tokens,
                                                        label_namespace="head_index_tags")

        fields["metadata"] = MetadataField({
            "words": words,
            # "upos_tags": upos_tags,
            # "xpos_tags": xpos_tags,
            # "feats": feats,
            # "lemmas": lemmas,
            # "lemma_rules": lemma_rules,
            "ids": ids,
            "multiword_ids": multiword_ids,
            "multiword_forms": multiword_forms
        })

        return Instance(fields)

    def _restructure(self, example: Any, local: bool) -> List[Dict]:
        if local:
            return [
                {
                    "id": token["id"],
                    "form": token["form"],
                    "head": int(token["head"]) if token["head"] not in ("_", "None") else 0,
                    "deprel": token["deprel"],
                    "multi_id": None
                }
                for token in example
            ]
        else:
            tokens, heads, deprels = example["tokens"], example["head"], example["deprel"]
            return [
                {
                    "id": i + 1,
                    "form": tokens[i],
                    "head": int(heads[i]) if heads[i] not in ("_", "None") else 0,
                    "deprel": deprels[i],
                    "multi_id": None
                }
                for i in range(len(tokens))
            ]

reader = UniversalDependenciesReader(split="train")

# print(f"Faroese: {len(fo_data)} sentences")
# first_fo = fo_data[0]
# fo_words = [tok.text for tok in first_fo.fields["tokens"].tokens]
# print("First Faroese sentence words:", fo_words, "\n")

# veps_data = list(reader.read("data/vep_vwt-ud-test.conllu"))
# print(f"Veps:    {len(veps_data)} sentences")
# first_ve = veps_data[0]
# veps_words = [tok.text for tok in first_ve.fields["tokens"].tokens]
# print("First Veps sentence words:", veps_words)

# deu_data = load_dataset('universal_dependencies', 'nl_lassysmall')
# print(deu_data)