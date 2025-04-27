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

from datasets import load_dataset
from overrides import overrides
from typing import Any, Callable, Dict, Iterable, List, Tuple

from thesis.src.dataset_readers.reader_utils import process_multiword_tokens, gen_lemma_rule

import ast

@DatasetReader.register("ud_reader")
class UniversalDependenciesReader(DatasetReader):
    def __init__(
        self,
        tokenizer: Tokenizer = None,
        token_indexers: Dict[str, TokenIndexer] = None,
        split: str = "train",
    ):
        super().__init__()
        self.split = split
        self.tokenizer = tokenizer or PretrainedTransformerTokenizer(
            model_name="bert-base-multilingual-cased",
        )
        # now refer to the indexer you imported up above
        self._token_indexers = token_indexers or {
            "tokens": PretrainedTransformerIndexer(
                model_name="bert-base-multilingual-cased"
            )
        }
    
    @overrides(check_signature=False)
    def _read(self, file_path: str) -> Iterable[Instance]:            
        tb = load_dataset('universal_dependencies', file_path, split=self.split)

        def restructure_example(example: Dict) -> List[Dict]:
            tokens = example['tokens']
            # lemmas = example['lemmas']
            # upos = example['upos']
            # xpos = example['xpos']
            # feats = example['feats']
            heads = example['head']
            deprels = example['deprel']
            ids = list(range(1, len(tokens) + 1))

            token_dicts = []
            for i in range(len(tokens)):
                token = {
                    'id': ids[i],
                    'form': tokens[i],
                    # 'lemma': lemmas[i],
                    # 'upostag': upos[i],
                    # 'xpostag': xpos[i],
                    # 'feats': ast.literal_eval(feats[i]) if feats[i] not in ('_', 'None', None) else {},
                    'head': int(heads[i]) if heads[i] not in ('None', '_') else 0,
                    'deprel': deprels[i],
                    'multi_id': None
                }
                token_dicts.append(token)
            return token_dicts

        for example in tb:
            annotation = restructure_example(example)
            annotation = process_multiword_tokens(annotation)

            multiword_tokens = [x for x in annotation if x["multi_id"] is not None]
            annotation = [x for x in annotation if x["id"] is not None]

            if len(annotation) == 0:
                continue

            def get_field(tag: str, map_fn: Callable[[Any], Any] = None):
                map_fn = map_fn if map_fn is not None else lambda x: x
                return [map_fn(x[tag]) if x[tag] is not None else "_" for x in annotation if tag in x]
            
            ids = [x['id'] for x in annotation]
            multiword_ids = [x['multi_id'] for x in multiword_tokens]
            multiword_forms = [x['form'] for x in multiword_tokens]
            
            words = get_field('form')
            # lemmas = get_field('lemma')
            # upos_tags = get_field('upostag')
            # xpos_tags = get_field('xpostag')
            # feats = get_field('feats', lambda x: '|'.join(k + '=' + v for k, v in x.items())
            #                                     if hasattr(x, 'items') else '_')
            heads = get_field('head')
            dep_rels = get_field('deprel')
            dependencies = list(zip(dep_rels, heads))

            yield self.text_to_instance({
                "words": words,
                # "lemmas": lemmas,
                # "lemma_rules": ['_'] * len(words),
                # "upos_tags": upos_tags,
                # "xpos_tags": xpos_tags,
                # "feats": feats,
                "dependencies": dependencies,
                "ids": ids,
                "multiword_ids": multiword_ids,
                "multiword_forms": multiword_forms
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
