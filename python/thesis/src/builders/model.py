import dill
from typing import Any, Dict

import torch

from allennlp.data import Vocabulary
from allennlp.models.archival import load_archive
from allennlp.models.model import Model
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import PretrainedTransformerEmbedder
from allennlp.modules.seq2seq_encoders.pass_through_encoder import PassThroughEncoder

from models import BaseModel, DependencyDecoder


def build_model(cfg: Dict[str, Any], vocab: Vocabulary, cuda_device: int = 0) -> Model:
    if cfg.model.get('finetune_on'):
        with open(cfg.model.finetune_on, 'rb') as f:
            model = dill.load(f)
        model.vocab = vocab

        if cuda_device >= 0:
            model = model.cuda(cuda_device)

        return model
    
    token_embedder = PretrainedTransformerEmbedder("bert-base-multilingual-cased")
    text_field_embedder = BasicTextFieldEmbedder({"tokens": token_embedder})

    shared_encoder = PassThroughEncoder(
        input_dim=text_field_embedder.get_output_dim()
    )

    decoder_encoder = PassThroughEncoder(
        input_dim=shared_encoder.get_output_dim()
    )

    decoder = DependencyDecoder(
        vocab=vocab,
        encoder=decoder_encoder,
        tag_representation_dim=cfg.model.decoder.tag_repr_dim,
        arc_representation_dim=cfg.model.decoder.arc_repr_dim,
        dropout=cfg.model.decoder.dropout
    )

    return BaseModel(
        vocab=vocab,
        text_field_embedder=text_field_embedder,
        encoder=shared_encoder,
        decoder=decoder,
        mix_embedding=12,
        layer_dropout=0.1
    )
