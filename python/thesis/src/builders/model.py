import torch
from allennlp.data import Vocabulary
from allennlp.models.archival import load_archive
from allennlp.models.model import Model
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import PretrainedTransformerEmbedder
from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper

from models.dependency_decoder import DependencyDecoder
from models.base_model import BaseModel

from typing import Any, Dict

def build_model(cfg: Dict[str, Any], vocab: Vocabulary, cuda_device: int = 0) -> Model:
    model_cfg = cfg['model']

    if model_cfg.get('finetune_on'):
        archive = load_archive(model_cfg['finetune_on'], cuda_device=cuda_device)
        model = archive.model
        model.vocab = vocab

        return model
    
    token_embedder = PretrainedTransformerEmbedder("bert-base-multilingual-cased")
    text_field_embedder = BasicTextFieldEmbedder({"tokens": token_embedder})

    shared_encoder = PytorchSeq2SeqWrapper(
        torch.nn.LSTM(
            input_size=text_field_embedder.get_output_dim(),
            hidden_size=384,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
        )
    )

    decoder_encoder = PytorchSeq2SeqWrapper(
        torch.nn.LSTM(
            input_size=shared_encoder.get_output_dim(),
            hidden_size=384,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
        )
    )
    decoder = DependencyDecoder(
        vocab=vocab,
        encoder=decoder_encoder,
        tag_representation_dim=256,
        arc_representation_dim=256,
        dropout=0.2,  # TODO: Check
    )

    return BaseModel(
        vocab=vocab,
        text_field_embedder=text_field_embedder,
        encoder=shared_encoder,
        decoder=decoder,
        mix_embedding=12,
        layer_dropout=0.1,
    )
