"""
Adapted from: https://github.com/Hyperparticle/udify/blob/master/udify/models/udify_model.py
"""

from allennlp.common.checks import check_dimensions_match
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import Seq2SeqEncoder, TextFieldEmbedder
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn.util import get_text_field_mask

from modules.scalar_mix import ScalarMixWithDropout
from overrides import overrides
from transformers import BertTokenizer
from typing import Any, Dict, Optional, List

import torch

@Model.register('base_model')
class BaseModel(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 decoder: Model,  # TODO: set default
                 post_encoder_embedder: TextFieldEmbedder = None,
                 dropout: float = 0.0,
                 word_dropout: float = 0.0,
                 mix_embedding: int = None,
                 layer_dropout: int = 0.0,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = RegularizerApplicator()):
        super(BaseModel, self).__init__(vocab, regularizer)

        self.vocab = vocab
        self.bert_vocab = BertTokenizer.from_pretrained()
        self.text_field_embedder = text_field_embedder
        self.post_encoder_embedder = post_encoder_embedder
        self.shared_encoder = encoder
        self.word_dropout = word_dropout
        self.dropout = torch.nn.Dropout(p=dropout)
        self.decoder = decoder
        self.scalar_mix = ScalarMixWithDropout(mix_embedding, do_layer_norm=False, dropout=layer_dropout)

        self.metrics = {}

        check_dimensions_match(text_field_embedder.get_output_dim(), encoder.get_output_dim(),
                               "text field embedding dim", "encoder input dim")
        initializer(self)
    
    @overrides
    def forward(
        self,
        tokens: Dict[str, torch.Tensor],
        head_tags: Optional[torch.Tensor]    = None,
        head_indices: Optional[torch.Tensor] = None,
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, torch.Tensor]:
        mask = get_text_field_mask(tokens)

        embedded_text_input = self.text_field_embedder(tokens)
        encoded_text = self.shared_encoder(embedded_text_input, mask)

        decoder_input = self.scalar_mix(encoded_text, mask)

        if self.post_encoder_embedder:
            post_embeddings = self.post_encoder_embedder(tokens)
            decoder_input = decoder_input + post_embeddings

        pred_output = self.decoder(
            decoder_input,
            mask,
            head_tags,
            head_indices,
            metadata
        )

        output_dict = {
            'heads': pred_output['heads'],
            'head_tags': pred_output['head_tags'],
            'arc_loss': pred_output.get('arc_loss', 0.0),
            'tag_loss': pred_output.get('tag_loss', 0.0),
            'mask': pred_output['mask'],
        }

        if 'loss' in pred_output:
            output_dict['loss'] = pred_output['loss']

        if metadata is not None:
            output_dict['words'] = [x['words'] for x in metadata]
            output_dict['ids'] = [x['ids'] for x in metadata if "ids" in x]
            output_dict['multiword_ids'] = [x['multiword_ids'] for x in metadata if 'multiword_ids' in x]
            output_dict['multiword_forms'] = [x['multiword_forms'] for x in metadata if 'multiword_forms' in x]

        return output_dict
