# train.py

from allennlp.data import Vocabulary
from allennlp.data.data_loaders import SimpleDataLoader
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import PretrainedTransformerEmbedder
from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper
from allennlp.training.gradient_descent_trainer import GradientDescentTrainer

import torch
import torch.optim as optim

from dataset_readers.dataset_reader import UniversalDependenciesReader
from models.dependency_decoder import DependencyDecoder
from models.base_model import BaseModel

def main():
    # 1. Read train + validation splits for Hungarian Szeged
    train_dataset = list(UniversalDependenciesReader(split="train").read("hu_szeged"))
    dev_dataset   = list(UniversalDependenciesReader(split="validation").read("hu_szeged"))

    # 2. Build the vocabulary
    vocab = Vocabulary.from_instances(train_dataset + dev_dataset)

    # 3. Embeddings and encoders
    token_embedder = PretrainedTransformerEmbedder("bert-base-multilingual-cased")
    text_field_embedder = BasicTextFieldEmbedder({"tokens": token_embedder})

    shared_encoder = PytorchSeq2SeqWrapper(
        torch.nn.LSTM(
            input_size=text_field_embedder.get_output_dim(),
            hidden_size=384,    # 384*2 = 768
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
        dropout=0.2,
    )

    # 4. Assemble the model
    model = BaseModel(
        vocab=vocab,
        text_field_embedder=text_field_embedder,
        encoder=shared_encoder,
        decoder=decoder,
        mix_embedding=12,
        layer_dropout=0.1,
    )

    # 5. DataLoaders + indexing
    train_loader = SimpleDataLoader(train_dataset, batch_size=20, shuffle=True)
    train_loader.index_with(vocab)

    dev_loader = SimpleDataLoader(dev_dataset, batch_size=20)
    dev_loader.index_with(vocab)

    # 6. Optimizer (single group, simple)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # 7. Trainer
    trainer = GradientDescentTrainer(
        model=model,
        optimizer=optimizer,
        data_loader=train_loader,
        validation_data_loader=dev_loader,
        num_epochs=25,
        serialization_dir="serialization_hu_szeged",
        cuda_device=-1,  # or -1 for CPU
    )

    print("Starting training!")
    trainer.train()

if __name__ == "__main__":
    main()
