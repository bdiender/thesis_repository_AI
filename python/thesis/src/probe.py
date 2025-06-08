import argparse
import dill
import os

import numpy as np
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel

from models import LanguageIDProbe

def load_embedder(model_dir):
    model = dill.load(open(os.path.join(model_dir, 'model.pkl'), 'rb'))
    return model.text_field_embedder._token_embedders['tokens']


def load_data(file_path: str):
    with open(file_path, 'r', encoding='utf-8') as f:
        sents, labels = [], []
        for line in f.readlines():
            sent, label = line.strip().split('\t', 2)
            sents.append(sent)
            labels.append(label)        
    return sents, labels


def get_sentence_embedding(embedder, tokenizer, sent, layer_idx=-1, device='cpu'):
    inputs = tokenizer(sent,
                       return_tensors='pt',
                       add_special_tokens=True).to(device)
    
    with torch.no_grad():
        if hasattr(embedder, 'transformer_model'):
            out = embedder.transformer_model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                token_type_ids=inputs.get('token_type_ids'),
                output_hidden_states=True,
                return_dict=True
            )
        else:
            out = embedder(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                token_type_ids=inputs.get('token_type_ids'),
                output_hidden_states=True,
                return_dict=True
            )
    
    last_hidden = out.hidden_states[layer_idx].squeeze(0)
    mask = inputs['attention_mask'].squeeze(0).unsqueeze(-1)
    pooled = (last_hidden * mask).sum(dim=0) / mask.sum()

    return pooled.cpu().numpy()
    

def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    embedder = load_embedder(args.model_dir) if args.model_dir is not None else BertModel.from_pretrained('bert-base-multilingual-cased')
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

    embedder.to(device)
    embedder.eval()

    train_set = f'{args.data_dir}/probing_data.train.tsv'
    test_set = f'{args.data_dir}/probing_data.test.tsv'

    train_sents, train_labels = load_data(train_set)
    test_sents, test_labels = load_data(test_set)

    langs = sorted(set(train_labels))
    langs_idx = {lang: i for i, lang in enumerate(langs)}

    for layer in range(12):
        X_train = np.array([get_sentence_embedding(embedder, tokenizer, s, layer_idx=layer, device=device) for s in train_sents])
        y_train = np.array([langs_idx[l] for l in train_labels])

        X_test = np.array([get_sentence_embedding(embedder, tokenizer, s, layer_idx=layer, device=device) for s in test_sents])
        y_test = np.array([langs_idx[l] for l in test_labels])

        probe = LanguageIDProbe()
        probe.fit(X_train, y_train, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
        probe.evaluate(X_test, y_test, labels=langs, output_dir=args.model_dir, layer_idx=layer)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', required=True,
                        help='Path that contains the datasets.')
    parser.add_argument('--model-dir', default=None,
                        help='Path to folder that contains model.pkl, default to pre-trained M-BERT if left empty.')
    parser.add_argument('--seed', default=1,
                        help='Random seed.')
    parser.add_argument('--batch-size', default=1,
                        help='Batch size.')
    parser.add_argument('--lr', default=1e-3,
                        help='Learning rate.')
    parser.add_argument('--epochs', default=10,
                        help='Number of epochs.')
    # TODO: Split dataset in this script
    args = parser.parse_args()
    main(args)
