import argparse
import numpy as np
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel


def load_embedder():
    '''
    Load either a pre-trained BERT of finetuned embedder.
    '''
    pass


def load_data():
    '''
    Load random sentences from Tatoeba
    '''
    pass


def get_sentence_embedding(sentence=str, model=None):
    '''
    Get sentence embedding by passing tokens and taking mean-pool.
    If model is none, use BERT base
    '''
    pass


def main(args):
    embedder = load_embedder(args.model_dir) if args.model_dir is not None else None
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

    train_set = f'{args.data_dir}/probing_data.train.tsv'
    test_set = f'{args.data_dir}/probing_data.test.tsv'

    train_sents, train_labels = load_data(train_set)
    test_sents, test_labels = load_data(test_set)

    langs = sorted(set(train_labels))
    langs_idx = {lang: i for i, lang in enumerate(langs)}

    X_train = np.array([get_sentence_embedding(embedder, tokenizer, s) for s in train_sents])
    y_train = np.array([langs_idx[l] for l in train_labels])

    X_test = np.array([get_sentence_embedding(embedder, tokenizer, s) for s in test_sents])
    y_test = np.array([langs_idx[l] for l in test_labels])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', default=None,
                        help='Path to model.pkl, default to pre-trained M-BERT if left empty.')
    parser.add_argument('--data-dir', required=True,
                        help='Path that contains the datasets.')
    # TODO: Split dataset in this file
    args = parser.parse_args()
    main(args)

