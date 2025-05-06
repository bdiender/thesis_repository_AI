import argparse
import os
import random

from conllu import parse_incr, serialize


def load_sentences(path):
    with open(path, encoding='utf-8') as f:
        return list(parse_incr(f))


def write_split(sents, path):
    with open(path, 'w', encoding='utf-8') as f:
        f.write(serialize(sents))


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        '--data-dir', type=str, default='data',
        help='Location of dataset folders.'
    )
    p.add_argument(
        '--seed', type=int, default=1,
        help='Random seed for shuffling data.'
    )
    p.add_argument(
        '--train-share', type=float, default=0.5,
        help='Share of data that will be in the train split.'
    )
    p.add_argument(
        '--dev-share', type=float, default=0.2,
        help='Share of data that will be in the dev split.'
    )
    args = p.parse_args()

    random.seed(args.seed)

    for lang in os.listdir(args.data_dir):
        lang_dir = os.path.join(args.data_dir, lang)
        if not os.path.isdir(lang_dir):
            continue

        sentences = []
        files = [f for f in os.listdir(lang_dir) if f.endswith('conllu')]
        if not files:
            continue
        for f in files:
            sentences.extend(load_sentences(os.path.join(lang_dir, f)))
        
        random.shuffle(sentences)
        n = len(sentences)
        n_train = int(args.train_share * n)
        n_dev = int(args.dev_share * n)
        
        splits = {
            'train': sentences[:n_train],
            'dev': sentences[n_train:n_dev],
            'test': sentences[n_train + n_dev:]
        }

        data_name = files[0].rsplit('.', 1)[0].rsplit('-', 1)[0]
        for split, sents in splits.items():
            out = os.path.join(lang_dir, f'{data_name}-{split}.conllu')
            write_split(sents, out)
