import argparse
import os
import json
import dill

from allennlp.predictors.predictor import Predictor
from allennlp.data import Vocabulary

from config import GLOBAL_CONFIG
from dataset_readers import UniversalDependenciesReader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        required=True,
        help='Key of configuration settings for run.'
    )
    args = parser.parse_args()
    cfg = GLOBAL_CONFIG.get(args.config)
    
    reader = UniversalDependenciesReader(split=cfg.dataset.splits.test)
    
    with open(os.path.join(cfg.output_dir, 'model.pkl'), 'rb') as f:
        model = dill.load(f)

    model.vocab = Vocabulary.from_files(os.path.join(cfg.output_dir, 'vocabulary'))
    predictor = Predictor(model, dataset_reader=reader)

    token_count = 0
    correct_uas = 0
    correct_las = 0
    sent_count = 0
    correct_uem = 0
    correct_lem = 0

    with open(os.path.join(cfg.out_dir, 'predictions.tsv')) as f:
        f.write(f'Token ID\tWord\tGold head\tPred. head\tGold tag\tPred. tag\n')
        for inst in reader.read(cfg.datset.name):
            result = predictor.predict_instance(inst)

            words = result['words']
            ids = result['ids']
            pred_heads = result['heads']
            pred_tags = result['head_tags']

            gold_heads = [int(t) for t in inst.fields['heads_indices'].labels]
            gold_tags = inst.fields['head_tags'].labels

            all_ua_correct = True
            all_la_correct = True

            for i, (word, tok_id, gh, gt) in enumerate(zip(words, ids, gold_heads, gold_tags)):
                ph = pred_heads[i + 1]
                
                try:
                    pt = model.vocab.get_token_from_index(pred_tags[i + 1], 'head_tags')
                except KeyError:
                    pt = '<UNK>'
                
                arc_ok = (ph == gh)
                lab_ok = (pt == gt)

                if arc_ok:
                    correct_uas += 1
                else:
                    all_ua_correct = False
                if lab_ok:
                    correct_las += 1
                else:
                    all_la_correct = False
                
                token_count += 1

                f.write(f'{tok_id}\t{word}\t{gh}\t{ph}\t{gt}\t{pt}\n')

            f.write('\n')

            sent_count += 1
            if all_ua_correct:
                correct_uem += 1
            if all_la_correct:
                correct_lem += 1

    uas = correct_uas / token_count
    las = correct_las / token_count
    uem = correct_uem / sent_count
    lem = correct_lem / sent_count

    with open(os.path.join(cfg.output_dir, 'eval_results.json'), 'w') as out:
        json.dump(
            {"UAS": uas, "LAS": las, "UEM": uem, "LEM": lem},
            out,
            indent=2
        )


if __name__ == '__main__':
    main()
