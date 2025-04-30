import io
import argparse

import dataset_readers.dataset_reader
import predictors.udify_predictor

from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor
from dataset_readers.conll18_ud_eval import load_conllu_file, evaluate


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model-path', required=True,
                   help='e.g. outputs/nld_test/best.th')
    p.add_argument('--test-file',  required=True,
                   help='e.g. data/hu_szeged-test.conllu')
    args = p.parse_args()

    archive   = load_archive(args.model_path, cuda_device=-1)
    predictor = Predictor.from_archive(archive, 'udify_predictor')
    reader    = predictor._dataset_reader

    instances = reader.read(args.test_file)

    sys_lines = []
    for inst in instances:
        out = predictor.predict_instance(inst)
        sys_lines.append(predictor.dump_line(out))
    system_conllu = ''.join(sys_lines)

    gold_ud   = load_conllu_file(args.test_file)
    system_ud = load_conllu_file(io.StringIO(system_conllu))

    scores = evaluate(gold_ud, system_ud)
    print(f"LAS F1 = {scores['LAS'].f1 * 100:.2f}")

if __name__ == '__main__':
    main()