# import io
# import argparse
# import torch
# import os
# from datasets import load_dataset

# import models.base_model
# from allennlp.data import Vocabulary
# from allennlp.predictors import Predictor

# from dataset_readers.dataset_reader import UniversalDependenciesReader
# from dataset_readers.conll18_ud_eval import load_conllu, load_conllu_file, evaluate
# from builders.model import build_model
# from config.loader import load_config
# from predictors.udify_predictor import UdifyPredictor

# def main():
#     p = argparse.ArgumentParser()
#     p.add_argument('--model-path', required=True)
#     p.add_argument('--test-file',  required=True)
#     args = p.parse_args()

#     cfg = load_config('configs/config.yaml', 'nl_lassysmall_test')
#     vocab = Vocabulary.from_files(os.path.join(args.model_path, 'vocabulary'))
#     model = build_model(cfg, vocab, cuda_device=-1)
#     model.load_state_dict(torch.load(os.path.join(args.model_path, 'weights.th'), map_location='cpu'))

#     predictor = UdifyPredictor(model, dataset_reader=UniversalDependenciesReader(split='test'))
#     reader = predictor._dataset_reader

#     if os.path.exists(args.test_file):
#         instances = list(reader.read(args.test_file))
#         gold_ud = load_conllu_file(args.test_file)
#     else:
#         instances = list(reader.read(args.test_file))
#         dataset = load_dataset("universal_dependencies", args.test_file, split="test")
#         conllu_data = "\n".join(dataset["text"])
#         gold_ud = load_conllu(io.StringIO(conllu_data))


#     sys_lines = []
#     for inst in instances:
#         out = predictor.predict_instance(inst)
#         sys_lines.append(predictor.dump_line(out))
#     system_conllu = ''.join(sys_lines)
#     system_ud = load_conllu(io.StringIO(system_conllu))

#     scores = evaluate(gold_ud, system_ud)
#     print(f"LAS F1 = {scores['LAS'].f1 * 100:.2f}")

# if __name__ == '__main__':
#     main()
