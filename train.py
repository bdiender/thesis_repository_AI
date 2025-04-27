from argparse import ArgumentParser
from datasets import load_dataset, load_metric
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

import datetime
import torch

def tokenize(batch):
    return tokenizer(batch['premise'], batch['hypothesis'], truncation=True, padding=True)

if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('-S', '--seed',
                    dest='seed',
                    help='The random seed to start with.',
                    default=1, type=int)
    parser.add_argument('-C', '--count',
                    dest='count',
                    help='The number of models trained in this run.',
                    default=1, type=int)
    parser.add_argument('-T', '--toydata',
                    dest='toydata',
                    help='Whether the toy data is used or not.',
                    default=False, type=bool)

    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError
    else:
        print(f"There are {torch.cuda.device_count()} GPUs available")
        for i in range(torch.cuda.device_count()):
            print(f"Device {i}: {torch.cuda.get_device_name(i)}")

    log = f'[{datetime.datetime.now()}] - Started'

    ## Load in data
    train_data = load_dataset('glue', 'mnli')['train']
    val_data = load_dataset('glue', 'mnli')['validation_matched']

    tokenizer = AutoTokenizer.from_pretrained('roberta-base')
    model = AutoModelForSequenceClassification.from_pretrained('roberta-base', num_labels=3)
    log += f'\n[{datetime.datetime.now()}] - Model and data loaded'

    ## Pre-processing
    train_data = train_data.map(tokenize, batched=True)
    val_data = val_data.map(tokenize, batched=True)

    ### Create toy data for training
    toy_data = train_data.select(range(1000))
    log += f'\n[{datetime.datetime.now()}] - Pre-processing complete'

    for s in range(args.seed, args.seed + args.count):
        training_args = TrainingArguments(
                output_dir=f'./output_{s}/',
                seed=s,
                save_strategy="no",
                # per_device_train_batch_size=16,
                # per_device_eval_batch_size=16,
                # gradient_accumulation_steps=2,
                # fp16=True,
                num_train_epochs=1)

        trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_data,
                eval_dataset=val_data,
                tokenizer=tokenizer,
                compute_metrics=load_metric('glue', 'mnli'))

        if args.toydata:
            trainer.train_dataset = toy_data

        log += f'\n[{datetime.datetime.now()}] - Starting training model {args.seed - s + 1}/{args.count}'
        trainer.train()
        log += f'\n[{datetime.datetime.now()}] - Finished training model {args.seed - s + 1}/{args.count}'
        model_name = f'roberta_mnli_s{s}.pt'

        if args.toydata:
            model_name = f'small_{model_name}'

        torch.save(model.state_dict(), f'./output_{s}/{model_name}')
        with open(f'./output_{s}/log.txt', 'w') as f:
            f.write(log)

