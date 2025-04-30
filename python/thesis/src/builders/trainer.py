from allennlp.data.data_loaders import SimpleDataLoader
from allennlp.models.model import Model
from allennlp.training.gradient_descent_trainer import GradientDescentTrainer
from allennlp.training.learning_rate_schedulers import CosineWithWarmupLearningRateScheduler

from torch.optim import Adam
from typing import Any, Dict


def build_trainer(
        model: Model,
        train_loader: SimpleDataLoader,
        dev_loader:SimpleDataLoader,
        cfg: Dict[str, Any],
        cuda_device: int
    ):

    training_cfg = cfg['training']

    classifier_params = list(model.decoder.parameters())
    model_params = [p for p in model.parameters() if p not in classifier_params]

    optimizer = Adam(
        {'params': model_params, 'lr': training_cfg['lr_model']},
        {'params': classifier_params, 'lr': training_cfg['lr_classifier']}
    )
    num_warmup_steps = int(training_cfg['warmup_rate'] * training_cfg['num_steps_per_epoch'])
    scheduler = CosineWithWarmupLearningRateScheduler(
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=training_cfg['num_steps_per_epoch']
        )

    return GradientDescentTrainer(
        model=model,
        optimizer=optimizer,
        data_loader=train_loader,
        validation_data_loader=dev_loader,
        learning_rate_scheduler=scheduler,
        num_epochs=training_cfg['epochs'],
        serialization_dir=training_cfg['output_dir'],
        cuda_device=cuda_device
    )
