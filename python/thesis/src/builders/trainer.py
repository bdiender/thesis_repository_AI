from typing import Any, Dict

from torch.optim import Adam

from allennlp.data.data_loaders import SimpleDataLoader
from allennlp.models.model import Model
from allennlp.training.gradient_descent_trainer import GradientDescentTrainer
from allennlp.training.learning_rate_schedulers import CosineWithWarmupLearningRateScheduler


def build_trainer(
        model: Model,
        train_loader: SimpleDataLoader,
        dev_loader:SimpleDataLoader,
        cfg: Dict[str, Any],
        cuda_device: int
    ):
    classifier_params = list(model.decoder.parameters())
    classifier_param_ids = {id(p) for p in classifier_params}
    model_params = [p for p in model.parameters() if id(p) not in classifier_param_ids]
    
    optimizer = Adam([
        {'params': model_params, 'lr': cfg.training.lr_model},
        {'params': classifier_params, 'lr': cfg.training.lr_classifier}
    ])
    
    num_warmup_steps = int(cfg.training.warmup_rate * cfg.training.num_steps_per_epoch)
    scheduler = CosineWithWarmupLearningRateScheduler(
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=cfg.training.num_steps_per_epoch
        )

    return GradientDescentTrainer(
        model=model,
        optimizer=optimizer,
        data_loader=train_loader,
        validation_data_loader=dev_loader,
        learning_rate_scheduler=scheduler,
        num_epochs=cfg.training.epochs,
        serialization_dir=cfg.output_dir,
        cuda_device=cuda_device
    )
