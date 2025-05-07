from typing import Any, Dict
import math

from torch.optim import Adam

from allennlp.data.data_loaders import SimpleDataLoader
from allennlp.models.model import Model
from allennlp.training.checkpointer import Checkpointer
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
    
    updates = cfg.training.total_updates
    num_steps_per_epoch = len(train_loader)
    epochs = math.ceil(updates / num_steps_per_epoch)
  
    optimizer = Adam([
        {'params': model_params, 'lr': cfg.training.lr_model},
        {'params': classifier_params, 'lr': cfg.training.lr_classifier}
    ])
    
    num_warmup_steps = int(cfg.training.warmup_rate * updates)
    scheduler = CosineWithWarmupLearningRateScheduler(
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=updates
    )

    checkpointer = Checkpointer(
        serialization_dir=cfg.output_dir,
        num_serialized_models_to_keep=0,
        keep_most_recent_by_count=0,
        save_completed_epochs=False,
        save_every_num_batches=None
    )

    return GradientDescentTrainer(
        model=model,
        optimizer=optimizer,
        data_loader=train_loader,
        validation_data_loader=dev_loader,
        learning_rate_scheduler=scheduler,
        num_epochs=epochs,
        serialization_dir=cfg.output_dir,
        cuda_device=cuda_device,
        checkpointer=checkpointer
    )
