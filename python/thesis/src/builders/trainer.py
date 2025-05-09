from typing import Any, Dict
import math

from torch.optim import Adam

from allennlp.data.data_loaders import SimpleDataLoader
from allennlp.models.model import Model
from allennlp.training.checkpointer import Checkpointer
from allennlp.training.gradient_descent_trainer import GradientDescentTrainer
from allennlp.training.learning_rate_schedulers import CosineWithWarmupLearningRateScheduler, NoamLR

from modules import UnfreezeBertCallback

def build_trainer(
        model: Model,
        train_loader: SimpleDataLoader,
        dev_loader:SimpleDataLoader,
        cfg: Dict[str, Any],
        cuda_device: int
    ):
    classifier_params = list(model.decoder.parameters())
    if cfg.training.freeze_classifier:
        for p in classifier_params:
            p.requires_grad = False
    classifier_param_ids = {id(p) for p in classifier_params}
    model_params = [p for p in model.parameters() if id(p) not in classifier_param_ids]
    
    updates = cfg.training.total_updates
    num_steps_per_epoch = len(train_loader)
    epochs = math.ceil(updates / num_steps_per_epoch)
  
    optimizer = Adam([
        {'params': model_params, 'lr': cfg.training.lr_model, 'weight_decay': cfg.training.weight_decay},
        {'params': classifier_params, 'lr': cfg.training.lr_classifier, 'weight_decay': cfg.training.weight_decay}
    ])
    
    scheduler = None
    if cfg.training.scheduler == 'cosine':
        num_warmup_steps = math.ceil(cfg.training.warmup_rate * updates)
        scheduler = CosineWithWarmupLearningRateScheduler(
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=updates
        )

    checkpointer = Checkpointer(
        serialization_dir=cfg.output_dir,
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
        callbacks=[UnfreezeBertCallback(cfg.output_dir,
            freeze_bert=cfg.training.freeze_bert,
            frozen_epochs=max(0, cfg.training.freeze_bert_until),
            lr_model=cfg.training.lr_model,
            weight_decay=cfg.training.weight_decay
        )],
        cuda_device=cuda_device,
        checkpointer=checkpointer
    )
