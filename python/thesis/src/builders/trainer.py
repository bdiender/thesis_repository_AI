from allennlp.models.model import Model
from allennlp.training.gradient_descent_trainer import GradientDescentTrainer
from allennlp.training.learning_rate_schedulers import CombinedLearningRateScheduler

from torch import optim

def build_trainer(
        model: Model,
        optimizer: Optimizer,
        scheduler: LearningRateScheduler,
        train_loader: DataLoader,
        dev_loader:DataLoader,
        cfg: Dict[str, Any]
    ):
    training_cfg = cfg['training']
    return GradientDescentTrainer(
        model=model,
        optimizer=optimizer,
        learning_rate_scheduler=scheduler,
        train_loader=train_loader,
        validation_data_loader=dev_loader,
        num_epochs=['epochs'],
        serialization_dir=training_cfg['output_dir'],
        cuda_device=cfg['cuda_device']
    )


def _build_optimizer_and_scheduler(model, cfg, num_steps_per_epoch):
    lr = cfg['training']['optimizer']['lr']
    optimizer = optim.Adam(model.parameters(), lr=lr)

    sched_cfg = cfg['training'].get('scheduler')
    if sched_cfg and sched_cfg.get('type') == 'combined':
        scheduler = CombinedLearningRateScheduler(
            optimizer,
            schedulers=sched_cfg['schedulers'],
            num_steps_per_epoch=num_steps_per_epoch
        )
        return optimizer, scheduler

    return optimizer, None
