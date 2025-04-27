from allennlp.training.learning_rate_schedulers import CombinedLearningRateScheduler
from torch import optim

def build_optimizer_and_scheduler(model, cfg, num_steps_per_epoch):
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
