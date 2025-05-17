from typing import Dict
from allennlp.training.callbacks.callback import TrainerCallback
import wandb

@TrainerCallback.register('wandb-metrics')
class WandBMetricsCallback(TrainerCallback):
    def __init__(self, serialization_dir: str) -> None:
        super().__init__(serialization_dir)

    def on_epoch(
        self,
        trainer,
        metrics: Dict[str, float],
        epoch: int,
        **kwargs
    ) -> None:
        to_log = {'epoch': epoch}
        for split in ('training', 'validation'):
            for name in ('loss', 'UAS', 'LAS', 'UEM', 'LEM'):
                key = f"{split}_{name}"
                if key in metrics:
                    to_log[f"{split}/{name}"] = metrics[key]
        wandb.log(to_log)
