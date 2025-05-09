from typing import Any, Dict

from allennlp.training.callbacks.callback import TrainerCallback

from allennlp.training.callbacks.callback import TrainerCallback

@TrainerCallback.register("unfreeze-bert")
class UnfreezeBertCallback(TrainerCallback):
    def __init__(
        self,
        serialization_dir: str,
        freeze_bert: bool = False,
        frozen_epochs: int = 1,
        lr_model: float = 1e-5,
        weight_decay: float = 0.0
    ):
        super().__init__(serialization_dir)
        self.frozen_epochs = frozen_epochs
        self.freeze_bert = freeze_bert
        self.lr_model = lr_model
        self.weight_decay = weight_decay

    def on_start(self, trainer, **kwargs):
        if self.freeze_bert:
            bert = trainer.model.text_field_embedder._token_embedders["tokens"].transformer_model

            bert_params = set(bert.parameters())
            for p in bert_params:
                p.requires_grad = False

            for group in trainer.optimizer.param_groups:
                group["params"] = [p for p in group["params"] if p not in bert_params]


    def on_epoch(self, trainer, metrics, epoch, **kwargs):
        if self.freeze_bert and epoch == self.frozen_epochs:
            bert = trainer.model.text_field_embedder._token_embedders["tokens"].transformer_model
            new_params = []
            for p in bert.parameters():
                p.requires_grad = True
                new_params.append(p)
            trainer.optimizer.add_param_group({
                "params": new_params,
                "lr": self.lr_model,
                "weight_decay": self.weight_decay
            })
