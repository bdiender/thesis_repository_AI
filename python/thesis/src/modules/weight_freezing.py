from allennlp.training.callbacks.callback import Events, TrainerCallback, handle_event

class UnfreezeBertCallback(TrainerCallback):
    def __init__(self, frozen_epochs: int = 1):
        super().__init__()
        self.frozen_epochs = frozen_epochs
    
    @handle_event(Events.TRAINING_START)
    def freeze(self, trainer, **kwargs):
        if trainer.model._configuration.training.freeze_bert:
            for p in trainer.model.text_field_embedder._token_embedders['tokens'].transformer_model.parameters():
                p.requires_grad = False
    
    @handle_event(Events.EPOCH_END)
    def unfreeze(self, trainer, **kwargs):
        if trainer._epochs_completed == self.frozen_epochs \
            and trainer.model._configuration.training.freeze_bert:
            bert_module = trainer.model.text_field_embedder._token_embedders['tokens'].transformer_model
            new_params = []

            for p in bert_module.parameters():
                p.requires_grad = True
                new_params.append(p)
            
            trainer.optimizer.add_param_group({
                'params': new_params,
                'lr': trainer.model._configuration.training.lr_model,
                'weight_decay': trainer.model._configuration.training.weight_decay
            })
