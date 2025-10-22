from transformers import TrainerCallback, Trainer, TrainingArguments, PreTrainedModel, set_seed

class LambdaUpdateCallback(TrainerCallback):
    def __init__(self, model, lambda_scheduler, total_epochs):
        self.model = model
        self.lambda_scheduler = lambda_scheduler
        self.total_epochs = total_epochs

    def on_epoch_begin(self, args, state, control, **kwargs):
        epoch = state.epoch if state.epoch is not None else 0
        new_lambda = self.lambda_scheduler(epoch, self.total_epochs)
        if hasattr(self.model, "grad_reverse"):
            self.model.grad_reverse.alpha = float(new_lambda)

