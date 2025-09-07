from transformers import TrainerCallback

import numpy as np

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
        # print(f"[Callback] Epoch {epoch:.0f}: lambda = {new_lambda:.4f}")

    @staticmethod
    def lambda_scheduler_func(epoch, total_epochs):

        p = epoch / total_epochs

        lambda_p = 2. / (1. + np.exp(-10 * p)) - 1
        return float(lambda_p)
