from transformers import Trainer

class LambdaScheduleTrainer(Trainer):
    def __init__(self, *args, model=None, lambda_scheduler=None, total_epochs=50, **kwargs):
        super().__init__(*args, model=model, **kwargs)
        self.lambda_scheduler = lambda_scheduler
        self.current_epoch = 0
        self.total_epochs = total_epochs

    def on_epoch_begin(self):
        if self.lambda_scheduler:
            new_lambda = self.lambda_scheduler(self.current_epoch, self.total_epochs)
            if hasattr(self.model, 'grad_reverse'):
                self.model.grad_reverse.alpha = new_lambda
            print(f"Epoch {self.current_epoch}: lambda = {new_lambda:.4f}")

        self.current_epoch += 1
