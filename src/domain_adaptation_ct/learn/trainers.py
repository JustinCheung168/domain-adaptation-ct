from transformers import TrainerCallback, Trainer, TrainingArguments, PreTrainedModel, set_seed
import torch
import torch.nn as nn

from domain_adaptation_ct.learn.architectures import ResNet50Baseline, ResNet50DANN
from domain_adaptation_ct.learn.loss import MaskedDomainAdversarialLoss
from domain_adaptation_ct.learn.metrics import make_metrics_fn

class BaselineTrainer(Trainer):
    """Trainer for ResNet50Baseline model with standard classification loss"""
    def compute_loss(self, model: ResNet50Baseline, inputs: dict[str, torch.Tensor], return_outputs: bool = False):
        pixel_values = inputs.pop("pixel_values")
        outputs = model(pixel_values)
        logits = outputs.logits

        labels = inputs.pop("labels")
        loss = model.loss_fn(logits, labels)

        if return_outputs:
            return loss, outputs
        return loss

class DANNTrainer(BaselineTrainer):
    """Trainer for ResNet50DANN model with both label and domain losses"""
    def compute_loss(self, model: ResNet50DANN, inputs: dict[str, torch.Tensor], return_outputs: bool = False):
        pixel_values = inputs.pop("pixel_values")
        outputs = model(pixel_values)
        logits1 = outputs.logits1
        logits2 = outputs.logits2

        labels1 = inputs.pop("labels1")
        labels2 = inputs.pop("labels2")

        loss = model.loss_fn(logits1, logits2, labels1, labels2)

        if return_outputs:
            return loss, outputs
        return loss


def train_model(
    train_dataset,
    eval_dataset,
    model: torch.nn.Module,
    output_dir: str,
    num_epochs: int,
    batch_size: int,
):
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_dir='./logs',
        logging_steps=10,
        log_level="info",
        logging_strategy="epoch",
        metric_for_best_model="eval_f1",
        learning_rate=0.1,
        weight_decay=1e-4,
        seed=42,
        optim="sgd"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=make_metrics_fn(model)
    )

    trainer.train()
    return trainer





