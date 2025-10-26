from typing import Type

from domain_adaptation_ct.learn.architectures import ResNet50Baseline, ResNet50DANN
from domain_adaptation_ct.learn.metrics import make_metrics_fn

from transformers import Trainer, TrainingArguments
import torch

class BaselineTrainer(Trainer):
    """
    Trainer for ResNet50Baseline model.
    """
    def compute_loss(self, model: ResNet50Baseline, inputs: dict[str, torch.Tensor], return_outputs: bool = False):
        pixel_values = inputs.pop("pixel_values")
        outputs = model(pixel_values)
        logits = outputs.logits

        labels = inputs.pop("labels")
        loss = model.loss_fn(logits, labels)

        if return_outputs:
            return loss, outputs
        return loss

class DANNTrainer(Trainer):
    """
    Trainer for ResNet50DANN model.
    """
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
    trainer_cls: Type[Trainer],
    train_dataset: torch.utils.data.Dataset,
    eval_dataset: torch.utils.data.Dataset,
    model: torch.nn.Module,
    output_dir: str,
    num_epochs: int,
    batch_size: int,
):
    """Wrapper for using one of the above trainers."""
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=0.1,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        weight_decay=1e-4,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        seed=42,
        optim="sgd"
    )

    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=make_metrics_fn(model)
    )

    trainer.train(resume_from_checkpoint=True)
    return trainer

def evaluate_model(
    trainer_cls: Type[Trainer],
    eval_dataset: torch.utils.data.Dataset,
    model: torch.nn.Module,
    output_dir: str,
    batch_size: int,
):
    evaluation_args = TrainingArguments(
        output_dir=output_dir,
        per_device_eval_batch_size=batch_size,
        seed=42,
    )

    evaluator = trainer_cls(
        model=model,
        args=evaluation_args,
        train_dataset=None,
        eval_dataset=eval_dataset,
        compute_metrics=make_metrics_fn(model)
    )

    metrics = evaluator.evaluate()
    return metrics
