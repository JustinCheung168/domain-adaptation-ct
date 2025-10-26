import datetime
import os
from typing import Type, Optional, Callable

from domain_adaptation_ct.learn.architectures import ResNet50Baseline, ResNet50DANN
from domain_adaptation_ct.learn.metrics import make_metrics_fn
from domain_adaptation_ct.learn.lambda_schedules import LambdaUpdateCallback

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
    model: torch.nn.Module,
    train_dataset: torch.utils.data.Dataset,
    eval_dataset: torch.utils.data.Dataset,
    fold_index: int,
    output_dir: str,
    num_train_epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    optim: str,
    resume_from_checkpoint: bool,
    lambda_scheduler: Optional[Callable[[int, int], float]] = None,
):
    """
    Wrapper for using one of the above trainers.
    """

    scheduler_name = lambda_scheduler.__name__
    date_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir_results = os.path.join(output_dir, f"{scheduler_name}_fold_{fold_index}_results_{date_str}")
    model_save_path = os.path.join(output_dir, f"{scheduler_name}_fold_{fold_index}_final_model_{date_str}")

    training_args = TrainingArguments(
        output_dir=output_dir_results,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_train_epochs,
        weight_decay=weight_decay,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        seed=42,
        optim=optim,
    )

    callbacks = []

    if lambda_scheduler is None:
        assert not hasattr(model, "grad_reverse"), "Do not specify a lambda scheduler if there is no `grad_reverse` layer."
        callbacks.append(LambdaUpdateCallback(model, lambda_scheduler, num_epochs))
    else:
        assert hasattr(model, "grad_reverse"), "Must specify a lambda scheduler with the `grad_reverse` layer."

    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=make_metrics_fn(model),
        callbacks=callbacks,
    )

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    trainer.save_model(model_save_path)

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
