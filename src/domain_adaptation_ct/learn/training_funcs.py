import datetime
import os
import logging
from typing import Optional, Callable

from domain_adaptation_ct.config.experiment_config import TrainingConfig, EvaluationConfig
from domain_adaptation_ct.dataset.dataset import DATASET_REGISTRY
from domain_adaptation_ct.learn.architectures import ARCHITECTURE_REGISTRY
from domain_adaptation_ct.learn.lambda_schedules import LAMBDA_SCHEDULER_REGISTRY, LambdaUpdateCallback
from domain_adaptation_ct.learn.metrics import make_metrics_fn
from domain_adaptation_ct.learn.trainers import TRAINER_REGISTRY
from domain_adaptation_ct.logging.logging_mixin import init_logging

import torch
from transformers import Trainer, TrainingArguments

def train_model(
    trainer_cls: type[Trainer],
    model: torch.nn.Module,
    train_dataset: torch.utils.data.Dataset,
    eval_dataset: torch.utils.data.Dataset,
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
    fold_index = 99 # TODO - integrate 5-fold cross validation

    callbacks = []

    if lambda_scheduler is None:
        assert not hasattr(model, "grad_reverse"), "Do not specify a lambda scheduler if there is no `grad_reverse` layer."
        scheduler_name = "no_lambda_scheduler"
    else:
        assert hasattr(model, "grad_reverse"), "Must specify a lambda scheduler with the `grad_reverse` layer."
        callbacks.append(LambdaUpdateCallback(model, lambda_scheduler, num_train_epochs))
        scheduler_name = lambda_scheduler.__name__

    date_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs(output_dir, exist_ok=True)
    output_dir_results = os.path.join(output_dir, f"{scheduler_name}_fold_{fold_index}_results_{date_str}")
    model_save_path = os.path.join(output_dir, f"{scheduler_name}_fold_{fold_index}_final_model_{date_str}")

    logging.info(f"Will write results to {output_dir_results}.")
    logging.info(f"Will write model to {model_save_path}.")

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

    logging.info(training_args)

    trainer = trainer_cls(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
        compute_metrics=make_metrics_fn(model),
        callbacks=callbacks,
    )

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    trainer.save_model(model_save_path)

    return trainer


def run_experiment_from_config_file(
    config_file: str
):
    """Top-level call if using a config file."""
    cfg = TrainingConfig(config_file).config_dict

    init_logging(log_dir=cfg["log_dir"])

    # Instantiate the model.
    architecture_cls = ARCHITECTURE_REGISTRY[cfg["architecture"]["cls_name"]]
    model = architecture_cls(**cfg["architecture"]["cls_init_args"])
    logging.info(f"Instantiated model {architecture_cls.__name__}. Summary:\n{model}")

    # Instantiate the datasets.
    dataset_cls = DATASET_REGISTRY[cfg["training"]["dataset"]["cls_name"]]
    train_dataset = dataset_cls.load(**cfg["training"]["dataset"]["cls_train_load_args"])
    logging.info(f"Instantiated train dataset {dataset_cls.__name__}, length {len(train_dataset)}")

    eval_dataset = dataset_cls.load(**cfg["training"]["dataset"]["cls_eval_load_args"])
    logging.info(f"Instantiated eval dataset {dataset_cls.__name__}, length {len(eval_dataset)}")

    trainer_cls = TRAINER_REGISTRY[cfg["training"]["trainer"]["cls_name"]]

    lambda_scheduler = None
    if "lambda_scheduler" in cfg["training"]:
        lambda_scheduler = LAMBDA_SCHEDULER_REGISTRY[cfg["training"]["lambda_scheduler"]]
        logging.info(f"Got lambda scheduler {lambda_scheduler.__name__}")
    else:
        logging.info(f"No lambda scheduler provided.")

    train_model(
        trainer_cls=trainer_cls,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        output_dir=cfg["training"]["training_arguments"]["output_dir"],
        num_train_epochs=cfg["training"]["training_arguments"]["num_train_epochs"],
        batch_size=cfg["training"]["training_arguments"]["batch_size"],
        learning_rate=cfg["training"]["training_arguments"]["learning_rate"],
        weight_decay=cfg["training"]["training_arguments"]["weight_decay"],
        optim=cfg["training"]["training_arguments"]["optim"],
        resume_from_checkpoint=cfg["training"]["training_arguments"]["resume_from_checkpoint"],
        lambda_scheduler=lambda_scheduler,
    )


def evaluate_model(
    evaluator_cls: type[Trainer],
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

    evaluator = evaluator_cls(
        model=model,
        args=evaluation_args,
        train_dataset=None,
        eval_dataset=eval_dataset,
        compute_metrics=make_metrics_fn(model)
    )

    metrics = evaluator.evaluate()
    return metrics


