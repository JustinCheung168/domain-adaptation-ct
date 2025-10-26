
def evaluate_model(eval_dataset, model, output_dir="./results", num_epochs=3, batch_size=32):
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_dir='./logs',
        logging_steps=10,
        # log_level="info",
        logging_strategy="epoch",
        # load_best_model_at_end=True,
        # metric_for_best_model="eval_f1",
        # greater_is_better=True,            
        learning_rate=0.1,
        weight_decay=1e-4,
        seed=42,
        optim="sgd"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=None,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics
    )

    metrics = trainer.evaluate()
    return metrics



import torch
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import Trainer, TrainingArguments, PreTrainedModel, ResNetConfig
from transformers.modeling_outputs import ModelOutput
from transformers.models.resnet.modeling_resnet import ResNetForImageClassification
from dataclasses import dataclass
from typing import Optional
from medmnist import OrganAMNIST
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np


def train_model(train_dataset, eval_dataset, model, output_dir="./results", num_epochs=3, batch_size=32):
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_dir='./logs',
        logging_steps=10,
        # log_level="info",
        logging_strategy="epoch",
        # load_best_model_at_end=True,
        # metric_for_best_model="eval_f1",
        # greater_is_better=True,            
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
        compute_metrics=compute_metrics
    )

    trainer.train()
    return trainer





