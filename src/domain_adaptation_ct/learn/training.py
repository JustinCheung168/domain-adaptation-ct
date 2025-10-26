from transformers import TrainerCallback, Trainer, TrainingArguments, PreTrainedModel, set_seed
import torch
import torch.nn as nn

from domain_adaptation_ct.learn.architectures import ResNet50Baseline, ResNet50DANN
from domain_adaptation_ct.learn.loss import MaskedDomainAdversarialLoss

class BaselineTrainer(Trainer):
    """Trainer for ResNet50Baseline model with standard classification loss"""
    def compute_loss(self, model: ResNet50Baseline, inputs: dict[str, torch.Tensor], return_outputs=False):
        pixel_values = inputs.pop("pixel_values")
        outputs = model(pixel_values)
        logits = outputs.logits

        labels = inputs.pop("labels")
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(logits, labels)

        if return_outputs:
            return loss, outputs
        return loss

class DANNTrainer(BaselineTrainer):
    """Trainer for ResNet50DANN model with both label and domain losses"""
    def compute_loss(self, model: ResNet50DANN, inputs, return_outputs=False):
        # Get base classification loss first
        labels1 = inputs.pop("labels1")
        labels2 = inputs.pop("labels2", None)
        

        if return_outputs:
            return loss, outputs
        return loss
