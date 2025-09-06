
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from datasets import load_dataset
from transformers import Trainer, TrainingArguments, PreTrainedModel, ResNetConfig
from transformers.modeling_outputs import ModelOutput
from transformers.models.resnet.modeling_resnet import ResNetForImageClassification
from dataclasses import dataclass
from typing import Optional
from medmnist import OrganAMNIST
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

# [1] Yaroslav Ganin, & Victor Lempitsky. (2015). Unsupervised Domain Adaptation by Backpropagation.
from gradient_reversal import GradientReversal

# Define the model output structure
@dataclass
class BranchedOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    branch1_logits: torch.FloatTensor = None
    branch2_logits: torch.FloatTensor = None


class FixedGradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.save_for_backward(torch.tensor(alpha, dtype=x.dtype, device=x.device))
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        alpha_tensor, = ctx.saved_tensors
        return -alpha_tensor * grad_output, None


class FixedGradientReversal(torch.nn.Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        return FixedGradientReversalFunction.apply(x, self.alpha)


# Define the branched model
class ResNetForMultiLabel(PreTrainedModel):
    config_class = ResNetConfig

    def __init__(self, config, num_d1_classes=11, num_d2_classes=5, loss_fn=torch.nn.CrossEntropyLoss(), lamb = 0.25, ld_scale = 1.0):
        super().__init__(config)
        self.resnet = ResNetForImageClassification(config).resnet

        self.pre_branch = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(2048, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5)
        )
        self.branch1 = torch.nn.Linear(512, num_d1_classes) # Branch for original labels

        #self.branch2 = torch.nn.Linear(512, num_d2_classes) # Branch for distortion labels

        # self.grad_reverse = GradientReversal(alpha=lamb)  # Gradient reversal layer
        self.grad_reverse = FixedGradientReversal(alpha=lamb)
        self.branch2 = torch.nn.Sequential(
            self.grad_reverse,
            torch.nn.Linear(512, num_d2_classes)
            
        )

        self.ld_scale = ld_scale
        self.loss_fn = loss_fn
        self.post_init()

    def forward(self, pixel_values, labels1=None, labels2=None):
        features = self.resnet(pixel_values).pooler_output
        features = features.flatten(1)

        x = self.pre_branch(features)
        logits1 = self.branch1(x)
        logits2 = self.branch2(x)

        loss = self.loss_fn(logits1, labels1)
        loss = (loss * (1 - labels2)).mean()

        loss2 = self.loss_fn(logits2, labels2)
        total_loss = loss + (self.ld_scale * loss2)

        return BranchedOutput(loss=total_loss, branch1_logits=logits1, branch2_logits=logits2)

# Prepare OrganAMNIST Dataset with dummy secondary labels for now
class OrganAMNISTDataset(Dataset):
    def __init__(self, split="train"):
        dataset = OrganAMNIST(split=split, size=224, download=True)
        self.data = dataset
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1))  # Grayscale to 3-channel
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, label1 = self.data[idx]  # Unpack the tuple
        image = self.transform(image)
        label1 = int(label1)
        label2 = label1 % 5  # Dummy secondary label
        return {"pixel_values": image, "labels1": label1, "labels2": label2}

class CustomImageDataset(Dataset):
    def __init__(self, images, labels1, labels2, transform=None):
        # Need to add shuffling
        self.images = images  # Should be torch.Tensor of shape [N, 3, 224, 224]
        self.labels1 = labels1
        self.labels2 = labels2

        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.repeat(3, 1, 1))  # Grayscale to 3-channel
            ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        label1 = int(self.labels1[idx])
        label2 = int(self.labels2[idx])

        if self.transform:
            img = self.transform(img)

        # return {"pixel_values": img, "labels1": label1, "labels2": label2}
        return {
            "pixel_values": img,
            "labels1": int(label1) if torch.is_tensor(label1) else label1,
            "labels2": int(label2) if torch.is_tensor(label2) else label2,
                }

from transformers import TrainerCallback

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


class CustomTrainer(Trainer):
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

def lambda_scheduler(epoch, total_epochs):

    p = epoch / total_epochs

    lambda_p = 2. / (1. + np.exp(-10 * p)) - 1
    return float(lambda_p)

def train_model(train_dataset, eval_dataset, model, output_dir="./results", num_epochs=3, batch_size=32, train=True):
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir='./logs',
        logging_steps=10,
        load_best_model_at_end=True,
        learning_rate=0.1,
        weight_decay=1e-4,
        seed=42,
        optim="sgd"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,  # For simplicity, using the same dataset for evaluation
        compute_metrics=make_metrics_fn(model),
        callbacks=[LambdaUpdateCallback(model, lambda_scheduler, num_epochs)]
    )

    # trainer = CustomTrainer(
    #     model=model,
    #     args=training_args,
    #     train_dataset=train_dataset,
    #     eval_dataset=eval_dataset,
    #     compute_metrics=make_metrics_fn(model),
    #     lambda_scheduler=lambda_scheduler
    # )

    if train:
        trainer.train()
    return trainer

# Metrics function
# def compute_metrics(eval_pred):
#     preds, labels = eval_pred
#     logits1, logits2 = preds

#     labels1 = labels[0]
#     labels2 = labels[1]

#     # Convert to NumPy
#     preds1 = np.argmax(logits1, axis=-1)
#     preds2 = np.argmax(logits2, axis=-1)

#     labels1 = np.array(labels1)
#     labels2 = np.array(labels2)

#     return {
#         "accuracy_branch1": accuracy_score(labels1, preds1),
#         "precision_branch1": precision_score(labels1, preds1, average="macro", zero_division=0),
#         "recall_branch1": recall_score(labels1, preds1, average="macro", zero_division=0),
#         "f1_branch1": f1_score(labels1, preds1, average="macro", zero_division=0),

#         "accuracy_branch2": accuracy_score(labels2, preds2),
#         "precision_branch2": precision_score(labels2, preds2, average="macro", zero_division=0),
#         "recall_branch2": recall_score(labels2, preds2, average="macro", zero_division=0),
#         "f1_branch2": f1_score(labels2, preds2, average="macro", zero_division=0),
#    }

def make_metrics_fn(model):
    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        logits1, logits2 = preds

        labels1 = labels[0]
        labels2 = labels[1]

        # Convert to NumPy
        preds1 = np.argmax(logits1, axis=-1)
        preds2 = np.argmax(logits2, axis=-1)

        labels1 = np.array(labels1)
        labels2 = np.array(labels2)

        # # Print current lambda (from GradientReversalLayer)
        # if hasattr(model, 'grad_reverse'):
        #     print(f"[Metrics] Current Lambda: {model.grad_reverse.alpha:.4f}")

        return {
            "accuracy_branch1": accuracy_score(labels1, preds1),
            "precision_branch1": precision_score(labels1, preds1, average="macro", zero_division=0),
            "recall_branch1": recall_score(labels1, preds1, average="macro", zero_division=0),
            "f1_branch1": f1_score(labels1, preds1, average="macro", zero_division=0),
            "accuracy_branch2": accuracy_score(labels2, preds2),
            "precision_branch2": precision_score(labels2, preds2, average="macro", zero_division=0),
            "recall_branch2": recall_score(labels2, preds2, average="macro", zero_division=0),
            "f1_branch2": f1_score(labels2, preds2, average="macro", zero_division=0),
            "lambda": model.grad_reverse.alpha if hasattr(model, 'grad_reverse') else None
        }
    return compute_metrics

def dataset_load(file_path):

    data = np.load(file_path, allow_pickle=True)
    images = data['images']
    labels1 = data['labels1']
    labels2 = data['labels2']

    return CustomImageDataset(images, labels1, labels2)