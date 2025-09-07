import torch
from transformers import PreTrainedModel, ResNetConfig
from transformers.models.resnet.modeling_resnet import ResNetForImageClassification

from src.architecture.output.single_output import SingleOutput

class ResNetSingleLabel(PreTrainedModel):
    config_class = ResNetConfig

    def __init__(self, config, num_classes=11):
        super().__init__(config)
        self.resnet = ResNetForImageClassification(config).resnet

        self.pre_branch = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(2048, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5)
        )
        self.branch1 = torch.nn.Linear(512, num_classes)
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.post_init()

    def forward(self, pixel_values, labels=None):
        features = self.resnet(pixel_values).pooler_output
        x = self.pre_branch(features)
        logits = self.branch1(x)

        loss = self.loss_fn(logits, labels) if labels is not None else None
        return SingleOutput(loss=loss, logits=logits)
