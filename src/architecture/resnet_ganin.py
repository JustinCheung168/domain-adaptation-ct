import torch
from transformers import PreTrainedModel, ResNetConfig
from transformers.models.resnet.modeling_resnet import ResNetForImageClassification

from src.architecture.layers.gradient_reversal import GradientReversal
from src.architecture.output.branched_output import BranchedOutput

class ResNetForMultiLabel(PreTrainedModel):
    """Branched model from Ganin & Lempitsky"""
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

        self.grad_reverse = GradientReversal(alpha=lamb)
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