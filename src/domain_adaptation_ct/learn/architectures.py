from dataclasses import dataclass
import torch
from transformers import PreTrainedModel, ResNetConfig
from transformers.modeling_outputs import ModelOutput
from transformers.models.resnet.modeling_resnet import ResNetForImageClassification

from domain_adaptation_ct.learn.gradient_reversal import GradientReversal
from domain_adaptation_ct.learn.loss import MaskedDomainAdversarialLoss

@dataclass
class SingleOutput(ModelOutput):
    """Defines the model output structure"""
    branch1_logits: torch.FloatTensor

class ResNet50Baseline(PreTrainedModel):
    """
    Baseline ResNet-50 model,
    with the same label predictor as the DANN variant for fairer comparison.
    """
    config_class = ResNetConfig

    def __init__(self, config, num_classes: int):
        """
        num_classes: Number of possible values for output label y.
        """
        super().__init__(config)
        # ResNet-50
        self.resnet = ResNetForImageClassification(config).resnet

        self.pre_branch = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(2048, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5)
        )
        self.branch1 = torch.nn.Linear(512, num_classes)

        # Loss function used only by trainer
        self.loss_fn = torch.nn.CrossEntropyLoss()

        if type(self) is ResNet50Baseline:
            # Children of this class should not call this post_init.
            self.post_init()

    def forward(self, pixel_values: torch.Tensor) -> SingleOutput:
        features = self.resnet(pixel_values).pooler_output
        features = self.pre_branch(features)

        logits = self.branch1(features)

        # Loss to be calculated outside of forward call.

        return SingleOutput(branch1_logits=logits)

@dataclass
class BranchedOutput(SingleOutput):
    """Defines the model output structure"""
    branch2_logits: torch.FloatTensor

class ResNet50DANN(ResNet50Baseline):
    """
    Defines the DANN (domain adversarial neural network) with a ResNet-50 feature extractor.
    This is a branched model.
    """

    def __init__(self, config, num_classes: int, lamb: float):
        """
        num_classes: Number of possible values for output label y.
        lamb: Initial value for lambda hyperparameter for gradient reversal layer.
        """
        # Inherit from ResNet50Baseline
        super().__init__(config, num_classes)

        self.grad_reverse = GradientReversal(lamb=lamb)

        self.domain_classifier = torch.nn.Linear(512, 1)
        
        self.branch2 = torch.nn.Sequential(
            self.grad_reverse,
            self.domain_classifier,
        )

        # Loss function used only by trainer.
        self.loss_fn = MaskedDomainAdversarialLoss()

        self.post_init()

    def forward(self, pixel_values) -> BranchedOutput:
        # Feature extractor G_f
        features = self.resnet(pixel_values).pooler_output
        features = self.pre_branch(features)

        # Label predictor G_y (branch for original labels)
        logits1 = self.branch1(features)

        # Gradient reversal layer R_lambda and
        # Domain classifier G_d (branch for domain labels)
        logits2 = self.branch2(features)

        # Loss to be calculated outside of forward call.

        return BranchedOutput(branch1_logits=logits1, branch2_logits=logits2)


