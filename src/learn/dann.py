from dataclasses import dataclass
import torch
from transformers import PreTrainedModel, ResNetConfig
from transformers.modeling_outputs import ModelOutput
from transformers.models.resnet.modeling_resnet import ResNetForImageClassification

@dataclass
class BranchedOutput(ModelOutput):
    """Defines the model output structure"""
    branch1_logits: torch.FloatTensor
    branch2_logits: torch.FloatTensor

class ResNetForMultiLabel(PreTrainedModel):
    """
    Defines the DANN (domain adversarial neural network) with a ResNet-50 feature extractor.
    This is a branched model.
    """
    config_class = ResNetConfig

    def __init__(self, config, num_classes: int, lamb: float):
        """
        num_classes: Number of possible values for output label y.
        """
        super().__init__(config)

        self.resnet = ResNetForImageClassification(config).resnet # ResNet-50
        self.pre_branch = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(2048, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5)
        )

        self.branch1 = torch.nn.Linear(512, num_classes)

        self.grad_reverse = FixedGradientReversal(alpha=lamb)

        self.domain_classifier = torch.nn.Linear(512, 1)
        
        self.branch2 = torch.nn.Sequential(
            self.grad_reverse,
            self.domain_classifier,
        )

        self.post_init()

    def forward(self, pixel_values, labels1, labels2):
        # Feature extractor G_f
        features = self.resnet(pixel_values).pooler_output
        features = features.flatten(1)
        features = self.pre_branch(features)

        # Label predictor G_y (branch for original labels)
        logits1 = self.branch1(features)

        # Gradient reversal layer R_lambda and
        # Domain classifier G_d (branch for domain labels)
        logits2 = self.branch2(features)

        return BranchedOutput(branch1_logits=logits1, branch2_logits=logits2)
