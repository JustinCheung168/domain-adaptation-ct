from dataclasses import dataclass
from typing import Optional

import torch
from transformers.modeling_outputs import ModelOutput

# Define the model output structure
@dataclass
class BranchedOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    branch1_logits: torch.FloatTensor = None
    branch2_logits: torch.FloatTensor = None
