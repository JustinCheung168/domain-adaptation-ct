from dataclasses import dataclass
from typing import Optional

import torch
from transformers.modeling_outputs import ModelOutput

# Define the model output structure
@dataclass
class SingleOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
