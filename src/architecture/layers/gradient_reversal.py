import torch

# Based on https://github.com/tadeephuy/GradientReversal.git
# [1] Yaroslav Ganin, & Victor Lempitsky. (2015). Unsupervised Domain Adaptation by Backpropagation.

class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.save_for_backward(torch.tensor(alpha, dtype=x.dtype, device=x.device))
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        alpha_tensor, = ctx.saved_tensors
        return -alpha_tensor * grad_output, None

class GradientReversal(torch.nn.Module):
    """x"""
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.alpha)
