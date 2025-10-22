import torch

class GradientReversalFunction(torch.autograd.Function):
    """Adapted from https://github.com/tadeephuy/GradientReversal/tree/master"""
    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: float) -> torch.Tensor:
        ctx.save_for_backward(torch.tensor(alpha, dtype=x.dtype, device=x.device))
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        alpha_tensor, = ctx.saved_tensors
        return -alpha_tensor * grad_output, None


class GradientReversal(torch.nn.Module):
    """Adapted from https://github.com/tadeephuy/GradientReversal/tree/master"""
    def __init__(self, alpha: float):
        super().__init__()
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return GradientReversalFunction.apply(x, self.alpha)
