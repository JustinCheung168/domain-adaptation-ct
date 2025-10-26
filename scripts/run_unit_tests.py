#!/usr/bin/env python3
import sys
import os

# Add root of repo to Python path.
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

# Enable logging.
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", handlers=[logging.StreamHandler()])

def unit_test_loss():
    from domain_adaptation_ct.learn.loss import MaskedDomainAdversarialLoss

    import torch
    
    logging.info("Running unit_test_loss.")
    torch.manual_seed(0)
    N = 10
    K = 5
    logits1 = torch.randn(N, K, requires_grad=True)
    logits2 = torch.randn(N, requires_grad=True)
    labels1 = torch.randint(0, K, (N,))
    labels2 = torch.randint(0, 2, (N,))
    ld_scale = 0.5

    logging.info("Raw logits for labels:\n\t%s", logits1)
    logging.info("Raw logits for domains:\n\t%s", logits2)
    logging.info("True labels:\n\t%s", labels1)
    logging.info("True domains:\n\t%s", labels2)

    loss_fn = MaskedDomainAdversarialLoss()
    loss_val = loss_fn(logits1, logits2, labels1, labels2, ld_scale)
    loss_val.backward()

    logging.info("Gradient of loss w.r.t. label logits:\n\t%s", logits1.grad)
    logging.info("Gradient of loss w.r.t. domain logits:\n\t%s", logits2.grad)

    zero_label_gradient_instance_indices, = torch.where(torch.sum(torch.abs(logits1.grad),dim=1) == 0.0)
    zero_domain_gradient_instance_indices, = torch.where(logits2.grad == 0.0)
    target_domain_instances, = torch.where(labels2 == 1)
    
    logging.info("Indices of instances whose label predictions made no contribution to the loss gradient:%s", zero_label_gradient_instance_indices)
    logging.info("Indices of instances whose domain predictions made no contribution to the loss gradient:%s", zero_domain_gradient_instance_indices)
    logging.info("Indices from the target domain:%s", target_domain_instances)

    assert torch.equal(zero_label_gradient_instance_indices, target_domain_instances), "Gradient of loss w.r.t. target domain instances should be 0 for all target domain instances."
    assert zero_domain_gradient_instance_indices.numel() == 0, "Every instance should affect the gradient of loss for the domain classifier."

    logging.info("unit_test_loss passed.")


def unit_test_gradient_reversal():
    from domain_adaptation_ct.learn.gradient_reversal import GradientReversal
    
    import torch

    logging.info("Running unit_test_gradient_reversal.")
    torch.manual_seed(0)
    N = 10
    K = 5
    alpha = 0.7
    f = torch.randn(N, K, requires_grad=True)

    logging.info("Features:\n\t%s", f)
    
    R_lambda = GradientReversal(alpha)

    f_after_gradient_reversal = R_lambda(f)
    assert torch.equal(f, f_after_gradient_reversal), "Forward pass should not change input."

    # Use the values of the gradient in a scalar to mimic backpropagation.
    # By the linearity of gradient, the gradient of loss with respect to the sum of all elements should equal the sum of the gradients of loss with respect to every element.
    # Expect gradient of loss with respect to every element in the matrix to have a gradient of negative alpha.
    mock_loss_value = f_after_gradient_reversal.sum()
    mock_loss_value.backward()

    logging.info("Gradient of loss w.r.t. f:\n\t%s", f.grad)
    expected_grad = -alpha * torch.ones_like(f)
    assert torch.all(f.grad == expected_grad), "Backward pass should scale gradient by negative alpha."

    print("unit_test_gradient_reversal passed.")


if __name__ == "__main__":
    print("Now running unit tests. If this script crashes, then one of the unit tests failed.")
    unit_test_loss()
    unit_test_gradient_reversal()
    print("All tests passed.")


