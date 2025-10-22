import numpy as np

def logistic_increasing_lambda_scheduler(epoch, total_epochs):
    """
    Based on Ganin, Y., & Lempitsky, V. (2015, June). Unsupervised domain adaptation by backpropagation.
    """
    progress = epoch / total_epochs
    lambda_p = 2. / (1. + np.exp(-10 * progress)) - 1
    return float(lambda_p)

def linear_increasing_lambda_scheduler(epoch, total_epochs):
    return epoch / total_epochs

def linear_decreasing_lambda_scheduler(epoch, total_epochs):
    return 1 - (epoch / total_epochs)

def constant_lambda_scheduler(epoch=None, total_epochs=None, lambda_value=0.5):
    return lambda_value

def parabolic_increasing_lambda_scheduler(epoch, total_epochs, start_value=0.0, end_value=1.0):
    progress = epoch / total_epochs
    return start_value + (end_value - start_value) * (progress ** 2)

def parabolic_decreasing_lambda_scheduler(epoch, total_epochs, start_value=0.0, end_value=1.0):
    progress = epoch / total_epochs
    return end_value - (end_value - start_value) * (progress ** 2)
