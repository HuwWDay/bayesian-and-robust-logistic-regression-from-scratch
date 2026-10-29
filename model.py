"""
Bayesian and Robust Logistic Regression from Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - sigmoid
import numpy as np


def sigmoid(a):
    """Numerically stable elementwise sigmoid mapping logits to probabilities.
    
    Prevents floating-point overflow for large positive or negative values by 
    ensuring evaluated exponents are never positive:
      - For a >= 0: 1 / (1 + exp(-a))
      - For a < 0:  exp(a) / (1 + exp(a))
    """
    a = np.asarray(a, dtype=float)
    out = np.empty_like(a)
    
    mask = a >= 0
    
    # a >= 0: -a <= 0, exp(-a) in (0, 1]
    out[mask] = 1.0 / (1.0 + np.exp(-a[mask]))
    
    # a < 0: a < 0, exp(a) in (0, 1)
    neg_mask = ~mask
    exp_a = np.exp(a[neg_mask])
    out[neg_mask] = exp_a / (1.0 + exp_a)
    
    return out

# Step 2 - logreg_nll
import numpy as np


def logreg_nll(w, X, y):
    # TODO: Compute the logits a = X @ w and return the mean of log(1 + exp(a)) - y * a as a float,
    #       using a stable softplus.
    a = X @ w 
    out = np.logaddexp(0.0, a)- y*a
    return float(out.mean())

