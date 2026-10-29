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

# Step 3 - logreg_grad
import numpy as np


def logreg_grad(w, X, y):
    # TODO: Return X^T (sigmoid(X w) - y) / N.
    N = y.shape
    mu = sigmoid(X @ w)
    return X.T @ (mu-y) / N

# Step 4 - logreg_hessian
import numpy as np


def logreg_hessian(w, X):
    # TODO: Return X^T S X / N with S = diag(mu * (1 - mu)) and mu = sigmoid(X w),
    #       without forming the N x N matrix.
    N, _ = X.shape
    mu = sigmoid(X @ w)
    s = mu * (1-mu)
    return (X.T * s) @ X / N

# Step 5 - numerical_gradient
import numpy as np


def numerical_gradient(f, w, eps=1e-5):
    """Computes the numerical gradient of f at w using central differences.
    
    Works for scalars and NumPy arrays of arbitrary shape without mutating w.
    """
    w = np.asarray(w, dtype=float)
    g = np.zeros_like(w)
    
    for i in range(w.size):
        e = np.zeros_like(w)
        e.flat[i] = eps
        g.flat[i] = (f(w + e) - f(w - e)) / (2.0 * eps)
        
    return g

