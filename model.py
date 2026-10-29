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

# Step 6 - load_iris_sepal
import numpy as np

SETOSA_SEPAL_LENGTH = (
    "5.1 4.9 4.7 4.6 5.0 5.4 4.6 5.0 4.4 4.9 5.4 4.8 4.8 4.3 5.8 5.7 5.4 5.1 5.7 5.1 5.4 5.1 4.6 5.1 4.8 "
    "5.0 5.0 5.2 5.2 4.7 4.8 5.4 5.2 5.5 4.9 5.0 5.5 4.9 4.4 5.1 5.0 4.5 4.4 5.0 5.1 4.8 5.1 4.6 5.3 5.0"
)
VERSICOLOR_SEPAL_LENGTH = (
    "7.0 6.4 6.9 5.5 6.5 5.7 6.3 4.9 6.6 5.2 5.0 5.9 6.0 6.1 5.6 6.7 5.6 5.8 6.2 5.6 5.9 6.1 6.3 6.1 6.4 "
    "6.6 6.8 6.7 6.0 5.7 5.5 5.5 5.8 6.0 5.4 6.0 6.7 6.3 5.6 5.5 5.5 6.1 5.8 5.0 5.6 5.7 5.7 6.2 5.1 5.7"
)
VIRGINICA_SEPAL_LENGTH = (
    "6.3 5.8 7.1 6.3 6.5 7.6 4.9 7.3 6.7 7.2 6.5 6.4 6.8 5.7 5.8 6.4 6.5 7.7 7.7 6.0 6.9 5.6 7.7 6.3 6.7 "
    "7.2 6.2 6.1 6.4 7.2 7.4 7.9 6.4 6.3 6.1 7.7 6.3 6.4 6.0 6.9 6.7 6.9 5.8 6.8 6.7 6.7 6.3 6.5 6.2 5.9"
)


def load_iris_sepal():
    setosa = np.array(SETOSA_SEPAL_LENGTH.split(), dtype=float)
    versicolor = np.array(VERSICOLOR_SEPAL_LENGTH.split(), dtype=float)
    virginica = np.array(VIRGINICA_SEPAL_LENGTH.split(), dtype=float)

    lengths = np.concatenate([setosa, versicolor, virginica])
    ones = np.ones_like(lengths)

    X = np.column_stack([ones, lengths])
    y = np.repeat(np.arange(3), 50)

    return X, y

# Step 7 - sgd_logreg
import numpy as np


def sgd_logreg(X, y, lr, n_epochs, rng):
    # TODO: From w = 0, for each epoch visit rng.permutation(N) and apply w -= lr * (mu_n - y_n) * x_n;
    #       record logreg_nll after every epoch. Return (w, history array).
    N, D = X.shape
    w = np.zeros(D)
    history = []
    for epoch in range(n_epochs):
        visit = rng.permutation(N)
        for n in visit: 
            mu_n = sigmoid(X[n] @ w)
            w -= lr*(mu_n - y[n])*X[n]
        history.append(logreg_nll(w, X, y))
    return w, np.asarray(history)

# Step 8 - irls_logreg
import numpy as np


def irls_logreg(X, y, max_iter=50, tol=1e-10):
    # TODO: Algorithm 10.1. From w = 0, repeatedly form s = mu (1 - mu) and z = a + (y - mu) / s,
    #       solve (X^T S X) w = X^T S z, record the NLL, stop when the update is below tol.
    N, D = X.shape 
    w = np.zeros(D)
    history = [logreg_nll(w, X, y)]
    for i in range(max_iter):
        a = X @ w 
        mu = sigmoid(a)
        s = mu*(1-mu)
        z = a + (y-mu)/s 
        XtS = X.T *s
        w_new = np.linalg.solve(XtS @ X, XtS @ z)
        history.append(logreg_nll(w_new, X, y))
        if max(abs(w_new-w)) < tol:
            break 
        w = w_new
    return w, np.asarray(history)

