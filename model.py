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

# Step 9 - fit_map_newton
import numpy as np


def fit_map_newton(X, y, lam, max_iter=100, tol=1e-10):
    """Find MAP weights under a Gaussian prior using Newton's method with backtracking.

    Minimizes PNLL(w) = NLL(w) + lam * w^T w over all D dimensions.
    """
    N, D = X.shape
    w = np.zeros(D, dtype=float)

    def pnll(w_eval):
        return logreg_nll(w_eval, X, y) + lam * np.dot(w_eval, w_eval)

    I = np.eye(D)

    for _ in range(max_iter):
        g = logreg_grad(w, X, y) + 2.0 * lam * w
        H = logreg_hessian(w, X) + 2.0 * lam * I

        step = np.linalg.solve(H, g)

        eta = 1.0
        current_pnll = pnll(w)

        # Backtracking line search
        while pnll(w - eta * step) > current_pnll and eta > 1e-10:
            eta /= 2.0

        update = eta * step
        w = w - update

        if np.max(np.abs(update)) < tol:
            break

    return w

# Step 10 - make_ring_data
import numpy as np


def make_ring_data(n, flip_prob, rng):
    """Generate a noisy two-class dataset with a circular decision boundary.

    Parameters
    ----------
    n : int
        Number of data points to generate.
    flip_prob : float
        Probability of flipping each label.
    rng : np.random.Generator or np.random.RandomState
        Random number generator.

    Returns
    -------
    X : np.ndarray of shape (n, 2)
        Feature matrix uniformly distributed in [-1, 1]^2.
    y : np.ndarray of shape (n,)
        Integer binary labels in {0, 1}.
    """
    # 1. Draw features uniformly on [-1, 1]^2
    X = rng.uniform(-1.0, 1.0, size=(n, 2))

    # 2. Clean labels: 1 inside x1^2 + x2^2 < 0.5, 0 outside
    radius_sq = X[:, 0] ** 2 + X[:, 1] ** 2
    y = (radius_sq < 0.5).astype(int)

    # 3. Draw random uniforms once and flip where draw < flip_prob
    flips = rng.random(n) < flip_prob
    y[flips] = 1 - y[flips]

    return X, y

# Step 11 - polynomial_features
import numpy as np


def polynomial_features(X, degree):
    """Expand 2-D input into all 2D monomials up to total degree `degree`.

    Parameters
    ----------
    X : np.ndarray of shape (N, 2)
        Two-dimensional input features [x1, x2].
    degree : int
        Maximum polynomial degree K.

    Returns
    -------
    Phi : np.ndarray of shape (N, M)
        Expanded feature matrix where M = (degree + 1) * (degree + 2) // 2.
        Columns are ordered by total degree d = 0..degree, and within degree d
        by decreasing power i of x1 from d down to 0:
        [1, x1, x2, x1^2, x1*x2, x2^2, ..., x2^K].
    """
    x1 = X[:, 0]
    x2 = X[:, 1]
    
    out = []
    for d in range(degree + 1):
        for i in range(d, -1, -1):
            out.append((x1 ** i) * (x2 ** (d - i)))
            
    return np.column_stack(out)

# Step 12 - weight_decay_sweep
import numpy as np


def weight_decay_sweep(X_train, y_train, X_test, y_test, lams):
    # TODO: For each lam fit fit_map_newton on the training set and record the train and test
    #       misclassification rates (predict 1 when X @ w > 0). Return two arrays.
    train_err, test_err = [], []
    for lam in lams:
        w = fit_map_newton(X_train, y_train, lam)
        train_err.append(np.mean((X_train @ w > 0) != (y_train == 1)))
        test_err.append(np.mean((X_test @ w > 0) != (y_test == 1)))
    return np.asarray(train_err), np.asarray(test_err)

# Step 13 - softmax_loglik_grad
import numpy as np


def softmax_loglik_grad(W, X, y, C):
    """Compute multinomial log-likelihood and gradient with last class clamped to 0.

    Parameters
    ----------
    W : np.ndarray of shape (C - 1, D)
        Weight matrix for the first C - 1 free classes.
    X : np.ndarray of shape (N, D)
        Input design matrix.
    y : np.ndarray of shape (N,)
        Integer class labels in {0, ..., C - 1}.
    C : int
        Total number of classes.

    Returns
    -------
    ll : float
        Summed log-likelihood (scalar float).
    G : np.ndarray of shape (C - 1, D)
        Gradient of the log-likelihood with respect to W.
    """
    N, D = X.shape

    # 1. Compute logits of shape (N, C) with class C - 1 fixed to 0
    # X @ W.T has shape (N, C - 1); zeros column has shape (N, 1)
    zeros = np.zeros((N, 1), dtype=float)
    A = np.hstack([X @ W.T, zeros])  # (N, C)

    # 2. Numerically stable Log-Sum-Exp along class axis (axis=1)
    a_max = np.max(A, axis=1, keepdims=True)
    exp_shifted = np.exp(A - a_max)
    sum_exp = np.sum(exp_shifted, axis=1, keepdims=True)
    lse = a_max.squeeze(axis=1) + np.log(sum_exp.squeeze(axis=1))

    # Log-likelihood: sum_n (A[n, y[n]] - lse[n])
    a_y = A[np.arange(N), y]
    ll = float(np.sum(a_y - lse))

    # 3. Softmax probabilities mu of shape (N, C)
    mu = exp_shifted / sum_exp

    # 4. One-hot indicator matrix Y of shape (N, C)
    Y = np.zeros_like(A)
    Y[np.arange(N), y] = 1.0

    # Gradient for free classes c in {0, ..., C - 2}:
    # G = (Y[:, :C-1] - mu[:, :C-1])^T @ X -> shape (C - 1, D)
    diff = Y[:, :C - 1] - mu[:, :C - 1]
    G = diff.T @ X

    return ll, G

# Step 14 - bohning_bound
import numpy as np


def bohning_bound(X, C):
    # TODO: Return -0.5 * kron(I - 11^T / C, X^T X) with I and 1 of size C - 1.
    K = C - 1
    M = np.eye(K) - np.ones((K, K)) / C 
    return -0.5 * np.kron(M, X.T @ X)

# Step 15 - fit_softmax_mm
import numpy as np


def fit_softmax_mm(X, y, C, n_iter):
    """Fit multiclass softmax regression using Böhning-bound Minorize-Maximize (MM).

    Parameters
    ----------
    X : np.ndarray of shape (N, D)
        Design matrix.
    y : np.ndarray of shape (N,)
        Integer labels in {0, ..., C - 1}.
    C : int
        Total number of classes.
    n_iter : int
        Number of MM iterations to perform.

    Returns
    -------
    W : np.ndarray of shape (C - 1, D)
        Learned weight matrix for the first C - 1 free classes.
    history : np.ndarray of shape (n_iter + 1,)
        Summed log-likelihood at initialization and after each iteration.
    """
    N, D = X.shape
    W = np.zeros((C - 1, D), dtype=float)

    # Precompute B^{-1} once outside the optimization loop
    B = bohning_bound(X, C)
    B_inv = np.linalg.inv(B)

    # Initial log-likelihood
    init_ll, _ = softmax_loglik_grad(W, X, y, C)
    history = [init_ll]

    for _ in range(n_iter):
        ll, G = softmax_loglik_grad(W, X, y, C)

        # Class-major flattening (row-major order: class 0, then class 1, etc.)
        w = W.reshape(-1)
        g = G.reshape(-1)

        # MM update: w <- w - B^{-1} g(w)
        w = w - B_inv @ g

        # Reshape back to (C - 1, D)
        W = w.reshape(C - 1, D)

        # Record log-likelihood after the step
        step_ll, _ = softmax_loglik_grad(W, X, y, C)
        history.append(step_ll)

    return W, np.asarray(history)

# Step 16 - make_separable_data
import numpy as np


def make_separable_data(n, margin, rng):
    """Generate 2-D data linearly separable through the origin by a given margin.

    Parameters
    ----------
    n : int
        Number of points to generate.
    margin : float
        Separation margin along unit normal u.
    rng : np.random.Generator or np.random.RandomState
        Random number generator.

    Returns
    -------
    X : np.ndarray of shape (n, 2)
        Point coordinates shifted to ensure separation.
    y : np.ndarray of shape (n,)
        Integer labels in {0, 1}.
    """
    # 1. Draw points from N(0, 2^2 I)
    X = rng.normal(0.0, 2.0, size=(n, 2))

    # 2. Fixed unit normal direction
    u = np.array([2.0, 1.0]) / np.sqrt(5.0)

    # 3. Label: 1 where X @ u > 0, else 0
    y = (X @ u > 0).astype(int)

    # 4. Push points away along normal u: +margin * u for y=1, -margin * u for y=0
    signs = 2 * y - 1  # maps 0 -> -1 and 1 -> +1
    X = X + margin * np.outer(signs, u)

    return X, y

# Step 17 - laplace_posterior
import numpy as np


def laplace_posterior(X, y, prior_var):
    """Approximate the weight posterior by a Gaussian at the MAP estimate.

    Parameters
    ----------
    X : np.ndarray of shape (N, D)
        Design matrix.
    y : np.ndarray of shape (N,)
        Binary labels in {0, 1}.
    prior_var : float
        Prior variance sigma^2 in p(w) = N(w | 0, sigma^2 * I).

    Returns
    -------
    w_map : np.ndarray of shape (D,)
        MAP weight estimate.
    Sigma : np.ndarray of shape (D, D)
        Posterior covariance matrix (inverse Hessian of negative log posterior).
    """
    N, D = X.shape

    # 1. Compute MAP estimate via Newton's method
    # Since fit_map_newton minimizes: NLL(w) + lam * ||w||^2 where NLL is the average over N,
    # the summed objective: sum_n -log p(y_n | x_n, w) + ||w||^2 / (2 * prior_var)
    # equals: N * [NLL(w) + lam * ||w||^2] with lam = 1 / (2 * N * prior_var).
    lam = 1.0 / (2.0 * N * prior_var)
    w_map = fit_map_newton(X, y, lam)

    # 2. Compute posterior covariance Sigma = H^{-1}
    # logreg_hessian returns the sample-averaged Hessian (1/N) * X^T S X.
    # The Hessian of the summed negative log posterior is:
    # H = N * logreg_hessian(w_map, X) + (1 / prior_var) * I
    H_sum = N * logreg_hessian(w_map, X) + (1.0 / prior_var) * np.eye(D)
    Sigma = np.linalg.inv(H_sum)

    return w_map, Sigma

# Step 18 - predictive_mc
import numpy as np


def predictive_mc(X_test, w_map, Sigma, n_samples, rng):
    # TODO: L = cholesky(Sigma); Z = rng.standard_normal((n_samples, D)); Ws = w_map + Z @ L.T;
    #       return the mean over samples of sigmoid(X_test @ Ws.T), shape (M,).
    M, D = X_test.shape
    L = np.linalg.cholesky(Sigma)
    Z = rng.standard_normal((n_samples, D))
    Ws = w_map + Z @ L.T
    out = X_test @ Ws.T 
    return np.mean(sigmoid(out), axis=1)

# Step 19 - predictive_probit
import numpy as np


def predictive_probit(X_test, w_map, Sigma):
    # TODO: m = X w_map, v = x^T Sigma x per row, kappa = (1 + pi v / 8)^(-1/2); return sigmoid(kappa m).
    m = X_test @ w_map
    v = np.einsum("nd,de,ne->n", X_test, Sigma, X_test)
    return sigmoid(m / np.sqrt(1+np.pi*v/8))

# Step 20 - calibration_metrics
import numpy as np


def calibration_metrics(p, y, n_bins=10):
    """Compute mean log loss (NLL) and Expected Calibration Error (ECE).

    Parameters
    ----------
    p : np.ndarray of shape (M,)
        Predicted probabilities for class 1.
    y : np.ndarray of shape (M,)
        Binary ground-truth labels in {0, 1}.
    n_bins : int, default=10
        Number of equal-width bins on [0, 1].

    Returns
    -------
    nll : float
        Mean negative log-likelihood (binary cross-entropy).
    ece : float
        Expected Calibration Error over non-empty bins.
    """
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    M = p.size

    # 1. Clip predictions to avoid log(0)
    p = np.clip(p, 1e-12, 1.0 - 1e-12)

    # 2. Mean negative log-likelihood (binary cross-entropy)
    nll = -np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))

    # 3. Bin assignments: b = min(floor(p * n_bins), n_bins - 1)
    # Using np.clip keeps the last bin [..., 1.0] closed at 1.0
    bin_indices = np.clip(np.floor(p * n_bins).astype(int), 0, n_bins - 1)

    # 4. Expected Calibration Error (ECE)
    ece = 0.0
    for b in range(n_bins):
        mask = bin_indices == b
        count = np.sum(mask)
        if count > 0:
            p_bin_mean = np.mean(p[mask])
            y_bin_mean = np.mean(y[mask])
            ece += (count / M) * np.abs(y_bin_mean - p_bin_mean)

    return float(nll), float(ece)

# Step 21 - small_data_study
import numpy as np


def small_data_study(
    n_tasks,
    n_train,
    dim,
    true_var,
    prior_var,
    rng,
    n_test=200,
    n_samples=1000,
):
    """Compare plug-in, probit, and Monte Carlo predictive distributions across random tasks.

    Parameters
    ----------
    n_tasks : int
        Number of independent synthetic tasks to simulate.
    n_train : int
        Number of training examples per task.
    dim : int
        Feature dimensionality D.
    true_var : float
        Variance of the true weight vector.
    prior_var : float
        Prior variance sigma^2 passed to the Laplace posterior.
    rng : np.random.Generator or np.random.RandomState
        Random number generator.
    n_test : int, default=200
        Number of test examples per task.
    n_samples : int, default=1000
        Number of Monte Carlo weight samples for predictive_mc.

    Returns
    -------
    dict
        Mapping {"plugin": (mean_nll, mean_ece), "probit": (...), "mc": (...)}.
    """
    scores = {
        "plugin": {"nll": [], "ece": []},
        "probit": {"nll": [], "ece": []},
        "mc": {"nll": [], "ece": []},
    }

    std_true = np.sqrt(true_var)

    for _ in range(n_tasks):
        # 1. w_true drawn first
        w_true = rng.normal(0.0, std_true, size=dim)

        # 2. X drawn second
        X = rng.normal(size=(n_train + n_test, dim))

        # 3. y drawn third
        y = (rng.random(n_train + n_test) < sigmoid(X @ w_true)).astype(int)

        # 4. Split train / test
        X_train, X_test = X[:n_train], X[n_train:]
        y_train, y_test = y[:n_train], y[n_train:]

        # 5. Fit Laplace approximation at MAP
        w_map, Sigma = laplace_posterior(X_train, y_train, prior_var)

        # 6. Predictions in exact order: plugin, probit, mc
        p_plugin = sigmoid(X_test @ w_map)
        p_probit = predictive_probit(X_test, w_map, Sigma)
        p_mc = predictive_mc(X_test, w_map, Sigma, n_samples, rng)

        # 7. Evaluate calibration metrics
        preds = {
            "plugin": p_plugin,
            "probit": p_probit,
            "mc": p_mc,
        }

        for method, p in preds.items():
            nll, ece = calibration_metrics(p, y_test)
            scores[method]["nll"].append(nll)
            scores[method]["ece"].append(ece)

    return {
        method: (
            float(np.mean(vals["nll"])),
            float(np.mean(vals["ece"])),
        )
        for method, vals in scores.items()
    }

# Step 22 - inject_label_noise
import numpy as np


def inject_label_noise(X, y, w_ref, n_flip, mode):
    """Flip labels either near the boundary or deep inside class 0.

    Parameters
    ----------
    X : np.ndarray of shape (N, D)
        Input features.
    y : np.ndarray of shape (N,)
        Binary labels in {0, 1}.
    w_ref : np.ndarray of shape (D,)
        Reference weight vector defining margins m = X @ w_ref.
    n_flip : int
        Number of labels to flip.
    mode : {"near", "far"}
        - "near": flip the n_flip points with the smallest |m|.
        - "far": flip the n_flip class-0 points with the most negative m.

    Returns
    -------
    y_noisy : np.ndarray of shape (N,)
        New array with flipped labels. Does not modify the input y.
    """
    if mode not in ("near", "far"):
        raise ValueError(f"Unknown mode: {mode!r}. Expected 'near' or 'far'.")

    # Copy to avoid modifying the input array
    y_noisy = np.array(y, copy=True)
    m = X @ w_ref

    if mode == "near":
        # Smallest abs(m) regardless of class
        sorted_indices = np.argsort(np.abs(m), kind="stable")
        flip_indices = sorted_indices[:n_flip]
        y_noisy[flip_indices] = 1 - y_noisy[flip_indices]

    elif mode == "far":
        # Among points with y == 0, find those with the most negative m
        class_0_indices = np.flatnonzero(y == 0)
        # Sort class-0 points by increasing m (most negative first)
        sorted_order = np.argsort(m[class_0_indices], kind="stable")
        flip_indices = class_0_indices[sorted_order[:n_flip]]
        y_noisy[flip_indices] = 1

    return y_noisy

# Step 23 - mixture_nll
import numpy as np


def mixture_nll(w, pi, X, y):
    # TODO: q_n = probability the logistic model gives the observed label; p_n = pi / 2 + (1 - pi) q_n;
    #       return -mean(log p_n) as a float.
    mu = sigmoid(X @ w)
    q = np.where(y==1, mu, 1-mu)
    p = 0.5* pi + (1-pi)*q
    return float(-np.mean(np.log(p)))

