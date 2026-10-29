"""
Bayesian and Robust Logistic Regression from Scratch scaffold.

Run this with: python scaffold.py
Uses functions defined in model.py.
"""

from model import *  # noqa: F401, F403 (pulls in your solution functions)

"""Bayesian and Robust Logistic Regression from Scratch: the full run (Murphy, PML Book 1, chapter 10).

Reproduces, at small scale, the chapter's experiments with the functions you built:
  1. derivative checks and the Iris fits (SGD vs IRLS, Fig 10.5 / Algorithm 10.1),
  2. polynomial features and weight decay (Fig 10.4 / 10.6),
  3. softmax regression on all three Iris species with Bohning's bound (10.3.4),
  4. the Laplace posterior on separable data, MC vs probit predictives (Fig 10.13 / 10.14),
     and a 100-task study of plug-in vs Bayesian predictions with twenty labels,
  5. robust logistic regression: the contaminated Iris set (Fig 10.10) and label noise far from
     and near the boundary with the mixture likelihood and the bi-tempered loss (Fig 10.12).
"""
import numpy as np


def main():
    # ---------------- Part 1-2: Iris, setosa vs versicolor, by sepal length ----------------
    X, y = load_iris_sepal()
    Xb, yb = X[y < 2], y[y < 2]
    w_probe = np.array([-20.0, 3.5])
    g_err = np.max(np.abs(logreg_grad(w_probe, Xb, yb) - numerical_gradient(lambda v: logreg_nll(v, Xb, yb), w_probe)))
    H_num = np.stack([numerical_gradient(lambda v: logreg_grad(v, Xb, yb)[j], w_probe) for j in range(2)])
    h_err = np.max(np.abs(logreg_hessian(w_probe, Xb) - H_num))
    print(f"Derivative check at w=(-20, 3.5): max |grad - finite diff| = {g_err:.1e}, max |Hessian - finite diff| = {h_err:.1e}")

    w_irls, h_irls = irls_logreg(Xb, yb)
    print(f"IRLS: converged in {len(h_irls) - 1} iterations, w = ({w_irls[0]:.3f}, {w_irls[1]:.3f}), "
          f"NLL = {h_irls[-1]:.5f}, decision boundary at {-w_irls[0] / w_irls[1]:.3f} cm of sepal length")
    for lr in (0.01, 0.1):
        w_sgd, h_sgd = sgd_logreg(Xb, yb, lr, 200, np.random.default_rng(0))
        print(f"SGD lr={lr}: NLL after 1 / 10 / 200 epochs = {h_sgd[0]:.4f} / {h_sgd[9]:.4f} / {h_sgd[-1]:.4f} "
              f"(gap to the IRLS optimum after 20,000 updates: {h_sgd[-1] - h_irls[-1]:.4f})")
    xs = (Xb[:, 1] - Xb[:, 1].mean()) / Xb[:, 1].std()
    Xs = np.column_stack([np.ones_like(xs), xs])
    _, h_std = sgd_logreg(Xs, yb, 0.1, 10, np.random.default_rng(0))
    print(f"SGD lr=0.1 on the standardized feature: gap after 10 epochs = {h_std[-1] - irls_logreg(Xs, yb)[1][-1]:.4f} "
          "(the ill-conditioned raw loss surface is what slows SGD down)")

    lams = np.logspace(-6, 0, 7)
    tr_all, te_all = [], []
    for seed in range(5):
        rng = np.random.default_rng(seed)
        X_tr, y_tr = make_ring_data(50, 0.1, rng)
        X_te, y_te = make_ring_data(2000, 0.1, rng)
        tr, te = weight_decay_sweep(polynomial_features(X_tr, 8), y_tr, polynomial_features(X_te, 8), y_te, lams)
        tr_all.append(tr)
        te_all.append(te)
    tr_m, te_m = np.mean(tr_all, axis=0), np.mean(te_all, axis=0)
    print("Weight decay on degree-8 polynomial features (45 features, 50 training points, 5 seeds, 10% label noise):")
    for lam, a, b in zip(lams, tr_m, te_m):
        print(f"  lambda={lam:.0e}  train error {a:.3f}  test error {b:.3f}")
    for deg in (1, 2):
        tr_d, te_d = [], []
        for seed in range(5):
            rng = np.random.default_rng(seed)
            X_tr, y_tr = make_ring_data(50, 0.1, rng)
            X_te, y_te = make_ring_data(2000, 0.1, rng)
            a, b = weight_decay_sweep(polynomial_features(X_tr, deg), y_tr, polynomial_features(X_te, deg), y_te, [1e-4])
            tr_d.append(a[0])
            te_d.append(b[0])
        print(f"  degree {deg} at lambda=1e-04: train error {np.mean(tr_d):.3f}  test error {np.mean(te_d):.3f}")

    # ---------------- Part 3: softmax regression by bound optimization ----------------
    W2, h2 = fit_softmax_mm(Xb, yb, 2, 400)
    n2 = int(np.argmax(-h2 / len(yb) - h_irls[-1] < 1e-6))
    print(f"Bohning MM, 2 classes: matches IRLS (w_0 = -w_irls: {np.allclose(-W2[0], w_irls, atol=1e-6)}), "
          f"needs {n2} iterations to get within 1e-6 nats of the optimum vs {len(h_irls) - 1} for IRLS, "
          "but inverts its matrix once")
    W3, h3 = fit_softmax_mm(X, y, 3, 400)
    acc3 = float(np.mean(np.argmax(np.column_stack([X @ W3.T, np.zeros(len(y))]), axis=1) == y))
    print(f"Bohning MM, 3 Iris species from sepal length: log-likelihood {h3[0]:.2f} -> {h3[-1]:.2f}, "
          f"never decreased: {bool(np.all(np.diff(h3) > -1e-9))}, training accuracy {acc3:.3f}")

    # ---------------- Part 4: Bayesian logistic regression ----------------
    Xsep, ysep = make_separable_data(30, 0.5, np.random.default_rng(0))
    norms = [np.linalg.norm(fit_map_newton(Xsep, ysep, lam)) for lam in (1e-2, 1e-4, 1e-6)]
    print("Separable data (30 points): ||w_map|| for lambda = 1e-2 / 1e-4 / 1e-6 = "
          + " / ".join(f"{v:.2f}" for v in norms) + "  (the MLE is at infinity)")
    w_map, Sigma = laplace_posterior(Xsep, ysep, 100.0)
    print(f"Laplace posterior with sigma^2=100: w_map = ({w_map[0]:.3f}, {w_map[1]:.3f}), "
          f"posterior sd = ({np.sqrt(Sigma[0, 0]):.3f}, {np.sqrt(Sigma[1, 1]):.3f}), "
          f"correlation = {Sigma[0, 1] / np.sqrt(Sigma[0, 0] * Sigma[1, 1]):.3f}")
    X_probe = np.array([[1.0, 1.0], [-6.0, 6.0], [4.0, -2.0], [-1.0, 2.0]])
    p_plug = sigmoid(X_probe @ w_map)
    p_mc = predictive_mc(X_probe, w_map, Sigma, 20000, np.random.default_rng(1))
    p_pr = predictive_probit(X_probe, w_map, Sigma)
    for xp, a, b, c in zip(X_probe, p_plug, p_mc, p_pr):
        print(f"  x=({xp[0]:+.0f}, {xp[1]:+.0f}): plug-in {a:.4f}  Monte Carlo {b:.4f}  probit {c:.4f}")
    print("Plug-in vs Bayesian predictive, 100 tasks, 20 training labels, 5 features, true weights ~ N(0, 4 I):")
    for prior_var in (100.0, 4.0):
        res = small_data_study(100, 20, 5, 4.0, prior_var, np.random.default_rng(0))
        print(f"  prior variance {prior_var:>5.0f}: " + "  ".join(f"{k} NLL {v[0]:.4f} ECE {v[1]:.4f}" for k, v in res.items()))

    # ---------------- Part 5: robust logistic regression ----------------
    out = np.array([4.2, 4.3, 4.4, 4.5, 4.4, 4.3])
    Xc = np.vstack([Xb, np.column_stack([np.ones(6), out])])
    yc = np.concatenate([yb, np.ones(6, dtype=int)])
    w_plain, _ = irls_logreg(Xc, yc)
    w_rob, pi_rob, h_rob = fit_robust_logreg(Xc, yc, 1e-3)
    r_out = clean_responsibility(w_rob, pi_rob, Xc, yc)[-6:]
    print("Contaminated Iris (6 versicolor labels at 4.2-4.5 cm, Fig 10.10): boundary "
          f"clean {-w_irls[0] / w_irls[1]:.3f} cm, plain fit {-w_plain[0] / w_plain[1]:.3f} cm, "
          f"mixture fit {-w_rob[0] / w_rob[1]:.3f} cm (pi = {pi_rob:.3f}, max P(clean) of the six = {r_out.max():.3f})")

    def accuracy(W, X_eval, y_eval):
        return float(np.mean(np.argmax(X_eval @ W.T, axis=1) == y_eval))

    u = np.array([0.0, 2.0, 1.0]) / np.sqrt(5.0)
    methods = {"logistic (t1=1, t2=1)": (1.0, 1.0), "bounded loss (t1=0.2)": (0.2, 1.0),
               "heavy tails (t2=4)": (1.0, 4.0), "bi-tempered (0.2, 4)": (0.2, 4.0)}
    for mode in ("far", "near"):
        scores = {k: [] for k in list(methods) + ["mixture likelihood"]}
        for seed in range(3):
            rng = np.random.default_rng(seed)
            Xn, yn0 = make_separable_data(200, 0.3, rng)
            Xt, yt = make_separable_data(2000, 0.3, rng)
            Xn = np.column_stack([np.ones(len(yn0)), Xn])
            Xt = np.column_stack([np.ones(len(yt)), Xt])
            yn = inject_label_noise(Xn, yn0, u, 20, mode)
            for name, (t1, t2) in methods.items():
                scores[name].append(accuracy(fit_bi_tempered(Xn, yn, 2, t1, t2, 0.5, 300), Xt, yt))
            w_m, _, _ = fit_robust_logreg(Xn, yn, 1e-3)
            scores["mixture likelihood"].append(float(np.mean((Xt @ w_m > 0) == (yt == 1))))
        where = "far from" if mode == "far" else "near"
        print(f"Label noise {where} the boundary (20 of 200 labels flipped, clean-test accuracy, mean of 3 seeds):")
        for name, v in scores.items():
            print(f"  {name:24s} {np.mean(v):.4f}")


if __name__ == "__main__":
    main()

