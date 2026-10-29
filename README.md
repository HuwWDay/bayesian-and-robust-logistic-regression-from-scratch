# Bayesian and Robust Logistic Regression from Scratch

Logistic regression is the model everyone thinks they know, until the data is small, separable or mislabeled. Build it from scratch in NumPy following chapter 10 of Murphy's Probabilistic Machine Learning: derive and check the gradient and Hessian, fit Iris by SGD and by iteratively reweighted least squares, tame polynomial features with weight decay, and fit softmax regression with Bohning's bound. Then go Bayesian: a Laplace posterior on separable data, Monte Carlo and probit predictives, and a 100-task study of when averaging over weights beats the plug-in with only twenty labels. Finish with robust logistic regression: a label-noise mixture likelihood fitted by EM on contaminated Iris, and the bi-tempered loss against label noise far from and near the decision boundary.

## How to run

```bash
python scaffold.py
```

## Steps

- [x] **1.** sigmoid
- [x] **2.** logreg_nll
- [x] **3.** logreg_grad
- [x] **4.** logreg_hessian
- [x] **5.** numerical_gradient
- [x] **6.** load_iris_sepal
- [x] **7.** sgd_logreg
- [x] **8.** irls_logreg
- [x] **9.** fit_map_newton
- [x] **10.** make_ring_data
- [x] **11.** polynomial_features
- [x] **12.** weight_decay_sweep
- [x] **13.** softmax_loglik_grad

---

Built on Deep-ML.
