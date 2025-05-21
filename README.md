# Joint Partial Regression

`graph-jpr` is a Python package written in Rust that implements joint partial regression and Huber lasso regression.

## Installation

To install this package, execute

```python
pip install graph-jpr
```

Import `JointPartialRegression` and `SparseHuber` by adding

```python
from graph_jpr import JointPartialRegression, SparseHuber
```

to the top of your Python code.

## Usage and features

To fit a model, instantiate a model object and call the `fit` method, e.g.,

```python
import numpy as np
from graph_jpr import JointPartialRegression

X = np.random.normal(size=(100, 10))
model = JointPartialRegression()
precision = model.fit(X, lambdas=0.1)
```

The `fit` method assumes the rows of the data matrix `X` correspond to the samples. 

`JointPartialRegression` and `SparseHuber`supports quadratic and Huber loss, which can be specified when instantiating the object with either `loss=l2` (default) or `loss='huber'`. When using Huber loss, the threshold parameter is specified by the `rho` parameter when calling `fit`. The `fit` method also supports regularization parameter selection via $K$-fold cross-validation, Aikake information criterion (AIC) or Bayesian information criterion (BIC) (see examples below). 

```python
precision = model.fit(X, n_lambdas=50, cv_folds=10)
precision = model.fit(X, n_lambdas=50, criterion='AIC')
precision = model.fit(X, n_lambdas=50, criterion='BIC')
```

If `n_lambdas` is specified, then the model selects the parameter range automatically, but it is possible to pre-specify the parameter grid by specifying `lambdas` when calling `fit` with either CV or AIC/BIC.

## Technical details

For fitting, the joint partial regression model uses the PD3O algorithm ([Yin, 2018](https://link.springer.com/article/10.1007/s10915-018-0680-3)), while the Huber lasso regression model uses an implementation FISTA ([Beck and Teboulle, 2009](https://epubs.siam.org/doi/10.1137/080716542)). The Rust crate [ndarray-linalg](https://docs.rs/ndarray-linalg/latest/ndarray_linalg/), which leverages [LAPACK](https://www.netlib.org/lapack/) routines, is used for all linear algebra operations. 

## Citing `graph-jpr`

If you use `graph-jpr`, please cite the following paper:

```
@misc{graph-jpr,
 title={Inverse Covariance and Partial Correlation Matrix Estimation via Joint Partial Regression}, 
 author={Samuel Erickson and Tobias Rydén},
 year={2025},
 eprint={2502.08414},
 archivePrefix={arXiv},
 primaryClass={stat.ML},
 url={https://arxiv.org/abs/2502.08414}, 
}
```
