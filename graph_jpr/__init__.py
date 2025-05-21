__version__ = "0.1.0"
__author__ = "Samuel Erickson Andersson"

from .graph_jpr import fit_sparse_huber, fit_jpr

__all__ = ["JointPartialRegression", "SparseHuber", "fit_sparse_huber", "fit_jpr"]

import numpy as np
from typing import Union, Iterable


class Estimator:
    def __init__(self):
        pass

    def _validate_params(self, X, lambdas, rho, n_lambdas, cv_folds, criterion):
        if self.mode == "huber" and rho <= 0.0:
            raise ValueError("rho must be greater than 0.0 when using Huber loss")

        if np.any(np.array(lambdas) < 0.0):
            raise ValueError("lambdas must be non-negative")

        if len(lambdas) > 0 and n_lambdas > 0:
            raise ValueError("Cannot pre-specify lambdas and use n_lambdas")

        use_model_selection = cv_folds > 0 or criterion == "AIC" or criterion == "BIC"

        if use_model_selection:
            if cv_folds > 0 and criterion != "":
                raise ValueError(
                    "Cannot use cross-validation and AIC/BIC at the same time"
                )
            elif len(lambdas) > 0 and n_lambdas > 0:
                raise ValueError("Cannot pre-specify lambdas and specify n_lambdas")
            elif len(lambdas) == 0 and n_lambdas == 0:
                raise ValueError(
                    "Must specify lambdas or n_lambdas when using CV or AIC/BIC"
                )

            if cv_folds < 2 and criterion == "":
                raise ValueError("Number of folds must be greater than 1")
            elif cv_folds > X.shape[0]:
                raise ValueError("Number of folds cannot exceed the number of samples")
        else:
            if len(lambdas) != X.shape[1]:
                raise ValueError(
                    "lambdas must be scalar or the same length as the number of features when not using CV or AIC/BIC"
                )

            if n_lambdas > 0:
                raise ValueError(
                    "Cannot specify n_lambdas when not using CV or AIC/BIC"
                )


class JointPartialRegression(Estimator):
    """
    Joint Partial Regression (JPR) for estimating a sparse precision matrix.

    Parameters
    ----------
    mode : str, default='l2'
        The mode of the loss function. Either 'l2' or 'huber'.
    fit_intercept : bool, default=True
        Whether to fit an intercept.
    max_iter : int, default=100
        Maximum number of iterations.
    tol : float, default=1e-6
        Tolerance for convergence.

    Attributes
    ----------
    precision_ : np.array
        The estimated precision matrix.
    intercepts_ : np.array
        The estimated intercepts.
    partial_corr_ : np.array
        The estimated partial correlation matrix.
    selected_lambdas_ : np.array
        The selected regularization parameters.
    lambdas_grid_ : np.array
        The grid of regularization parameters used for model selection.

    Example
    --------
    >>> import numpy as np
    >>> from graph_jpr import JointPartialRegression
    >>> X = np.random.normal(size=(100, 10))
    >>> model = JointPartialRegression()
    >>> precision = model.fit(X, lambdas=0.1)
    """

    def __init__(
        self,
        mode: str = "l2",
        fit_intercept: bool = True,
        max_iter: int = 100,
        tol: float = 1e-6,
    ):
        if mode not in ["l2", "huber"]:
            raise ValueError('mode must be either "l2" or "huber"')
        self.mode = mode
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol
        self.precision_ = None
        self.partial_corr_ = None
        self.selected_lambdas_ = None
        self.lambdas_grid_ = None

    def fit(
        self,
        X: np.array,
        lambdas: Union[float, Iterable[float]] = [],
        rho: float = 0.0,
        n_lambdas: int = 0,
        cv_folds: int = 0,
        criterion: str = "",
    ):
        """
        Fit the model to the data.

        Parameters
        ----------
        X : np.array
            The data matrix.
        lambdas : Union[float, List[float]]
            The regularization parameter(s). If a float, the same value is used
            for all features.
        rho : float, default=0.0
            The Huber loss threshold. Only used if mode='huber'.
        n_lambdas : int, default=0
            The number of regularization parameters to test when using CV or
            AIC/BIC, when lambdas is not pre-specified.
        cv_folds : int, default=0
            If greater than 1, cross-validation is used for regularization
            parameter selection. The value specifies the number of folds.
        criterion : str, default=''
            The criterion for model selection. Available options are Aikaike
            information criterion ('AIC'), and Bayesian information criterion
            ('BIC').

        Returns
        -------
        precision_ : np.array
            The estimated precision matrix.
        """
        if isinstance(lambdas, float):
            lambdas = [lambdas] * X.shape[1]
        else:
            lambdas = list(lambdas)

        self._validate_params(X, lambdas, rho, n_lambdas, cv_folds, criterion)

        (
            self.precision_,
            self.intercepts_,
            self.selected_lambdas_,
            self.lambdas_grid_,
        ) = fit_jpr(
            X,
            lambdas,
            rho,
            self.fit_intercept,
            n_lambdas,
            cv_folds,
            criterion,
            self.max_iter,
            self.tol,
        )
        T = 1 / np.diag(self.precision_) ** 0.5
        self.partial_corr_ = -np.diag(T) @ self.precision_ @ np.diag(T)

        return self.precision_


class SparseHuber(Estimator):
    """
    l1-regularized Huber regression.

    Parameters
    ----------
    fit_intercept : bool, default=True
        Whether to fit an intercept.
    max_iter : int, default=100
        Maximum number of iterations.
    tol : float, default=1e-6
        Tolerance for convergence.

    Attributes
    ----------
    coef_ : np.array
        The estimated coefficients.
    intercept_ : float
        The estimated intercept.
    selected_lambda_ : float
        The selected regularization parameter.

    Example
    --------
    >>> import numpy as np
    >>> from graph_jpr import SparseHuber
    >>> X = np.random.normal(size=(100, 10))
    >>> y = X @ np.random.normal(size=10) + np.random.normal(size=100)
    >>> model = SparseHuber()
    >>> coef, intercepts = model.fit(X, y, lambdas=0.1, rho=1.0)
    """

    def __init__(
        self, fit_intercept: bool = True, max_iter: int = 100, tol: float = 1e-6
    ):
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol
        self.coef_ = None
        self.intercept_ = None
        self.selected_lambda_ = None

    def fit(
        self,
        X: np.array,
        y: np.array,
        lambdas: Union[float, Iterable[float]],
        rho: float,
        n_lambdas: int = 0,
        cv_folds: int = 0,
        criterion: str = "",
        theta_start=None,
        intercept_start=None,
    ):
        """
        Fit the model to the data.

        Parameters
        ----------
        X : np.array
            The data matrix.
        y : np.array
            The response vector.
        lambdas : float
            The regularization parameter or a list of parameters to test.
        rho : float
            The Huber loss threshold.
        n_lambdas : int, default=0
            The number of regularization parameters to test when using CV or
            AIC/BIC, when lambdas is not pre-specified.
        cv_folds : int, default=0
            If greater than 1, cross-validation is used for regularization
            parameter selection. The value specifies the number of folds.
        criterion : str, default=''
            The criterion for model selection. Available options are Aikaike
            information criterion ('AIC'), and Bayesian information criterion
            ('BIC').
        theta_start : np.array, default=None
            Initial guess for the coefficients.

        Returns
        -------
        coef_ : np.array
            The estimated coefficients.
        """
        if isinstance(lambdas, float):
            lambdas = [lambdas]
        else:
            lambdas = list(lambdas)

        if theta_start is None:
            theta_start = np.zeros(X.shape[1])

        if intercept_start is None:
            intercept_start = 0.0

        self._validate_params(X, lambdas, rho, cv_folds, criterion)

        alpha = 1 / np.linalg.norm(X, ord=2) ** 2
        self.coef_, self.intercept_, self.selected_lambda_, self.lambda_grid = (
            fit_sparse_huber(
                X,
                y,
                lambdas,
                rho,
                self.fit_intercept,
                n_lambdas,
                cv_folds,
                criterion,
                alpha,
                self.max_iter,
                self.tol,
                theta_start=theta_start,
            )
        )
        return self.coef_, self.intercept_

    def predict(self, X: np.array):
        return X @ self.coef_ + self.intercept_
