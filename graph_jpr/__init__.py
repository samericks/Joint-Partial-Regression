__version__ = '0.1.0'
__author__ = 'Samuel Erickson Andersson'

from .graph_jpr import fit_sparse_huber, fit_jpr

__all__ = [
    'JointPartialRegression', 
    'SparseHuber', 
    'fit_sparse_huber', 
    'fit_jpr'
]

import numpy as np
from typing import Union, List, Iterable

class JointPartialRegression:
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
        
    Example
    --------
    >>> import numpy as np
    >>> from graph_jpr import JointPartialRegression
    >>> X = np.random.normal(size=(100, 10))
    >>> model = JointPartialRegression()
    >>> precision = model.fit(X, lmbdas=0.1)
    """
    def __init__(self, mode: str='l2', fit_intercept: bool=True, max_iter: int=100, tol: float=1e-6):
        if mode not in ['l2', 'huber']:
            raise ValueError('mode must be either "l2" or "huber"')
        self.mode = mode
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol
        self.precision_ = None

    def fit(self, X: np.array, lmbdas: Union[float, List[float]], rho: float=0.0):
        """
        Fit the model to the data.
        
        Parameters
        ----------
        X : np.array
            The data matrix.
        lmbdas : Union[float, List[float]]
            The regularization parameter(s). If a float, the same value is used for all features.
        rho : float, default=0.0
            The Huber loss threshold. Only used if mode='huber'.
        cv_folds : int, default=0
            Number of cross-validation folds. If greater than 1, the best lambda is selected using  cross-validation.
            
        Returns
        -------
        precision_ : np.array
            The estimated precision matrix.
            
        Example
        --------
        >>> import numpy as np
        >>> from graph_jpr import JointPartialRegression
        >>> X = np.random.normal(size=(100, 10))
        >>> model = JointPartialRegression()
        >>> precision = model.fit(X, lmbdas=0.1)
        """
        if isinstance(lmbdas, float):
            lmbdas = [lmbdas] * X.shape[1]
            
        if self.mode == 'huber' and rho <= 0.0:
            raise ValueError('rho > 0.0 must be specified for Huber loss')
        
        if np.any(np.array(lmbdas) < 0.0) or len(lmbdas) != X.shape[1]:
            raise ValueError('lmbdas must be scalar or have the same length as the number of features and be non-negative')
        
        self.precision_, self.intercepts_ = fit_jpr(X, lmbdas, rho, self.fit_intercept, 0, self.max_iter, self.tol)

        return self.precision_
    
    def fit_cv(self, X: np.array, lmbdas: Iterable[float], rho: float=0.0, cv_folds: int=5):
        """
        Fit the model to the data using cross-validation for regularization
        parameter selection.
        
        Parameters
        ----------
        X : np.array
            The data matrix.
        lmbdas : Iterable[float]
            The regularization parameters to test.
        rho : float, default=0.0
            The Huber loss threshold. Only used if mode='huber'.
        cv_folds : int, default=5
            Number of cross-validation folds.
            
        Returns
        -------
        precision_ : np.array
            The estimated precision matrix.
            
        Example
        --------
        >>> import numpy as np
        >>> from graph_jpr import JointPartialRegression
        >>> X = np.random.normal(size=(100, 10))
        >>> model = JointPartialRegression()
        >>> precision = model.fit_cv(X, lmbdas=[0.1, 0.2, 0.3], cv_folds=5)
        """
        lmbdas = list(lmbdas)

        if np.any(np.array(lmbdas) < 0.0) or len(lmbdas) < 1:
            raise ValueError('lmbdas must be non-negative')
        
        if len(lmbdas) < 1:
            raise ValueError('lmbdas must be non-empty')
        
        if cv_folds < 2:
            raise ValueError('cv_folds must be greater than 1')

        self.precision_, self.intercepts_ = fit_jpr(X, lmbdas, rho, self.fit_intercept, cv_folds, self.max_iter, self.tol)

        return self.precision_
    
class SparseHuber:
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
        
    Example
    --------
    >>> import numpy as np
    >>> from graph_jpr import SparseHuber
    >>> X = np.random.normal(size=(100, 10))
    >>> y = X @ np.random.normal(size=10) + np.random.normal(size=100)
    >>> model = SparseHuber()
    >>> coef, intercepts = model.fit(X, y, lmbda=0.1, rho=1.0)    
    """
    def __init__(self, fit_intercept: bool=True, max_iter: int=100, tol: float=1e-6):
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol
        self.coef_ = None
                
    def fit(self, X: np.array, y: np.array, lmbda: float, rho: float, theta_start=None):
        if theta_start is None:
            theta_start = np.zeros(X.shape[1])

        alpha = 1 / np.linalg.norm(X, ord=2)**2
        self.coef_, self.intercept_ = fit_sparse_huber(X, y, [lmbda], rho, self.fit_intercept, 0, alpha, self.max_iter, self.tol, theta_start=theta_start)
        return self.coef_, self.intercept_
    
    def fit_cv(self, X: np.array, y: np.array, lmbdas: Iterable[float], rho: float, cv_folds: int=5, theta_start=None):
        lmbdas = list(lmbdas)

        if np.any(np.array(lmbdas) < 0.0) or len(lmbdas) < 1:
            raise ValueError('lmbdas must be non-negative')
        
        if len(lmbdas) < 1:
            raise ValueError('lmbdas must be non-empty')
        
        if theta_start is None:
            theta_start = np.zeros(X.shape[1])
        
        if cv_folds < 2:
            raise ValueError('cv_folds must be greater than 1')

        alpha = 1 / np.linalg.norm(X, ord=2)**2
        self.coef_, self.intercept_ = fit_sparse_huber(X, y, lmbdas, rho, self.fit_intercept, cv_folds, alpha, self.max_iter, self.tol, theta_start=theta_start)
        return self.coef_, self.intercept_

    def predict(self, X: np.array):
        return X @ self.coef_ + self.intercept_