import numpy as np
from scipy import linalg, optimize

class FactorAnalysis:
    """
    Factor Analysis using Maximum Likelihood Estimation.

    Parameters
    ----------
    n_factors : int
        Number of factors to extract.

    Attributes
    ----------
    loadings_ : ndarray of shape (n_features, n_factors)
        Factor loadings matrix.
    uniquenesses_ : ndarray of shape (n_features,)
        Uniquenesses of each feature.

    """

    def __init__(self, n_factors):
        self.n_factors = n_factors
        self.loadings_ = None
        self.uniquenesses_ = None

    def fit(self, X):
        """
        Fit the factor analysis model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        Returns
        -------
        self : object
            Fitted estimator.
        
        """
        n_samples, n_features = X.shape
        corr = np.corrcoef(X, rowvar=False)
        
        def objective(uniquenesses):
            diag_unique = np.diag(uniquenesses)
            _, s, Vt = linalg.svd(corr - diag_unique)
            loadings = Vt[:self.n_factors, :].T * np.sqrt(s[:self.n_factors])
            return -np.sum(np.log(s[self.n_factors:])) + np.sum(np.log(uniquenesses))

        initial_uniquenesses = np.ones(n_features)
        res = optimize.minimize(objective, initial_uniquenesses, method='L-BFGS-B', bounds=[(0.005, 1)] * n_features)
        
        self.uniquenesses_ = res.x
        diag_unique = np.diag(self.uniquenesses_)
        _, s, Vt = linalg.svd(corr - diag_unique)
        self.loadings_ = Vt[:self.n_factors, :].T * np.sqrt(s[:self.n_factors])
        
        return self

    def transform(self, X):
        """
        Apply dimensionality reduction to X using the fitted model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to transform.

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_factors)
            Transformed data.
            
        """
        if self.loadings_ is None:
            raise ValueError("FactorAnalysis must be fitted before transform")
        return X @ self.loadings_