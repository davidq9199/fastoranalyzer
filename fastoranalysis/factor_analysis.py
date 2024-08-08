import numpy as np
from scipy import linalg, optimize

class FactorAnalysis:
    """
    Factor Analysis using Maximum Likelihood Estimation.

    Parameters
    ----------
    n_factors : int
        Number of factors to extract.
    rotation : {'varimax', None}, default=None
        Method for rotation of factors.

    Attributes
    ----------
    loadings_ : ndarray of shape (n_features, n_factors)
        Factor loadings matrix.
    uniquenesses_ : ndarray of shape (n_features,)
        Uniquenesses of each feature.

    """

    def __init__(self, n_factors, rotation=None):
        if not isinstance(n_factors, (int, np.integer)) or n_factors <= 0:
            raise ValueError("n_factors must be a positive integer")
        if rotation not in [None, 'varimax']:
            raise ValueError("rotation must be either None or 'varimax'")
        
        self.n_factors = n_factors
        self.rotation = rotation
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
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError("Expected 2D array, got %dD array instead" % X.ndim)
        n_samples, n_features = X.shape
        if n_features < self.n_factors:
            raise ValueError("n_features=%d must be >= n_factors=%d" % 
                             (n_features, self.n_factors))


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

        if self.rotation == 'varimax':
            self.loadings_ = self._varimax_rotation(self.loadings_)
        
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
            raise ValueError("FactorAnalysis model is not fitted yet.")
        
        X = np.asarray(X)
        if X.shape[1] != self.loadings_.shape[0]:
            raise ValueError("X has %d features, but FactorAnalysis is expecting %d features" %
                             (X.shape[1], self.loadings_.shape[0]))
        
        return X @ self.loadings_
    
    def _varimax_rotation(self, loadings, max_iter=1000, tol=1e-5):
        """Perform varimax rotation on loadings."""
        n_factors = loadings.shape[1]
        rotation_matrix = np.eye(n_factors)
        var = 0

        for _ in range(max_iter):
            old_var = var
            comp = np.dot(loadings, rotation_matrix)
            u, s, v = linalg.svd(np.dot(loadings.T, comp**3 - (1/3) * np.dot(comp, np.diag(np.sum(comp**2, axis=0)))))
            rotation_matrix = np.dot(u, v)
            var = np.sum(s)
            if var - old_var < tol:
                break

        return np.dot(loadings, rotation_matrix)