import numpy as np
from scipy import linalg, stats, optimize

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
        self.n_factors = n_factors
        self.rotation = rotation
        self.loadings_ = None
        self.uniquenesses_ = None
        self.n_iter_ = None
        self.loglike_ = None
        self.chi_square_ = None
        self.dof_ = None
        self.p_value_ = None

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
        n_samples, n_features = X.shape

        if n_features < self.n_factors:
            raise ValueError("n_features must be at least n_factors")

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

        self.n_iter_ = res.nit
        self.loglike_ = -res.fun
        self.dof_ = ((n_features - self.n_factors)**2 - n_features - self.n_factors) / 2
        self.chi_square_ = (n_samples - 1 - (2 * n_features + 5) / 6 - (2 * self.n_factors) / 3) * res.fun
        self.p_value_ = 1 - stats.chi2.cdf(self.chi_square_, self.dof_)

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
    
    def score(self, X):
        """
        Compute factor scores using the fitted model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to compute scores for.

        Returns
        -------
        scores : ndarray of shape (n_samples, n_factors)
            Factor scores.

        """
        if self.loadings_ is None:
            raise ValueError("FactorAnalysis model is not fitted yet.")

        X = np.asarray(X)
        if X.shape[1] != self.loadings_.shape[0]:
            raise ValueError("X has %d features, but FactorAnalysis is expecting %d features" %
                             (X.shape[1], self.loadings_.shape[0]))

        corr = np.corrcoef(X, rowvar=False)
        inv_corr = linalg.inv(corr)
        return X @ inv_corr @ self.loadings_
    
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

        return loadings @ rotation_matrix