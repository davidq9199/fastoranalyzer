import numpy as np
from scipy import linalg, stats, optimize

class FactorAnalysis:
    """
    Factor Analysis using Maximum Likelihood Estimation.

    This class implements factor analysis, a statistical method used to describe
    variability among observed, correlated variables in terms of a potentially
    lower number of unobserved variables called factors.

    Parameters
    ----------
    n_factors : int
        Number of factors to extract.
    rotation : {'varimax', None}, default=None
        Method for rotation of factors. If None, no rotation is performed.

    Attributes
    ----------
    loadings_ : ndarray of shape (n_features, n_factors)
        Factor loadings matrix.
    uniquenesses_ : ndarray of shape (n_features,)
        Uniquenesses of each feature.
    n_iter_ : int
        Number of iterations in the optimization.
    loglike_ : float
        Log-likelihood of the fitted model.
    chi_square_ : float
        Chi-square statistic for the goodness of fit.
    dof_ : int
        Degrees of freedom for the chi-square test.
    p_value_ : float
        P-value for the chi-square test.

    Methods
    -------
    fit(X)
        Fit the factor analysis model.
    transform(X)
        Apply dimensionality reduction to X using the fitted model.
    score(X)
        Compute factor scores using the fitted model.

    Examples
    --------
    >>> import numpy as np
    >>> from fastoranalysis import FactorAnalysis
    >>> X = np.random.rand(100, 5)
    >>> fa = FactorAnalysis(n_factors=2)
    >>> fa.fit(X)
    >>> transformed_X = fa.transform(X)
    >>> scores = fa.score(X)

    Notes
    -----
    The factor analysis model is:
    x = Λf + e
    where x is a p-element vector, Λ is a p × k matrix of loadings,
    f is a k-element vector of scores, and e is a p-element vector of errors.

    """

    def __init__(self, n_factors, rotation=None, scores='regression', control=None):
        if not isinstance(n_factors, int) or n_factors <= 0:
            raise ValueError("n_factors must be a positive integer")
        if rotation not in ['varimax', None]:
            raise ValueError("rotation must be 'varimax' or None")
        if scores not in ['regression', 'bartlett']:
            raise ValueError("scores must be 'regression' or 'bartlett'")
        
        self.n_factors = n_factors
        self.rotation = rotation
        self.scores_method = scores
        self.control = control or {}
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
        
        default_options = {'maxiter': 1000}
        options = {**default_options, **self.control}
        
        res = optimize.minimize(objective, initial_uniquenesses, method='L-BFGS-B', 
                                bounds=[(0.005, 1)] * n_features, 
                                options=options)

        self.uniquenesses_ = res.x
        diag_unique = np.diag(self.uniquenesses_)
        _, s, Vt = linalg.svd(corr - diag_unique)
        self.loadings_ = Vt[:self.n_factors, :].T * np.sqrt(s[:self.n_factors])

        if self.rotation == 'varimax':
            self.loadings_ = self._varimax_rotation(self.loadings_)

        self.n_iter_ = res.nit
        self.loglike_ = -res.fun
        self.dof_ = int(((n_features - self.n_factors)**2 - n_features - self.n_factors) / 2)
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
            raise ValueError(f"X has {X.shape[1]} features, but FactorAnalysis is expecting {self.loadings_.shape[0]} features")
        
        return self.score(X)
    
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
            raise ValueError(f"X has {X.shape[1]} features, but FactorAnalysis is expecting {self.loadings_.shape[0]} features")

        if self.scores_method == 'regression':
            return self._regression_scores(X)
        elif self.scores_method == 'bartlett':
            return self._bartlett_scores(X)
    
    def _regression_scores(self, X):
        """Compute regression scores."""
        corr = np.corrcoef(X, rowvar=False)
        inv_corr = linalg.inv(corr)
        return X @ inv_corr @ self.loadings_

    def _bartlett_scores(self, X):
        """Compute Bartlett's scores."""
        X_centered = X - X.mean(axis=0)
        U = np.diag(1 / self.uniquenesses_)
        M = self.loadings_.T @ U @ self.loadings_
        M_inv = linalg.inv(M)
        return X_centered @ U @ self.loadings_ @ M_inv
    
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