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
        if rotation not in ['varimax', 'promax', None]:
            raise ValueError("rotation must be 'varimax', 'promax', or None")
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
        self.rotation_matrix_ = None
        self.n_obs_ = None
        self.correlation_ = None
        self.call_ = f"FactorAnalysis(n_factors={n_factors}, rotation='{rotation}', scores='{scores}')"
        self.rotmat_ = None  

    def fit(self, X=None, covmat=None, n_obs=None):
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
        if X is None and covmat is None:
            raise ValueError("Either X or covmat must be provided")
        
        if X is not None:
            X = np.asarray(X)
            n_samples, n_features = X.shape
            self.n_obs_ = n_samples
            self.correlation_ = np.corrcoef(X, rowvar=False)
        elif covmat is not None:
            if n_obs is None:
                raise ValueError("n_obs must be provided when using a covariance matrix")
            covmat = np.asarray(covmat)
            n_features = covmat.shape[0]
            self.n_obs_ = n_obs
            self.correlation_ = self._cov2cor(covmat)
        
        if n_features < self.n_factors:
            raise ValueError("n_features must be at least n_factors")

        def objective(uniquenesses):
            diag_unique = np.diag(uniquenesses)
            _, s, Vt = linalg.svd(self.correlation_ - diag_unique)
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
        _, s, Vt = linalg.svd(self.correlation_ - diag_unique)
        
        self.unrotated_loadings_ = Vt[:self.n_factors, :].T * np.sqrt(s[:self.n_factors])
        self.loadings_ = self.unrotated_loadings_.copy()

        if self.rotation == 'varimax':
            self.loadings_, self.rotmat_ = self._varimax_rotation(self.loadings_)
            self.scaling_factors_ = np.ones(self.loadings_.shape[0])
        elif self.rotation == 'promax':
            self.loadings_, self.rotmat_, self.scaling_factors_ = self._promax_rotation(self.loadings_)
        else:
            self.rotmat_ = np.eye(self.n_factors)
            self.scaling_factors_ = np.ones(self.loadings_.shape[0])

        self.n_iter_ = res.nit
        self.converged_ = res.success
        self.criteria_ = {'objective': res.fun}
        self.loglike_ = -res.fun

        self.dof_ = ((n_features - self.n_factors)**2 - n_features - self.n_factors) // 2

        if self.dof_ > 0:
            self.STATISTIC = (self.n_obs_ - 1 - (2 * n_features + 5) / 6 - (2 * self.n_factors) / 3) * res.fun
            self.PVAL = stats.chi2.sf(self.STATISTIC, self.dof_)

        if self.scores_method != 'none' and X is not None:
            self.scores_ = self.transform(X)

        return self
    
    def _cov2cor(self, cov):
        """Convert covariance matrix to correlation matrix."""
        std = np.sqrt(np.diag(cov))
        cor = cov / np.outer(std, std)
        return np.clip(cor, -1, 1)
        
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
    
    def _varimax_rotation(self, loadings, normalize=True, max_iter=1000, tol=1e-5):
        """Perform varimax rotation on loadings."""
        n_factors = loadings.shape[1]
        rotation_matrix = np.eye(n_factors)
        var = 0

        if normalize:
            communalities = np.sum(loadings**2, axis=1)
            loadings = loadings / np.sqrt(communalities[:, None])

        for _ in range(max_iter):
            old_var = var
            comp = loadings @ rotation_matrix
            u, s, v = linalg.svd(loadings.T @ (comp**3 - (1/3) * comp @ np.diag(np.sum(comp**2, axis=0))))
            rotation_matrix = u @ v
            var = np.sum(s)
            if var - old_var < tol:
                break

        rotated_loadings = loadings @ rotation_matrix
        if normalize:
            rotated_loadings = rotated_loadings * np.sqrt(communalities[:, None])

        return rotated_loadings, rotation_matrix
    
    def _promax_rotation(self, loadings, power=4):
        """Perform promax rotation on loadings."""
        X, rotation = self._varimax_rotation(loadings)
        
        h2 = np.sum(X**2, axis=1)
        X_normalized = X / np.sqrt(h2[:, None])
        
        P = X_normalized ** power
        U = linalg.inv(X_normalized.T @ X_normalized) @ (X_normalized.T @ P)
        d = np.diag(np.sqrt(np.sum(U**2, axis=0)))
        U = U @ linalg.inv(d)
        
        rotated_loadings = loadings @ rotation @ U
        
        h2_rotated = np.sum(rotated_loadings**2, axis=1)
        scaling_factors = np.sqrt(np.sum(loadings**2, axis=1) / h2_rotated)
        rotated_loadings *= scaling_factors[:, None]
        
        promax_rotation = rotation @ U
        
        return rotated_loadings, promax_rotation, scaling_factors

    def get_factor_variance(self):
        """
        Compute variance explained by each factor.

        Returns
        -------
        variance : ndarray of shape (3, n_factors)
            Array with variance, proportion of variance and cumulative proportion of variance explained.
        
        """
        if self.loadings_ is None:
            raise ValueError("FactorAnalysis model is not fitted yet.")

        loadings = self.loadings_

        variance = np.sum(loadings**2, axis=0)
        total_variance = np.sum(variance)
        proportion = variance / total_variance
        cumulative = np.cumsum(proportion)

        return np.vstack((variance, proportion, cumulative))
    
    def _format_matrix(self, matrix, digits=3):
        """Format a matrix for printing."""
        return np.array2string(matrix, precision=digits, suppress_small=True)  
          
    def _format_loadings(self, cutoff=0.1, digits=2):
        """Format loadings matrix for printing."""
        loadings = self.loadings_.copy()
        loadings[np.abs(loadings) < cutoff] = 0.0
        formatted = np.array2string(loadings, precision=digits, suppress_small=True, floatmode='fixed')
        formatted = formatted.replace("0.00", " 0.0")
        return formatted

    def __str__(self):
        """Return a string representation of the FactorAnalysis results."""
        if self.loadings_ is None:
            return "FactorAnalysis model is not fitted yet."
        
        output = []
        output.append(f"Call:\n{self.call_}\n")
        
        output.append("Uniquenesses:")
        uniquenesses_str = " ".join([f"{u:.3f}" for u in self.uniquenesses_])
        output.append(uniquenesses_str + "\n")
        
        output.append("Loadings:")
        loadings_str = self._format_loadings()
        output.append(loadings_str + "\n")
        
        ss_loadings = np.sum(self.loadings_**2, axis=0)
        output.append("               SS loadings    Proportion Var")
        for i, ss in enumerate(ss_loadings):
            prop_var = ss / self.loadings_.shape[0]
            output.append(f"Factor {i+1:<8} {ss:.3f}           {prop_var:.3f}")
        
        if self.rotation is not None and self.rotmat_ is not None:
            output.append("\nFactor Correlations:")
            corr = self.rotmat_ @ self.rotmat_.T
            corr_str = self._format_matrix(corr)
            output.append(corr_str)
        
        output.append(f"\nThe degrees of freedom for the model is {self.dof_}")
        if hasattr(self, 'STATISTIC') and hasattr(self, 'PVAL'):
            output.append(f"Test of the hypothesis that {self.n_factors} factors are sufficient.")
            output.append(f"The chi square statistic is {self.STATISTIC:.2f} on {self.dof_} degrees of freedom.")
            output.append(f"The p-value is {self.PVAL:.3g}")
        else:
            output.append(f"The fit was {self.criteria_['objective']:.4f}")
        
        return "\n".join(output)