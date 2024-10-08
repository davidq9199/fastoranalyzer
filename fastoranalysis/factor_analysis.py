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
    rotation : {'varimax', 'promax', None}, default='varimax'
    use_smc : bool, default=True
        Whether to use Squared Multiple Correlations for initial uniqueness values.
        Method for rotation of factors. 
        - 'varimax': Perform varimax rotation.
        - 'promax': Perform promax rotation.
        - None: No rotation is performed.
    scores : {'regression', 'bartlett'}, default='regression'
        Method for computing factor scores.
        - 'regression': Compute regression scores.
        - 'bartlett': Compute Bartlett scores.
    na_action : {'omit', 'fail'}, default='omit'
        Specifies the action to take if missing values are found.
        - 'omit': Remove samples with missing values.
        - 'fail': Raise an error if missing values are encountered.
    control : dict, optional
        A dictionary of control parameters for the optimization algorithm.

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
    rotation_matrix_ : ndarray of shape (n_factors, n_factors)
        Rotation matrix used to rotate the factors.
    n_obs_ : int
        Number of observations used in the analysis.
    correlation_ : ndarray of shape (n_features, n_features)
        Correlation matrix of the input data.
    converged_ : bool
        Whether the optimization algorithm converged.
    criteria_ : dict
        Dictionary containing optimization criteria.

    Methods
    -------
    fit(X=None, covmat=None, n_obs=None, subset=None, na_action='omit', start=None)
        Fit the factor analysis model.
    transform(X)
        Apply dimensionality reduction to X using the fitted model.
    score(X)
        Compute factor scores using the fitted model.
    get_factor_variance()
        Compute variance explained by each factor.

    Examples
    --------
    >>> import numpy as np
    >>> from fastoranalysis import FactorAnalysis
    >>> X = np.random.rand(100, 5)
    >>> fa = FactorAnalysis(n_factors=2)
    >>> fa.fit(X)
    >>> transformed_X = fa.transform(X)
    >>> scores = fa.score(X)
    >>> variance = fa.get_factor_variance()

    Notes
    -----
    The factor analysis model is:
    x = Λf + e
    where x is a p-element vector, Λ is a p × k matrix of loadings,
    f is a k-element vector of scores, and e is a p-element vector of errors.
    """

    def __init__(self, n_factors, rotation='varimax', scores='regression', na_action='omit', control=None, use_smc=True):
        if not isinstance(n_factors, int) or n_factors <= 0:
            raise ValueError("n_factors must be a positive integer")
        if rotation not in ['varimax', 'promax', None]:
            raise ValueError("rotation must be 'varimax', 'promax', or None")
        if scores not in ['regression', 'bartlett']:
            raise ValueError("scores must be 'regression' or 'bartlett'")
        if na_action not in ['omit', 'fail']:
            raise ValueError("na_action must be 'omit' or 'fail'")
        
        self.n_factors = n_factors
        self.rotation = rotation
        self.scores_method = scores
        self.use_smc = use_smc
        self.na_action = na_action
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

    def fit(self, X=None, covmat=None, n_obs=None, subset=None, na_action='omit', start=None):
        """
        Fit the factor analysis model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features), optional
            The input samples. If both X and covmat are provided, X takes precedence.
        covmat : array-like of shape (n_features, n_features), optional
            The covariance matrix. If provided without X, n_obs must also be specified.
        n_obs : int, optional
            The number of observations. Required if covmat is provided without X.
        subset : array-like, optional
            Indices of samples to use. If provided, only these samples are used.
        na_action : {'omit', 'fail'}, default='omit'
            Specifies how to handle missing values:
            - 'omit': Remove samples with any missing values.
            - 'fail': Raise an error if there are any missing values.
        start : array-like of shape (n_features,) or (n_features, n_starts), optional
            Starting values for uniquenesses. If 2D, each column is tried as a starting value.

        Returns
        -------
        self : object
            Fitted estimator.

        Raises
        ------
        ValueError
            If neither X nor covmat is provided, or if covmat is provided without n_obs.
            If the number of features is less than the number of factors.
            If unable to optimize from the given starting value(s).
            
        """
        if X is None and covmat is None:
            raise ValueError("Either X or covmat must be provided")
        
        if X is not None:
            X = np.asarray(X)
            if subset is not None:
                X = X[subset]
            
            if np.isnan(X).any():
                if na_action == 'omit':
                    mask = ~np.isnan(X).any(axis=1)
                    X = X[mask]
                elif na_action == 'fail':
                    raise ValueError("Input contains NaN values")
                else:
                    raise ValueError("Invalid na_action. Choose 'omit' or 'fail'")
            
            n_samples, n_features = X.shape
            if n_samples == 0:
                raise ValueError("0 samples left after handling missing values")
            
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

        if self.use_smc:
            start = self._smc(self.correlation_)

        if start is None:
            nstart = self.control.get('nstart', 1)
            start = np.random.uniform(0.1, 0.9, (n_features, nstart))
        if start is not None:
            start = np.asarray(start)
            if start.ndim == 1:
                start = start.reshape(-1, 1)
            if start.shape[0] != n_features:
                raise ValueError(f"'start' must have exactly {n_features} elements")
            if start.ndim > 2:
                raise ValueError(f"'start' must be a 1D or 2D array")
        nstart = start.shape[1]

        best_res = None
        best_objective = np.inf

        for i in range(nstart):
            res = self._fit_single(start[:, i], n_features)
            if res.fun < best_objective:
                best_res = res
                best_objective = res.fun
                if best_objective < self.control.get('tol', 1e-5):
                    break

        if best_res is None:
            raise ValueError("Unable to optimize from the given starting value(s)")

        self.uniquenesses_ = best_res.x
        self.converged_ = best_res.success
        diag_unique = np.diag(self.uniquenesses_)
        corr_minus_diag = self.correlation_ - diag_unique
        s, Vt = linalg.eigh(corr_minus_diag)
        s = s[::-1]
        Vt = Vt[:, ::-1].T
        
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

        self.n_iter_ = best_res.nit
        self.converged_ = best_res.success
        self.criteria_ = {'objective': best_res.fun}
        self.loglike_ = -best_res.fun

        self.dof_ = ((n_features - self.n_factors)**2 - n_features - self.n_factors) // 2

        if self.dof_ > 0:
            self.STATISTIC = (self.n_obs_ - 1 - (2 * n_features + 5) / 6 - (2 * self.n_factors) / 3) * best_res.fun
            self.PVAL = stats.chi2.sf(self.STATISTIC, self.dof_)

        if self.scores_method != 'none' and X is not None:
            self.scores_ = self.transform(X)

        return self
    
    def _smc(self, corr_matrix):
        """Compute squared multiple correlations."""
        inv_corr = linalg.inv(corr_matrix)
        return 1 - 1 / np.diag(inv_corr)
    
    def _fit_single(self, start, n_features):
        """Fit the model with a single starting value."""
        def objective(uniquenesses):
            diag_unique = np.diag(uniquenesses)
            corr_minus_diag = self.correlation_ - diag_unique
            s = linalg.eigvalsh(corr_minus_diag)[::-1]
            return -np.sum(np.log(s[self.n_factors:])) + np.sum(np.log(uniquenesses))

        default_options = {'maxiter': 1000, 'ftol': 1e-6}
        options = {**default_options, **self.control}
        options.pop('nstart', None)
        
        return optimize.minimize(objective, start, method='L-BFGS-B', 
                                bounds=[(0.005, 1)] * n_features, 
                                options=options)

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
            Transformed data (factor scores).

        Raises
        ------
        ValueError
            If the model has not been fitted yet.
            If X has a different number of features than the fitted model.
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
        X_centered = X - X.mean(axis=0)
        inv_corr = linalg.inv(self.correlation_)
        return X_centered @ inv_corr @ self.loadings_

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
            - Row 0: Variance explained by each factor
            - Row 1: Proportion of total variance explained by each factor
            - Row 2: Cumulative proportion of variance explained

        Raises
        ------
        ValueError
            If the FactorAnalysis model is not fitted yet.

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