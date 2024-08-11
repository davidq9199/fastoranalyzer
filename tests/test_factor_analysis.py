import pytest
import numpy as np
from fastoranalysis import FactorAnalysis

@pytest.fixture
def sample_data():
    np.random.seed(58)
    return np.random.rand(100, 5)

@pytest.fixture
def fitted_fa(sample_data):
    fa = FactorAnalysis(n_factors=2, rotation='varimax')
    fa.fit(sample_data)
    return fa

@pytest.fixture
def sample_covmat(sample_data):
    return np.cov(sample_data, rowvar=False)

def test_factor_analysis_fit_attributes(sample_data):
    fa = FactorAnalysis(n_factors=2)
    fa.fit(sample_data)
    assert fa.loadings_.shape == (5, 2)
    assert fa.uniquenesses_.shape == (5,)
    assert isinstance(fa.n_iter_, int)
    assert isinstance(fa.loglike_, float)
    assert isinstance(fa.criteria_['objective'], float)
    assert fa.correlation_.shape == (5, 5)
    assert isinstance(fa.n_obs_, int)
    assert isinstance(fa.dof_, int)

def test_factor_analysis_invalid_init():
    with pytest.raises(ValueError, match="n_factors must be a positive integer"):
        FactorAnalysis(n_factors=0)
    with pytest.raises(ValueError, match="n_factors must be a positive integer"):
        FactorAnalysis(n_factors=-1)
    with pytest.raises(ValueError, match="rotation must be 'varimax' or None"):
        FactorAnalysis(n_factors=2, rotation='invalid_rotation')
    with pytest.raises(ValueError, match="scores must be 'regression' or 'bartlett'"):
        FactorAnalysis(n_factors=2, scores='invalid_method')

def test_factor_analysis_score(sample_data):
    fa = FactorAnalysis(n_factors=2)
    fa.fit(sample_data)
    scores = fa.score(sample_data)
    assert scores.shape == (100, 2)

def test_factor_analysis_raw_data_input(sample_data):
    fa = FactorAnalysis(n_factors=2)
    fa.fit(X=sample_data)
    assert fa.loadings_.shape == (5, 2)
    assert fa.uniquenesses_.shape == (5,)
    assert fa.n_obs_ == 100
    assert fa.correlation_.shape == (5, 5)

def test_factor_analysis_covmat_input(sample_covmat):
    fa = FactorAnalysis(n_factors=2)
    fa.fit(covmat=sample_covmat, n_obs=100)
    assert fa.loadings_.shape == (5, 2)
    assert fa.uniquenesses_.shape == (5,)
    assert fa.n_obs_ == 100
    assert fa.correlation_.shape == (5, 5)

def test_factor_analysis_covmat_input_no_n_obs(sample_covmat):
    fa = FactorAnalysis(n_factors=2)
    with pytest.raises(ValueError, match="n_obs must be provided when using a covariance matrix"):
        fa.fit(covmat=sample_covmat)

def test_factor_analysis_no_input():
    fa = FactorAnalysis(n_factors=2)
    with pytest.raises(ValueError, match="Either X or covmat must be provided"):
        fa.fit()

def test_factor_analysis_both_inputs(sample_data, sample_covmat):
    fa = FactorAnalysis(n_factors=2)
    fa.fit(X=sample_data, covmat=sample_covmat, n_obs=100)
    # prioritize X over covmat
    assert fa.n_obs_ == 100
    assert np.allclose(fa.correlation_, np.corrcoef(sample_data, rowvar=False))

def test_factor_analysis_too_many_factors(sample_data):
    fa = FactorAnalysis(n_factors=10) 
    with pytest.raises(ValueError, match="n_features must be at least n_factors"):
        fa.fit(X=sample_data)

def test_factor_analysis_no_scores_with_covmat(sample_covmat):
    fa = FactorAnalysis(n_factors=2, scores='regression')
    fa.fit(covmat=sample_covmat, n_obs=100)
    assert not hasattr(fa, 'scores_')

def test_factor_analysis_scoring_methods(sample_data):
    fa_regression = FactorAnalysis(n_factors=2, scores='regression')
    fa_regression.fit(sample_data)
    scores_regression = fa_regression.score(sample_data)

    fa_bartlett = FactorAnalysis(n_factors=2, scores='bartlett')
    fa_bartlett.fit(sample_data)
    scores_bartlett = fa_bartlett.score(sample_data)

    assert scores_regression.shape == (100, 2)
    assert scores_bartlett.shape == (100, 2)
    assert not np.allclose(scores_regression, scores_bartlett)

def test_factor_analysis_control_parameter(sample_data):
    fa_default = FactorAnalysis(n_factors=2)
    fa_default.fit(sample_data)

    fa_custom = FactorAnalysis(n_factors=2, control={'maxiter': 1})
    fa_custom.fit(sample_data)

    assert fa_custom.n_iter_ <= 1
    assert fa_default.n_iter_ >= fa_custom.n_iter_ 

    fa_more_iter = FactorAnalysis(n_factors=2, control={'maxiter': 100})
    fa_more_iter.fit(sample_data)
    
    assert fa_more_iter.n_iter_ >= fa_custom.n_iter_  

def test_factor_analysis_invalid_init():
    with pytest.raises(ValueError):
        FactorAnalysis(n_factors=0)
    with pytest.raises(ValueError):
        FactorAnalysis(n_factors=2, rotation='invalid_rotation')

def test_factor_analysis_fit(sample_data):
    fa = FactorAnalysis(n_factors=2)
    fa.fit(sample_data)
    assert fa.loadings_.shape == (5, 2)
    assert fa.uniquenesses_.shape == (5,)

def test_factor_analysis_fit_invalid_input():
    fa = FactorAnalysis(n_factors=2)
    with pytest.raises(ValueError):
        fa.fit(np.random.rand(10))
    with pytest.raises(ValueError):
        fa.fit(np.random.rand(10, 1))  # n_features < n_factors

def test_factor_analysis_transform(sample_data):
    fa = FactorAnalysis(n_factors=2)
    fa.fit(X=sample_data)
    transformed = fa.transform(sample_data)
    assert transformed.shape == (100, 2)

def test_factor_analysis_transform_before_fit(sample_data):
    fa = FactorAnalysis(n_factors=2)
    with pytest.raises(ValueError):
        fa.transform(sample_data)

def test_factor_analysis_transform_invalid_input(sample_data):
    fa = FactorAnalysis(n_factors=2)
    fa.fit(sample_data)
    with pytest.raises(ValueError):
        fa.transform(np.random.rand(100, 6))  # incorrect number of features

def test_factor_analysis_with_varimax_rotation(sample_data):
    fa = FactorAnalysis(n_factors=2, rotation='varimax')
    fa.fit(sample_data)
    assert fa.loadings_.shape == (5, 2)
    
    # loadings are different from unrotated solution
    fa_unrotated = FactorAnalysis(n_factors=2, rotation=None)
    fa_unrotated.fit(sample_data)
    assert not np.allclose(fa.loadings_, fa_unrotated.loadings_)

def test_normalized_varimax_rotation():
    np.random.seed(42)
    X = np.random.rand(100, 5)
    
    fa_normalized = FactorAnalysis(n_factors=2, rotation='varimax')
    fa_normalized.fit(X)
    
    fa_unnormalized = FactorAnalysis(n_factors=2, rotation='varimax')
    fa_unnormalized.fit(X)
    fa_unnormalized.loadings_, _ = fa_unnormalized._varimax_rotation(fa_unnormalized.unrotated_loadings_, normalize=False)
    
    assert not np.allclose(fa_normalized.loadings_, fa_unnormalized.loadings_)
    
    original_communalities = np.sum(fa_normalized.unrotated_loadings_**2, axis=1)
    rotated_communalities = np.sum(fa_normalized.loadings_**2, axis=1)

    assert np.allclose(original_communalities, rotated_communalities)
    assert np.allclose(np.sum(fa_normalized.loadings_**2), np.sum(fa_normalized.unrotated_loadings_**2))

def test_factor_analysis_promax_rotation(sample_data):
    fa = FactorAnalysis(n_factors=2, rotation='promax')
    fa.fit(sample_data)
    assert fa.loadings_.shape == (5, 2)
    assert fa.rotmat_.shape == (2, 2)

def test_factor_analysis_rotation_options(sample_data):
    for rotation in [None, 'varimax', 'promax']:
        fa = FactorAnalysis(n_factors=2, rotation=rotation)
        fa.fit(X=sample_data)
        assert fa.loadings_.shape == (5, 2)
        if rotation:
            assert hasattr(fa, 'rotmat_')
            assert fa.rotmat_.shape == (2, 2)

def test_factor_analysis_get_factor_variance(sample_data):
    fa = FactorAnalysis(n_factors=2)
    fa.fit(sample_data)
    variance = fa.get_factor_variance()
    
    assert variance.shape == (3, 2)
    assert np.allclose(np.sum(variance[1, :]), 1)  # proportions sum to 1
    assert np.allclose(variance[2, -1], 1)  # cumulative proportion ends at 1
    assert np.all(variance >= 0)  # all values should be non-negative
    assert np.all(variance[1, :] <= 1)  # proportions should be <= 1
    assert np.all(np.diff(variance[2, :]) >= 0)  # cumulative proportions should be increasing

def test_factor_analysis_get_factor_variance_not_fitted():
    fa = FactorAnalysis(n_factors=2)
    with pytest.raises(ValueError, match="FactorAnalysis model is not fitted yet"):
        fa.get_factor_variance()

def test_factor_analysis_rotation_consistency(sample_data):
    rotations = [None, 'varimax', 'promax']
    
    for rotation in rotations:
        fa = FactorAnalysis(n_factors=2, rotation=rotation)
        fa.fit(sample_data)
        
        reconstructed_loadings = fa.unrotated_loadings_ @ fa.rotation_matrix_
        assert np.allclose(fa.loadings_, reconstructed_loadings, atol=1e-5)

        if rotation in [None, 'varimax']:
            assert np.allclose(fa.rotation_matrix_ @ fa.rotation_matrix_.T, np.eye(2), atol=1e-5)

        assert np.linalg.matrix_rank(fa.rotation_matrix_) == fa.rotation_matrix_.shape[0]

def test_factor_analysis_rotation_consistency(sample_data):
    rotations = [None, 'varimax', 'promax']
    
    for rotation in rotations:
        fa = FactorAnalysis(n_factors=2, rotation=rotation)
        fa.fit(sample_data)
        
        reconstructed_loadings = fa.unrotated_loadings_ @ fa.rotmat_
        if rotation == 'promax':
            reconstructed_loadings *= fa.scaling_factors_[:, None]
        
        assert np.allclose(fa.loadings_, reconstructed_loadings, atol=1e-5)

        if rotation in [None, 'varimax']:
            assert np.allclose(fa.rotmat_ @ fa.rotmat_.T, np.eye(2), atol=1e-5)

        assert np.linalg.matrix_rank(fa.rotmat_) == fa.rotmat_.shape[0]

        if rotation == 'promax':
            assert hasattr(fa, 'scaling_factors_')
            assert fa.scaling_factors_.shape == (fa.loadings_.shape[0],)

def test_factor_analysis_reproducibility(sample_data):
    fa1 = FactorAnalysis(n_factors=2)
    fa2 = FactorAnalysis(n_factors=2)
    fa1.fit(sample_data)
    fa2.fit(sample_data)
    assert np.allclose(fa1.loadings_, fa2.loadings_)
    assert np.allclose(fa1.uniquenesses_, fa2.uniquenesses_)

def test_factor_analysis_str_representation(fitted_fa):
    str_representation = str(fitted_fa)
    
    assert "Call:" in str_representation
    assert "Uniquenesses:" in str_representation
    assert "Loadings:" in str_representation
    assert "SS loadings" in str_representation
    assert "Proportion Var" in str_representation
    assert f"Factor 1" in str_representation
    assert f"Factor 2" in str_representation
    assert f"Factor 3" not in str_representation
    
    assert "The degrees of freedom for the model is" in str_representation
    assert ("The fit was" in str_representation) or ("Test of the hypothesis" in str_representation)

def test_factor_analysis_format_loadings(fitted_fa):
    formatted_loadings = fitted_fa._format_loadings()
    
    assert isinstance(formatted_loadings, str)
    
    assert "0.00" not in formatted_loadings
    
    assert ".00" in formatted_loadings or " 0.0" in formatted_loadings or "-0." in formatted_loadings


def test_factor_analysis_str_representation_not_fitted():
    fa = FactorAnalysis(n_factors=2)
    assert str(fa) == "FactorAnalysis model is not fitted yet."

def test_factor_analysis_format_matrix(fitted_fa):
    matrix = np.array([[1.23456, 2.34567], [3.45678, 4.56789]])
    
    formatted_matrix = fitted_fa._format_matrix(matrix)
    
    assert isinstance(formatted_matrix, str)
    assert ".235" in formatted_matrix or ".23" in formatted_matrix

def test_factor_analysis_str_with_rotation(sample_data):
    fa = FactorAnalysis(n_factors=2, rotation='varimax')
    fa.fit(sample_data)
    str_representation = str(fa)
    
    assert "Factor Correlations:" in str_representation

def test_factor_analysis_str_with_hypothesis_test(sample_data):
    fa = FactorAnalysis(n_factors=2)
    fa.fit(sample_data)
    fa.STATISTIC = 10.5  
    fa.PVAL = 0.01  
    str_representation = str(fa)
    
    assert "Test of the hypothesis that 2 factors are sufficient." in str_representation
    assert "The chi square statistic is" in str_representation
    assert "The p-value is" in str_representation

def test_factor_analysis_attributes(sample_data):
    fa = FactorAnalysis(n_factors=2)
    fa.fit(X=sample_data)
    assert hasattr(fa, 'n_iter_')
    assert hasattr(fa, 'converged_')
    assert hasattr(fa, 'loglike_')
    assert hasattr(fa, 'dof_')
    assert hasattr(fa, 'STATISTIC')
    assert hasattr(fa, 'PVAL')