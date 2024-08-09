import pytest
import numpy as np
from fastoranalysis import FactorAnalysis

@pytest.fixture
def sample_data():
    np.random.seed(42)
    return np.random.rand(100, 5)

def test_factor_analysis_fit_attributes(sample_data):
    fa = FactorAnalysis(n_factors=2)
    fa.fit(sample_data)
    assert fa.loadings_.shape == (5, 2)
    assert fa.uniquenesses_.shape == (5,)
    assert isinstance(fa.n_iter_, int)
    assert isinstance(fa.loglike_, float)
    assert isinstance(fa.chi_square_, float)
    assert isinstance(fa.dof_, int)
    assert isinstance(fa.p_value_, float)
    assert 0 <= fa.p_value_ <= 1

def test_factor_analysis_invalid_init():
    with pytest.raises(ValueError, match="n_factors must be a positive integer"):
        FactorAnalysis(n_factors=0)
    with pytest.raises(ValueError, match="n_factors must be a positive integer"):
        FactorAnalysis(n_factors=-1)
    with pytest.raises(ValueError, match="rotation must be 'varimax' or None"):
        FactorAnalysis(n_factors=2, rotation='invalid_rotation')

def test_factor_analysis_score(sample_data):
    fa = FactorAnalysis(n_factors=2)
    fa.fit(sample_data)
    scores = fa.score(sample_data)
    assert scores.shape == (100, 2)

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
    fa.fit(sample_data)
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

def test_factor_analysis_reproducibility(sample_data):
    fa1 = FactorAnalysis(n_factors=2)
    fa2 = FactorAnalysis(n_factors=2)
    fa1.fit(sample_data)
    fa2.fit(sample_data)
    assert np.allclose(fa1.loadings_, fa2.loadings_)
    assert np.allclose(fa1.uniquenesses_, fa2.uniquenesses_)