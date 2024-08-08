import pytest
import numpy as np
from fastoranalysis import FactorAnalysis

@pytest.fixture
def sample_data():
    np.random.seed(42)
    return np.random.rand(100, 5)

def test_factor_analysis_initialization():
    fa = FactorAnalysis(n_factors=2)
    assert fa.n_factors == 2
    assert fa.rotation is None
    assert fa.loadings_ is None
    assert fa.uniquenesses_ is None

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