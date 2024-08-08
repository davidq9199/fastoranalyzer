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
    assert fa.loadings_ is None
    assert fa.uniquenesses_ is None

def test_factor_analysis_fit(sample_data):
    fa = FactorAnalysis(n_factors=2)
    fa.fit(sample_data)
    assert fa.loadings_.shape == (5, 2)
    assert fa.uniquenesses_.shape == (5,)

def test_factor_analysis_transform(sample_data):
    fa = FactorAnalysis(n_factors=2)
    fa.fit(sample_data)
    transformed = fa.transform(sample_data)
    assert transformed.shape == (100, 2)

def test_factor_analysis_transform_before_fit(sample_data):
    fa = FactorAnalysis(n_factors=2)
    with pytest.raises(ValueError):
        fa.transform(sample_data)