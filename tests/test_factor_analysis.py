import pytest
import numpy as np
from fastoranalysis import FastorAnalysis

def test_fastor_analysis_initialization():
    fa = FastorAnalysis(n_factors=2)
    assert fa.n_factors == 2

def test_fastor_analysis_fit_input():
    fa = FastorAnalysis(n_factors=2)
    X = np.random.rand(100, 5)
    fa.fit(X)