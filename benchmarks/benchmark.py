import os
import time
import tracemalloc
import numpy as np
import pandas as pd
from fastoranalysis.factor_analysis import FactorAnalysis
from factor_analyzer import FactorAnalyzer

def load_dataset(file_path):
    return np.loadtxt(file_path, delimiter=',')

def benchmark_function(func, *args):
    tracemalloc.start()
    start_time = time.time()
    func(*args)
    end_time = time.time()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    execution_time = end_time - start_time
    peak_memory_mb = peak / (1024 * 1024)
    return execution_time, peak_memory_mb

def benchmark(dataset_path, n_factors):
    dataset = load_dataset(dataset_path)
    df = pd.DataFrame(dataset)
    
    def run_fastor():
        fa_fastor = FactorAnalysis(n_factors=n_factors)
        fa_fastor.fit(df.values)
        return fa_fastor.loadings_, fa_fastor.uniquenesses_

    def run_analyzer():
        fa_analyzer = FactorAnalyzer(n_factors=n_factors, method='ml', rotation=None)
        fa_analyzer.fit(df)
        return fa_analyzer.loadings_, fa_analyzer.get_uniquenesses()

    fastor_time, fastor_memory = benchmark_function(run_fastor)
    analyzer_time, analyzer_memory = benchmark_function(run_analyzer)

    return {
        "fastor_time": fastor_time,
        "analyzer_time": analyzer_time,
        "fastor_memory": fastor_memory,
        "analyzer_memory": analyzer_memory,
    }

def main():
    dataset_dir = './benchmarks/datasets'
    datasets = {
        "1MB": (os.path.join(dataset_dir, 'dataset_1MB_50features.csv'), 5),
        "10MB": (os.path.join(dataset_dir, 'dataset_10MB_50features.csv'), 5),
        "50MB": (os.path.join(dataset_dir, 'dataset_50MB_50features.csv'), 5),
        "100MB": (os.path.join(dataset_dir, 'dataset_100MB_50features.csv'), 5),
    }

    results = {}
    for name, (file_path, n_factors) in datasets.items():
        print(f"Benchmarking dataset: {name}")
        result = benchmark(file_path, n_factors)
        results[name] = result
        print(f"FastorAnalyzer: Time = {result['fastor_time']:.4f}s, Memory = {result['fastor_memory']:.2f}MB")
        print(f"FactorAnalyzer: Time = {result['analyzer_time']:.4f}s, Memory = {result['analyzer_memory']:.2f}MB")
        print("-" * 50)

    with open('benchmark_results.csv', 'w') as f:
        f.write("Dataset,Fastor Time (s),FactorAnalyzer Time (s),Fastor Memory (MB),FactorAnalyzer Memory (MB)\n")
        for name, result in results.items():
            f.write(f"{name},{result['fastor_time']:.4f},{result['analyzer_time']:.4f},{result['fastor_memory']:.2f},{result['analyzer_memory']:.2f}\n")

if __name__ == "__main__":
    main()