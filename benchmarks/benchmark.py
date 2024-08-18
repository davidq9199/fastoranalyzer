import os
import time
import tracemalloc
import numpy as np
import pandas as pd
from fastoranalysis.factor_analysis import FactorAnalysis
from factor_analyzer import FactorAnalyzer

def load_dataset(file_path):
    return np.loadtxt(file_path, delimiter=',')

def benchmark_function(func, *args, n_runs=10):
    times = []
    memories = []
    for _ in range(n_runs):
        tracemalloc.start()
        start_time = time.perf_counter()
        func(*args)
        end_time = time.perf_counter()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        execution_time = end_time - start_time
        peak_memory_mb = peak / (1024 * 1024)
        times.append(execution_time)
        memories.append(peak_memory_mb)
    
    return {
        'mean_time': np.mean(times),
        'std_time': np.std(times),
        'median_time': np.median(times),
        'min_time': np.min(times),
        'max_time': np.max(times),
        'mean_memory': np.mean(memories),
        'std_memory': np.std(memories),
        'median_memory': np.median(memories),
        'min_memory': np.min(memories),
        'max_memory': np.max(memories)
    }

def benchmark(dataset_path, n_factors):
    dataset = load_dataset(dataset_path)
    df = pd.DataFrame(dataset)
    
    def run_fastor():
        fa_fastor = FactorAnalysis(n_factors=n_factors)
        fa_fastor.fit(df.values)
    
    def run_analyzer():
        fa_analyzer = FactorAnalyzer(n_factors=n_factors, method='ml', rotation=None)
        fa_analyzer.fit(df)
    
    fastor_results = benchmark_function(run_fastor)
    analyzer_results = benchmark_function(run_analyzer)
    
    return {
        "fastor": fastor_results,
        "analyzer": analyzer_results
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
        results[name] = benchmark(file_path, n_factors)
        
        fastor = results[name]['fastor']
        analyzer = results[name]['analyzer']
        
        print(f"FastorAnalyzer:")
        print(f"  Time: mean={fastor['mean_time']:.4f}s, std={fastor['std_time']:.4f}s, median={fastor['median_time']:.4f}s")
        print(f"  Memory: mean={fastor['mean_memory']:.2f}MB, std={fastor['std_memory']:.2f}MB, median={fastor['median_memory']:.2f}MB")
        print(f"FactorAnalyzer:")
        print(f"  Time: mean={analyzer['mean_time']:.4f}s, std={analyzer['std_time']:.4f}s, median={analyzer['median_time']:.4f}s")
        print(f"  Memory: mean={analyzer['mean_memory']:.2f}MB, std={analyzer['std_memory']:.2f}MB, median={analyzer['median_memory']:.2f}MB")
        print("-" * 50)
    
    with open('benchmark_results.csv', 'w') as f:
        f.write("Dataset,Method,Mean Time (s),Std Time (s),Median Time (s),Min Time (s),Max Time (s),")
        f.write("Mean Memory (MB),Std Memory (MB),Median Memory (MB),Min Memory (MB),Max Memory (MB)\n")
        for name, result in results.items():
            for method in ['fastor', 'analyzer']:
                r = result[method]
                f.write(f"{name},{method},{r['mean_time']:.6f},{r['std_time']:.6f},{r['median_time']:.6f},")
                f.write(f"{r['min_time']:.6f},{r['max_time']:.6f},{r['mean_memory']:.6f},{r['std_memory']:.6f},")
                f.write(f"{r['median_memory']:.6f},{r['min_memory']:.6f},{r['max_memory']:.6f}\n")

if __name__ == "__main__":
    main()