import os
import time
import tracemalloc
import numpy as np
import pandas as pd
from fastoranalysis.factor_analysis import FactorAnalysis
from factor_analyzer import FactorAnalyzer
import subprocess

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
        'min_time': min(times),
        'max_time': max(times),
        'mean_memory': np.mean(memories),
        'std_memory': np.std(memories),
        'median_memory': np.median(memories),
        'min_memory': min(memories),
        'max_memory': max(memories)
    }

def run_r_benchmark(file_path, n_factors):
    try:
        r_script_path = os.path.join('benchmarks', 'benchmark.r')
        result = subprocess.run(['Rscript', r_script_path, file_path, str(n_factors)], 
                                capture_output=True, text=True, check=True)
        r_results = pd.read_csv('r_factanal_results.csv')
        result_dict = r_results[r_results['dataset'] == os.path.basename(file_path)].to_dict('records')[0]
        
        for key, value in result_dict.items():
            if pd.isna(value):
                result_dict[key] = 0
        
        return result_dict
    except subprocess.CalledProcessError as e:
        print("Error running R script:")
        print("Stdout:", e.stdout)
        print("Stderr:", e.stderr)
        raise

def benchmark(dataset_path, n_factors):
    dataset = load_dataset(dataset_path)
    df = pd.DataFrame(dataset)
    
    def run_fastor():
        fa_fastor = FactorAnalysis(n_factors=n_factors)
        fa_fastor.fit(df.values)
    
    def run_factor():
        fa_factor = FactorAnalyzer(n_factors=n_factors, method='ml', rotation=None)
        fa_factor.fit(df)
    
    fastor_results = benchmark_function(run_fastor)
    factor_results = benchmark_function(run_factor)
    factanal_results = run_r_benchmark(dataset_path, n_factors)
    
    return {
        "fastor": fastor_results,
        "factor": factor_results,
        "factanal": factanal_results
    }

def main():
    dataset_dir = os.path.join('benchmarks', 'datasets')
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
        
        for method in ['fastor', 'factor', 'factanal']:
            r = results[name][method]
            print(f"{method.capitalize()}:")
            print(f"  Time: mean={r['mean_time']:.4f}s, std={r['std_time']:.4f}s, median={r['median_time']:.4f}s")
            print(f"  Memory: mean={r['mean_memory']:.2f}MB, std={r['std_memory']:.2f}MB, median={r['median_memory']:.2f}MB")
        print("-" * 50)
    
    with open('benchmark_results.csv', 'w') as f:
        f.write("Dataset,Method,Mean Time (s),Std Time (s),Median Time (s),Min Time (s),Max Time (s),")
        f.write("Mean Memory (MB),Std Memory (MB),Median Memory (MB),Min Memory (MB),Max Memory (MB)\n")
        for name, result in results.items():
            for method in ['fastor', 'factor', 'factanal']:
                r = result[method]
                f.write(f"{name},{method},{r['mean_time']:.6f},{r['std_time']:.6f},{r['median_time']:.6f},")
                f.write(f"{r['min_time']:.6f},{r['max_time']:.6f},{r['mean_memory']:.6f},{r['std_memory']:.6f},")
                f.write(f"{r['median_memory']:.6f},{r['min_memory']:.6f},{r['max_memory']:.6f}\n")

if __name__ == "__main__":
    main()