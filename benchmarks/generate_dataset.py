import os
import numpy as np

def generate_dataset_by_size(n_features, n_factors, target_mb):
    sample_size_estimate = int((target_mb * 1024 * 1024) / (n_features * 8))
    np.random.seed(0)
    loadings = np.random.rand(n_features, n_factors)
    factors = np.random.rand(sample_size_estimate, n_factors)
    noise = np.random.normal(size=(sample_size_estimate, n_features))
    data = factors @ loadings.T + noise
    return data

def save_dataset_to_csv(dataset, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    np.savetxt(file_path, dataset, delimiter=',')

def main():
    sizes_mb = [1, 10, 50, 100]
    n_features = 50
    n_factors = 5
    
    for size in sizes_mb:
        dataset = generate_dataset_by_size(n_features, n_factors, size)
        file_name = f'dataset_{size}MB_{n_features}features.csv'
        save_dataset_to_csv(dataset, f'./benchmarks/datasets/{file_name}')

if __name__ == "__main__":
    main()