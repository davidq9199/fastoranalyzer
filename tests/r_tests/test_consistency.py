import subprocess
import json
import os
import sys
import numpy as np
from shutil import which

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from fastoranalysis import FactorAnalysis

def find_r_script():
    r_script = which("Rscript")
    if r_script is None:
        possible_paths = [
            r"C:\Program Files\R\R-4.1.3\bin\Rscript.exe",
            r"C:\Program Files\R\R-4.0.5\bin\Rscript.exe",
            r"C:\Program Files\R\R-3.6.3\bin\Rscript.exe",
        ]
        for path in possible_paths:
            if os.path.exists(path):
                return path
        raise FileNotFoundError("Rscript executable not found. Please ensure R is installed and in your PATH.")
    return r_script

def run_r_tests():
    print("Running R tests...")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    r_script_path = os.path.join(script_dir, "test_r_factanal.R")
    
    if not os.path.exists(r_script_path):
        raise FileNotFoundError(f"R script not found at {r_script_path}")
    
    r_executable = find_r_script()
    
    os.chdir(script_dir)
    
    result = subprocess.run([r_executable, r_script_path], capture_output=True, text=True)
    if result.returncode != 0:
        print("R script failed with the following error:")
        print(result.stderr)
        raise Exception("R script execution failed")
    print("R tests completed successfully")

def load_r_results():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_path = os.path.join(script_dir, "output", "r_factanal_results.json")
    
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"R results not found at {results_path}")
    
    with open(results_path, 'r') as f:
        return json.load(f)

def load_test_data(case):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "output", f"test_data_case_{case}.csv")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Test data not found at {data_path}")
    
    return np.loadtxt(data_path, delimiter=',')

def compare_results(py_result, r_result, rtol=1e2, atol=1e3):
    try:
        assert np.allclose(py_result.loadings_, r_result['loadings'], rtol=rtol, atol=atol), "Loadings do not match"
    except AssertionError:
        print("Python loadings:")
        print(py_result.loadings_)
        print("R loadings:")
        print(np.array(r_result['loadings']))
        print("Absolute difference:")
        print(np.abs(py_result.loadings_ - np.array(r_result['loadings'])))
        raise

    assert np.allclose(py_result.uniquenesses_, r_result['uniquenesses'], rtol=rtol, atol=atol), "Uniquenesses do not match"
    assert np.allclose(py_result.correlation_, r_result['correlation'], rtol=rtol, atol=atol), "Correlation matrices do not match"
    assert py_result.n_factors == r_result['factors'][0], "Number of factors does not match"
    assert py_result.dof_ == r_result['dof'][0], "Degrees of freedom do not match"
    
def run_python_tests(r_results):
    print("Running Python tests and comparing with R results...")
    for case in range(1, 4): 
        for rotation in ['varimax', 'promax', 'none']:
            X = load_test_data(case)
            r_result = r_results[f'case_{case}_{rotation}']
            
            n_factors = r_result['factors'][0]
            
            print(f"\nTesting case {case} with {rotation} rotation")
            print(f"Number of factors: {n_factors}")
            print(f"Shape of X: {X.shape}")
            
            try:
                rotation_param = None if rotation == 'none' else rotation
                fa = FactorAnalysis(n_factors=n_factors, rotation=rotation_param)
                fa.fit(X)
                compare_results(fa, r_result)
                print(f"Case {case} with {rotation} rotation: PASSED")
            except Exception as e:
                print(f"Case {case} with {rotation} rotation: FAILED")
                print(f"Error: {str(e)}")
                
                print("Subset of R result:")
                for key in ['factors', 'dof', 'method']:
                    print(f"  {key}: {r_result.get(key)}")
                print("  loadings shape:", np.array(r_result['loadings']).shape)
                print("  uniquenesses length:", len(r_result['uniquenesses']))

def main():
    try:
        run_r_tests()
        r_results = load_r_results()
        run_python_tests(r_results)
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()