from multiprocessing import Pool, current_process
import os
import psutil
import pickle
from causal_inference import CausalInference
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor


def process_instance(args):
    idx, x_instance, ci = args
    process_name = current_process().name
    print(f"Process {process_name} is processing row {idx}")
    phi_normalized = ci.compute_modified_shap_proba(x_instance, is_classifier=False)
    return (idx, phi_normalized)

def main():
    #####################################
    #     Load Data and Train Model     #
    #####################################
    base_dir = '../../'
    result_dir = base_dir + 'result/R/'
    print("Starting ML Pipeline...")
    print(f"Base directory set to: {base_dir}")
    print("Loading data...")
    data_path = '../../dataset/Synthetic_CV_Dec.csv'
    df = pd.read_csv(data_path)
    print("Data loaded successfully.")
    X = df.drop(columns=['cv_risk'])
    y = df['cv_risk']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("Training Random Forest model...")
    rf = RandomForestRegressor(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
    rf.fit(X_train, y_train)
    print("Model Trained Successfully!")

    #####################################
    #          Causal Inference         #
    #####################################
    ci = CausalInference(data=X_train, model=rf, target_variable='cv_risk')
    ci.load_causal_strengths(result_dir + 'Mean_Causal_Effect_CV_Dec.json')

    instances = [(idx, pd.Series(X_test.iloc[idx], index=X_test.columns), ci) for idx in range(len(X_test))]

    total_available = len(os.sched_getaffinity(0))
    physical_ratio = psutil.cpu_count(logical=False) / psutil.cpu_count(logical=True)
    n_cores = int(total_available * physical_ratio)
    print(f"Using {n_cores} physical cores for parallel processing")
    
    with Pool(processes=n_cores) as pool:
        results = []
        total_instances = len(instances)
        
        for result in pool.imap_unordered(process_instance, instances):
            results.append(result)
            print(f"Processed {len(results)}/{total_instances} instances")

    results.sort(key=lambda x: x[0])
    phi_normalized_list = [phi for _, phi in results]

    with open(result_dir + 'Causal_SHAP_CV_Dec_42.pkl', 'wb') as f:
        pickle.dump(phi_normalized_list, f)
    print("Causal SHAP values computed and saved successfully.")

if __name__ == '__main__':
    main()