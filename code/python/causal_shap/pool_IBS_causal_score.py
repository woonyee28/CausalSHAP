from multiprocessing import Pool, cpu_count, current_process
import pandas as pd
import numpy as np
import os
import psutil
import pickle
from data_processing import DataProcessor
from models import ModelTrainer
from causal_inference import CausalInference

def process_instance(args):
    idx, x_instance, ci = args
    process_name = current_process().name
    print(f"Process {process_name} is processing row {idx}")
    phi_normalized = ci.compute_modified_shap_proba(x_instance)
    return (idx, phi_normalized)

def main():
    # Define base directory and result directory
    base_dir = '../../../'
    result_dir = base_dir + 'result/R/'
    
    print("Starting ML Pipeline...")
    print(f"Base directory set to: {base_dir}")

    data_path = base_dir + 'dataset/' + 'data_full.xlsx'
    
    # Data processing
    print("Loading data...")
    data_processor = DataProcessor(data_path=str(data_path))
    df = data_processor.load_data_metabolites()
    print("Data loaded successfully.")

    print("Encoding labels...")
    df_encoded, label_encoder = data_processor.encode_labels(df, label_column='Group')
    print("Labels encoded successfully.")

    X = df_encoded.drop(columns=['Group'])
    y = df_encoded['Group']

    X = X[["xylose", "xanthosine", "uracil", "ribulose/xylulose", "valylglutamine",
           "tryptophylglycine", "succinate", "valine betaine", "ursodeoxycholate sulfate (1)",
           "tricarballylate", "succinimide", "thymine", "syringic acid", "serotonin", "ribitol"]]
    
    # Model training
    print("Training Random Forest model...")
    param_dist = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 7],
        'min_samples_leaf': [1, 2, 4]
    }
    model_trainer = ModelTrainer(X, y, random_state=1010)
    model, best_params = model_trainer.train_random_forest(param_dist)
    print("Model Trained Successfully!")

    # Initialize CausalInference
    ci = CausalInference(data=model_trainer.X_train, model=model, target_variable='Prob_Class_1')
    ci.load_causal_strengths(result_dir + 'Mean_Causal_Effect_IBS.json')

    # Prepare data for parallel processing
    instances = [(idx, pd.Series(model_trainer.X_test.iloc[idx], index=model_trainer.X_test.columns), ci) 
                for idx in range(len(model_trainer.X_test))]

    # Get physical core count
    total_available = len(os.sched_getaffinity(0))
    physical_ratio = psutil.cpu_count(logical=False) / psutil.cpu_count(logical=True)
    n_cores = int(total_available * physical_ratio)
    print(f"Using {n_cores} physical cores for parallel processing")
    
    # Initialize pool with process initialization to set process names
    with Pool(processes=n_cores) as pool:
        # Add tqdm for progress tracking
        results = []
        total_instances = len(instances)
        
        for result in pool.imap_unordered(process_instance, instances):
            results.append(result)
            print(f"Processed {len(results)}/{total_instances} instances")

    # Sort results and extract values
    results.sort(key=lambda x: x[0])
    phi_normalized_list = [phi for _, phi in results]

    # Save results
    with open(result_dir + 'Causal_SHAP_IBS_1010.pkl', 'wb') as f:
        pickle.dump(phi_normalized_list, f)
    print("Causal SHAP values computed and saved successfully.")

if __name__ == '__main__':
    main()