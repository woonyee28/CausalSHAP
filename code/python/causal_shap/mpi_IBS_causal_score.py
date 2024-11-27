from data_processing import DataProcessor
from models import ModelTrainer
from causal_inference import CausalInference
import pandas as pd
import numpy as np
import pickle

import warnings
warnings.filterwarnings("ignore")

from mpi4py import MPI

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

print(f"comm = {comm}")
print(f"rank = {rank}")
print(f"size = {size}")

# Define base directory and result directory (should be accessible to all processes)
base_dir = '../../../'
result_dir = base_dir + 'result/'

if rank == 0:
    print("Starting ML Pipeline...")
    print(f"Base directory set to: {base_dir}")

    data_path = base_dir + 'dataset/' + 'data_full.xlsx'
    raw_data_path = base_dir + 'dataset/' + 'result_raw.xlsx'

    report_file_path = result_dir + 'report.txt'

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
    y = df_encoded['Group']

    print("Training Random Forest model...")
    param_dist = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 7],
        'min_samples_leaf': [1, 2, 4]
    }
    model_trainer = ModelTrainer(X, y)
    model, best_params = model_trainer.train_random_forest(param_dist)

    print("Evaluating model...")
    accuracy, report = model_trainer.evaluate_model()
    print(accuracy)
    print(report)
    print("Model evaluation completed.")

    print("Saving trained model...")
    model_trainer.save_model(str(result_dir + 'best_random_forest_model.pkl'))
    print("Model saved successfully.")

    # Prepare data to broadcast
    X_train_data = model_trainer.X_train
    X_test_data = model_trainer.X_test
    X_train_columns = X.columns
    model_data = model  # Assuming the model is serializable
    ci_gamma = None  # Will be loaded after ci is initialized
else:
    # Initialize variables for other ranks
    X_train_data = None
    X_test_data = None
    X_train_columns = None
    model_data = None
    ci_gamma = None

# Broadcast necessary data and model to all processes
X_train_data = comm.bcast(X_train_data, root=0)
X_test_data = comm.bcast(X_test_data, root=0)
X_train_columns = comm.bcast(X_train_columns, root=0)
model_data = comm.bcast(model_data, root=0)

# Reconstruct DataFrames from broadcasted data
X_train_scaled_df = pd.DataFrame(X_train_data, columns=X_train_columns)
X_test_scaled_df = pd.DataFrame(X_test_data, columns=X_train_columns)

# Initialize the CausalInference object on all processes
ci = CausalInference(data=X_train_scaled_df, model=model_data, target_variable='Prob_Class_1')

# Load the causal strengths (assuming the file is accessible to all processes)
ci.load_causal_strengths(result_dir + 'Mean_Causal_Effect_IBS.json')

# Now, proceed with distributing the computation of SHAP values
N = len(X_test_scaled_df)
indices = np.arange(N)
split_indices = np.array_split(indices, size)
my_indices = split_indices[rank]

print(f"Process {rank} computing indices: {my_indices}")

# Each process computes the causal SHAP values for its assigned indices
my_results = []
for idx in my_indices:
    x_instance = pd.Series(X_test_scaled_df.iloc[idx], index=X_test_scaled_df.columns)
    phi_normalized = ci.compute_modified_shap_proba(x_instance)
    my_results.append((idx, phi_normalized))

# Gather the results at the root process
all_results = comm.gather(my_results, root=0)

if rank == 0:
    # Combine and sort the results
    combined_results = []
    for res in all_results:
        combined_results.extend(res)
    combined_results.sort(key=lambda x: x[0])
    phi_normalized_list = [phi for i, phi in combined_results]

    # Save the results to a file
    with open(result_dir + 'phi_normalized_list.pkl', 'wb') as f:
        pickle.dump(phi_normalized_list, f)
    print("Causal SHAP values computed and saved successfully.")
