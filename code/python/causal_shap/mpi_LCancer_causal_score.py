import pandas as pd
import numpy as np
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
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
    data_path = '../../../dataset/lungcancerdataset.csv'
    df = pd.read_csv(data_path)

    df = df.drop(columns=['serno', 'followup-time', 'pc1', 'pc2', 'pc3', 'amed', 'dash'])
    df = df.dropna(subset=['lung cancer'])

    X = df.drop(columns=['lung cancer'])
    y = df['lung cancer']

    print("Original class distribution:")
    print(pd.Series(y).value_counts())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    rus = RandomUnderSampler(random_state=42)
    X_train_resampled, y_train_resampled = rus.fit_resample(X_train_scaled, y_train)

    print("\nResampled class distribution:")
    print(pd.Series(y_train_resampled).value_counts())

    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train_resampled, y_train_resampled)

    y_pred = rf.predict(X_test_scaled)
    y_pred_proba = rf.predict_proba(X_test_scaled)[:, 1]

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nROC AUC Score:", roc_auc_score(y_test, y_pred_proba))

    # Prepare data to broadcast
    X_train_data = X_train_resampled
    X_test_data = X_test_scaled
    X_train_columns = X.columns
    model_data = rf  
    ci_gamma = None  
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
ci.load_causal_strengths(result_dir + 'Mean_Causal_Effect_LCancer.json')

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
