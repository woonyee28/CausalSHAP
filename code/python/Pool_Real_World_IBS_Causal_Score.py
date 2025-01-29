from multiprocessing import Pool, current_process
import pandas as pd
import os
import psutil
import pickle
from sklearn.ensemble import RandomForestClassifier
from causal_inference import CausalInference
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder

def process_instance(args):
    idx, x_instance, ci = args
    process_name = current_process().name
    print(f"Process {process_name} is processing row {idx}")
    phi_normalized = ci.compute_modified_shap_proba(x_instance)
    return (idx, phi_normalized)

def main():
    #####################################
    #     Load Data and Train Model     #
    #####################################
    base_dir = '../../'
    result_dir = base_dir + 'result/R/'
    print("Starting ML Pipeline...")
    print(f"Base directory set to: {base_dir}")
    data_path = base_dir + 'dataset/' + 'data_full.xlsx'
    print("Loading data...")
    df = pd.read_excel(data_path)
    df = df.drop(columns=['HAD_Anxiety', 'Patient', 'Batch_metabolomics', 'BH', 'Sex', 'Age', 'BMI','Race','Education','HAD_Depression','STAI_Tanxiety', 'Diet_Category','Diet_Pattern'])
    print("Data loaded successfully.")
    print("Encoding labels...")
    label_encoder = LabelEncoder()
    df['Group'] = label_encoder.fit_transform(df['Group'])
    df_encoded = df
    print("Labels encoded successfully.")

    X = df_encoded.drop(columns=['Group'])
    y = df_encoded['Group']

    X = X[["xylose", "xanthosine", "uracil", "ribulose/xylulose", "valylglutamine",
           "tryptophylglycine", "succinate", "valine betaine", "ursodeoxycholate sulfate (1)",
           "tricarballylate", "succinimide", "thymine", "syringic acid", "serotonin", "ribitol"]]
    
    print("Training Random Forest model...")
    param_dist = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 7],
        'min_samples_leaf': [1, 2, 4]
    }
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i)

    rf = RandomForestClassifier(random_state=42)
    random_search = RandomizedSearchCV(
    estimator=rf, param_distributions=param_dist, n_iter=50, cv=3, n_jobs=-1, verbose=2, random_state=42)
    random_search.fit(X_train, y_train)
    model = random_search.best_estimator_
    best_params = random_search.best_params_

    print("Model Trained Successfully!")

    #####################################
    #          Causal Inference         #
    #####################################
    ci = CausalInference(data=X_train, model=model, target_variable='Prob_Class_1')
    ci.load_causal_strengths(result_dir + 'Mean_Causal_Effect_IBS.json')

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

    with open(result_dir + 'Causal_SHAP_IBS_1010.pkl', 'wb') as f:
        pickle.dump(phi_normalized_list, f)
    print("Causal SHAP values computed and saved successfully.")

if __name__ == '__main__':
    main()