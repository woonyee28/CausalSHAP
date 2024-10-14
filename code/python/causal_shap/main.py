# main.py
import argparse
from pathlib import Path
from data_processing import DataProcessor
from models import ModelTrainer
from feature_selection import FeatureSelector
from visualization import Visualizer
from causal_inference import CausalInference

def main(base_dir):
    print("Starting ML Pipeline...")
    base_dir = Path(base_dir).resolve()
    print(f"Base directory set to: {base_dir}")

    data_path = base_dir / 'dataset' / 'data_full.xlsx'
    raw_data_path = base_dir / 'dataset' / 'result_raw.xlsx'
    result_dir = base_dir / 'result'
    result_dir.mkdir(parents=True, exist_ok=True)
    print(f"Result directory set to: {result_dir}")

    report_file_path = result_dir / 'report.txt'

    print("Loading data...")
    data_processor = DataProcessor(data_path=str(data_path))
    df = data_processor.load_data_metabolites()
    print("Data loaded successfully.")

    print("Preprocessing raw data...")
    raw_df = data_processor.preprocess_raw_data(raw_data_path=str(raw_data_path))
    print("Raw data preprocessed successfully.")

    print("Encoding labels...")
    df_encoded, label_encoder = data_processor.encode_labels(df, label_column='Group')
    print("Labels encoded successfully.")

    X = df_encoded.drop(columns=['Group'])
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
    print("Model trained successfully.")

    print("Evaluating model...")
    accuracy, report = model_trainer.evaluate_model()
    print("Model evaluation completed.")

    print("Saving trained model...")
    model_trainer.save_model(str(result_dir / 'best_random_forest_model.pkl'))
    print("Model saved successfully.")

    print("Performing feature selection using Gini importance...")
    feature_selector = FeatureSelector(model, model_trainer.X_train)
    selected_features_gini = feature_selector.gini_importance(threshold=0.01)
    print("Feature selection (Gini importance) completed.")

    print("Performing feature selection using SHAP importance...")
    selected_features_shap = feature_selector.shap_importance(model_trainer.X_test, threshold=0.005)
    print("Feature selection (SHAP importance) completed.")

    print("Generating SHAP summary plot...")
    visualizer = Visualizer()
    shap_summary_path = result_dir / 'shap_summary.png'
    visualizer.plot_shap_summary(model, model_trainer.X_test, str(shap_summary_path))
    print(f"SHAP summary plot saved at: {shap_summary_path}")

    print("Performing causal inference...with SHAP selected and Group")
    causal_features = selected_features_shap.to_list() + ['Group']
    df_causal = df_encoded[causal_features]
    causal_inference = CausalInference(df_causal)
    causal_graph = causal_inference.run_pc_algorithm()
    print("Causal inference completed.")

    print("Drawing causal graph...")
    causal_graph_path = result_dir / 'causal_graph.png'
    causal_inference.draw_graph(str(causal_graph_path))
    print(f"Causal graph saved at: {causal_graph_path}")

    print("Writing report...")
    with report_file_path.open('w') as report_file:
        report_file.write("First few rows of the dataset:\n")
        report_file.write(df.head().to_string())
        report_file.write("\n\n")
        report_file.write("Best Parameters:\n")
        report_file.write(str(best_params))
        report_file.write("\n\n")
        report_file.write(f"Accuracy: {accuracy * 100:.2f}%\n\n")
        report_file.write("Classification Report:\n")
        report_file.write(report)
        report_file.write("\n\n")
        report_file.write("Selected Features (Gini Importance):\n")
        report_file.write(', '.join(selected_features_gini))
        report_file.write("\n\n")
        report_file.write("Selected Features (SHAP Importance):\n")
        report_file.write(', '.join(selected_features_shap))
        report_file.write("\n\n")
        report_file.write(f"SHAP summary plot saved as '{shap_summary_path}'.\n\n")
        report_file.write(f"Causal graph saved as '{causal_graph_path}'.\n")
    print("Report written successfully.")

    print(f"Report generated at '{report_file_path}'.")
    print("Process completed successfully.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run ML Pipeline and Generate Report')
    parser.add_argument('--base_dir', type=str, default='../../..', help='Base directory path')
    args = parser.parse_args()
    main(args.base_dir)
