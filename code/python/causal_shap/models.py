# models.py
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report, accuracy_score
import joblib

class ModelTrainer:
    def __init__(self, X, y, test_size=0.2, random_state=42):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state)
        self.model = None
        self.best_params = None

    def train_random_forest(self, param_dist, n_iter=50, cv=3):
        rf = RandomForestClassifier(random_state=42)
        random_search = RandomizedSearchCV(
            estimator=rf, param_distributions=param_dist, n_iter=n_iter,
            cv=cv, n_jobs=-1, verbose=2, random_state=42)
        random_search.fit(self.X_train, self.y_train)
        self.model = random_search.best_estimator_
        self.best_params = random_search.best_params_
        return self.model, self.best_params

    def evaluate_model(self):
        y_pred = self.model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        report = classification_report(self.y_test, y_pred)
        return accuracy, report

    def save_model(self, file_path):
        joblib.dump(self.model, file_path)
