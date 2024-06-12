import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import joblib
import os


# Veri setlerini yükleme
def load_data(X_train_path, X_test_path, y_train_path, y_test_path):
    X_train = pd.read_csv(X_train_path)
    X_test = pd.read_csv(X_test_path)
    y_train = pd.read_csv(y_train_path).values.ravel().astype('int')  # Hedef değişkeni integer tipine dönüştür
    y_test = pd.read_csv(y_test_path).values.ravel().astype('int')  # Hedef değişkeni integer tipine dönüştür

    return X_train, X_test, y_train, y_test


# Model eğitimi ve değerlendirmesi
def train_and_evaluate_model(X_train, X_test, y_train, y_test, model, model_name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_prob)
    cm = confusion_matrix(y_test, y_pred)

    print(f"Model: {model_name}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    print("Confusion Matrix:")
    print(cm)
    print("\n")

    # Modeli kaydetme
    os.makedirs('../models', exist_ok=True)
    joblib.dump(model, f"../models/{model_name}.pkl")
    print(f"Model saved as models/{model_name}.pkl")


# Ana işlem fonksiyonu
def main(X_train_path, X_test_path, y_train_path, y_test_path):
    X_train, X_test, y_train, y_test = load_data(X_train_path, X_test_path, y_train_path, y_test_path)

    # Lojistik Regresyon
    lr = LogisticRegression(max_iter=1000)
    train_and_evaluate_model(X_train, X_test, y_train, y_test, lr, "logistic_regression")

    # Random Forest
    rf = RandomForestClassifier(n_estimators=100)
    train_and_evaluate_model(X_train, X_test, y_train, y_test, rf, "random_forest")

    # XGBoost
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    train_and_evaluate_model(X_train, X_test, y_train, y_test, xgb, "xgboost")


if __name__ == "__main__":
    X_train_path = "../data/processed/X_train_engineered.csv"  # Eğitim veri seti yolu
    X_test_path = "../data/processed/X_test_engineered.csv"  # Test veri seti yolu
    y_train_path = "../data/processed/y_train.csv"  # Eğitim hedef değişkeni veri seti yolu
    y_test_path = "../data/processed/y_test.csv"  # Test hedef değişkeni veri seti yolu
    main(X_train_path, X_test_path, y_train_path, y_test_path)
