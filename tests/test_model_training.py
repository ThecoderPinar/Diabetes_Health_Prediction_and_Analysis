import pytest
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import joblib
import os


# Veri setlerini yükleme
def load_data(train_path, test_path):
    X_train = pd.read_csv(train_path)
    X_test = pd.read_csv(test_path)
    y_train = pd.read_csv(train_path.replace('X_', 'y_')).values.ravel().astype('int')
    y_test = pd.read_csv(test_path.replace('X_', 'y_')).values.ravel().astype('int')
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
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, f"models/{model_name}.pkl")
    print(f"Model saved as models/{model_name}.pkl")


@pytest.fixture
def sample_data():
    X_train = pd.DataFrame({
        'Age': [25, 35, 45, 55],
        'BMI': [22.5, 24.5, 28.0, 30.0]
    })
    y_train = pd.Series([0, 1, 0, 1])
    return X_train, y_train


def test_train_and_evaluate_model(sample_data):
    X_train, y_train = sample_data
    X_test, y_test = X_train.copy(), y_train.copy()

    # Logistic Regression
    lr = LogisticRegression()
    train_and_evaluate_model(X_train, X_test, y_train, y_test, lr, "logistic_regression")

    # Random Forest
    rf = RandomForestClassifier(n_estimators=10)
    train_and_evaluate_model(X_train, X_test, y_train, y_test, rf, "random_forest")

    # XGBoost
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    train_and_evaluate_model(X_train, X_test, y_train, y_test, xgb, "xgboost")

    # Modellerin kaydedildiğini kontrol etme
    assert os.path.exists('models/logistic_regression.pkl')
    assert os.path.exists('models/random_forest.pkl')
    assert os.path.exists('models/xgboost.pkl')


def test_load_data(tmpdir):
    # Geçici veri dosyaları oluşturma
    train_file = tmpdir.join("X_train.csv")
    test_file = tmpdir.join("X_test.csv")
    train_file.write("Age,BMI\n25,22.5\n35,24.5\n45,28.0\n55,30.0")
    test_file.write("Age,BMI\n25,22.5\n35,24.5\n45,28.0\n55,30.0")

    y_train_file = tmpdir.join("y_train.csv")
    y_test_file = tmpdir.join("y_test.csv")
    y_train_file.write("Diagnosis\n0\n1\n0\n1")
    y_test_file.write("Diagnosis\n0\n1\n0\n1")

    # Veri setlerini yükleme
    X_train, X_test, y_train, y_test = load_data(str(train_file), str(test_file))

    # Veri setlerinin doğru yüklendiğini kontrol etme
    assert X_train.shape == (4, 2)
    assert X_test.shape == (4, 2)
    assert y_train.shape == (4,)
    assert y_test.shape == (4,)
