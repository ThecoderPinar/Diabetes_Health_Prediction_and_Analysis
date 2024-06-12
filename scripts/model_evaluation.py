import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, classification_report
import joblib


# Modeli ve veri setlerini yükleme
def load_model_and_data(model_path, X_test_path, y_test_path):
    model = joblib.load(model_path)
    X_test = pd.read_csv(X_test_path)
    y_test = pd.read_csv(y_test_path).values.ravel().astype('int')
    return model, X_test, y_test


# ROC eğrisini çizme
def plot_roc_curve(model, X_test, y_test, model_name):
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    roc_auc = roc_auc_score(y_test, y_pred_prob)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic - {model_name}')
    plt.legend(loc="lower right")
    plt.savefig(f'plots/{model_name}_roc_curve.png')
    plt.show()


# Confusion Matrix çizme
def plot_confusion_matrix(y_test, y_pred, model_name):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Diabetes', 'Diabetes'],
                yticklabels=['No Diabetes', 'Diabetes'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.savefig(f'plots/{model_name}_confusion_matrix.png')
    plt.show()


# Modeli değerlendirme
def evaluate_model(model_path, X_test_path, y_test_path, model_name):
    model, X_test, y_test = load_model_and_data(model_path, X_test_path, y_test_path)
    y_pred = model.predict(X_test)

    # ROC Curve
    plot_roc_curve(model, X_test, y_test, model_name)

    # Confusion Matrix
    plot_confusion_matrix(y_test, y_pred, model_name)

    # Classification Report
    print(f"Classification Report for {model_name}:\n")
    print(classification_report(y_test, y_pred))


# Ana işlem fonksiyonu
def main():
    model_paths = ["models/logistic_regression.pkl", "models/random_forest.pkl", "models/xgboost.pkl"]
    model_names = ["Logistic Regression", "Random Forest", "XGBoost"]
    X_test_path = "../data/processed/X_test_engineered.csv"
    y_test_path = "../data/processed/y_test.csv"

    for model_path, model_name in zip(model_paths, model_names):
        evaluate_model(model_path, X_test_path, y_test_path, model_name)


if __name__ == "__main__":
    main()
