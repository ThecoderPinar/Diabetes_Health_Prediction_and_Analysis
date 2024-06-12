import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, classification_report
import joblib
import os


# Veri setini yükleme
def load_data(file_path):
    df = pd.read_csv(file_path)
    return df


# Veri setlerini yükleme
def load_train_test_data(X_train_path, X_test_path, y_train_path, y_test_path):
    X_train = pd.read_csv(X_train_path)
    X_test = pd.read_csv(X_test_path)
    y_train = pd.read_csv(y_train_path).values.ravel().astype('int')
    y_test = pd.read_csv(y_test_path).values.ravel().astype('int')
    return X_train, X_test, y_train, y_test


# Modeli kaydetme
def save_model(model, model_name):
    os.makedirs('../models', exist_ok=True)
    joblib.dump(model, f"../models/{model_name}.pkl")
    print(f"Model saved as models/{model_name}.pkl")


# Modeli yükleme
def load_model(model_path):
    model = joblib.load(model_path)
    return model


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
    os.makedirs('plots', exist_ok=True)
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
    os.makedirs('plots', exist_ok=True)
    plt.savefig(f'plots/{model_name}_confusion_matrix.png')
    plt.show()


# Modeli değerlendirme
def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)

    # ROC Curve
    plot_roc_curve(model, X_test, y_test, model_name)

    # Confusion Matrix
    plot_confusion_matrix(y_test, y_pred, model_name)

    # Classification Report
    print(f"Classification Report for {model_name}:\n")
    print(classification_report(y_test, y_pred))


# Eğitim ve test setlerini kaydetme
def save_datasets(X_train, X_test, y_train, y_test, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    X_train.to_csv(os.path.join(output_dir, 'X_train.csv'), index=False)
    X_test.to_csv(os.path.join(output_dir, 'X_test.csv'), index=False)
    y_train.to_csv(os.path.join(output_dir, 'y_train.csv'), index=False)
    y_test.to_csv(os.path.join(output_dir, 'y_test.csv'), index=False)
    print(f"Datasets saved to {output_dir}")
