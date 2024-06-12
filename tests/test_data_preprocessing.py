import pytest
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import os


# Veri setini yükleme
def load_data(file_path):
    df = pd.read_csv(file_path)
    return df


# Veri Keşfi (Exploratory Data Analysis)
def perform_eda(df):
    print("Veri Setinin İlk 5 Satırı:\n", df.head())
    print("\nVeri Seti Hakkında Bilgiler:\n", df.info())
    print("\nVeri Setindeki Eksik Değerler:\n", df.isnull().sum())
    print("\nTemel İstatistikler:\n", df.describe())


# Eksik verileri işleme
def handle_missing_values(df):
    imputer = SimpleImputer(strategy='mean')
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    df[numeric_columns] = imputer.fit_transform(df[numeric_columns])
    return df


# Kategorik değişkenleri dönüştürme
def encode_categorical_values(df):
    # Kategorik sütunları belirle
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns
    # Encoder oluştur
    encoder = OneHotEncoder(sparse_output=False, drop='first')
    # Kategorik sütunları dönüştür
    encoded_columns = pd.DataFrame(encoder.fit_transform(df[categorical_columns]),
                                   columns=encoder.get_feature_names_out(categorical_columns))
    # Orijinal kategorik sütunları düşür
    df = df.drop(categorical_columns, axis=1)
    # Yeni sütunları ekle
    df = pd.concat([df, encoded_columns], axis=1)
    return df


# Verileri normalleştirme
def normalize_data(df):
    scaler = StandardScaler()
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
    return df


# Verileri eğitim ve test setlerine ayırma
def split_data(df, target_column, test_size=0.2, random_state=42):
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test


# Eğitim ve test setlerini kaydetme
def save_datasets(X_train, X_test, y_train, y_test, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    X_train.to_csv(os.path.join(output_dir, 'X_train.csv'), index=False)
    X_test.to_csv(os.path.join(output_dir, 'X_test.csv'), index=False)
    y_train.to_csv(os.path.join(output_dir, 'y_train.csv'), index=False)
    y_test.to_csv(os.path.join(output_dir, 'y_test.csv'), index=False)
    print(f"Datasets saved to {output_dir}")


@pytest.fixture
def sample_data():
    data = {
        'Age': [25, 35, 45, 55],
        'Gender': ['0', '1', '0', '1'],  # 'Gender' sütunu string tipinde olmalı
        'BMI': [22.5, 24.5, 28.0, 30.0],
        'Diagnosis': [0, 1, 0, 1]
    }
    return pd.DataFrame(data)


def test_load_data(sample_data):
    df = sample_data
    assert not df.empty


def test_handle_missing_values(sample_data):
    df = sample_data.copy()
    df.loc[0, 'BMI'] = None
    df = handle_missing_values(df)
    assert df['BMI'].isnull().sum() == 0


def test_encode_categorical_values(sample_data):
    df = sample_data
    df = encode_categorical_values(df)
    assert 'Gender_1' in df.columns


def test_normalize_data(sample_data):
    df = sample_data
    df = normalize_data(df)
    assert df['Age'].mean() < 1


def test_split_data(sample_data):
    df = sample_data
    X_train, X_test, y_train, y_test = split_data(df, 'Diagnosis')
    assert len(X_train) + len(X_test) == len(df)


def test_save_datasets(tmpdir, sample_data):
    df = sample_data
    X_train, X_test, y_train, y_test = split_data(df, 'Diagnosis')
    output_dir = tmpdir.mkdir("data")
    save_datasets(X_train, X_test, y_train, y_test, output_dir)
    assert (output_dir / 'X_train.csv').check()
