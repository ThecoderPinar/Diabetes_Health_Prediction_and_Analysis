import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import os


# Veri setini yükleme
def load_data(file_path):
    df = pd.read_csv(file_path)
    return df


# Veri Keşfi (Exploratory Data Analysis)
def perform_eda(df, dataset_name=""):
    print(f"{dataset_name} Veri Setinin İlk 5 Satırı:\n", df.head())
    print(f"\n{dataset_name} Veri Seti Hakkında Bilgiler:\n", df.info())
    print(f"\n{dataset_name} Veri Setindeki Eksik Değerler:\n", df.isnull().sum())
    print(f"\n{dataset_name} Temel İstatistikler:\n", df.describe())

    # Kategorik değişkenlerin dağılımı
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns
    for column in categorical_columns:
        plt.figure(figsize=(10, 5))
        sns.countplot(data=df, x=column)
        plt.title(f'{dataset_name} {column} Dağılımı')
        plt.show()

    # Sayısal değişkenlerin dağılımı
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    df[numeric_columns].hist(bins=15, figsize=(20, 15))
    plt.suptitle(f'{dataset_name} Sayısal Değişkenlerin Dağılımı')
    plt.show()

    # Korelasyon matrisi (DoctorInCharge sütunu çıkarıldı)
    if 'DoctorInCharge' in df.columns:
        df_corr = df.drop(columns=['DoctorInCharge'])
    else:
        df_corr = df.copy()
    plt.figure(figsize=(15, 10))
    sns.heatmap(df_corr.corr(), annot=True, cmap='coolwarm')
    plt.title(f'{dataset_name} Korelasyon Matrisi')
    plt.show()


# Eksik verileri işleme
def handle_missing_values(df):
    imputer = SimpleImputer(strategy='mean')
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    df[numeric_columns] = imputer.fit_transform(df[numeric_columns])
    return df


# Kategorik değişkenleri dönüştürme
def encode_categorical_values(df):
    categorical_columns = df.select_dtypes(include=['object']).columns
    encoder = OneHotEncoder(sparse_output=False, drop='first')
    encoded_columns = pd.DataFrame(encoder.fit_transform(df[categorical_columns]))
    encoded_columns.columns = encoder.get_feature_names_out(categorical_columns)
    df = df.drop(categorical_columns, axis=1)
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


# Ana işlem fonksiyonu
def main(file_path, target_column, output_dir):
    df = load_data(file_path)
    perform_eda(df, dataset_name="Orijinal")
    df = handle_missing_values(df)
    df = encode_categorical_values(df)
    df = normalize_data(df)
    X_train, X_test, y_train, y_test = split_data(df, target_column)
    save_datasets(X_train, X_test, y_train, y_test, output_dir)

    # Eğitim ve test setleri için sütun analizleri
    perform_eda(pd.concat([X_train, y_train], axis=1), dataset_name="Eğitim Seti")
    perform_eda(pd.concat([X_test, y_test], axis=1), dataset_name="Test Seti")

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    file_path = "../data/raw/diabetes_data.csv"  # Veri seti yolu
    target_column = "Diagnosis"  # Hedef değişken
    output_dir = "../data/processed"  # Kaydedilecek dizin
    X_train, X_test, y_train, y_test = main(file_path, target_column, output_dir)
    print("Veri ön işleme tamamlandı ve veriler eğitim/test setlerine ayrıldı.")
