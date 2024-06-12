import pandas as pd
from sklearn.preprocessing import PolynomialFeatures


# Yeni özellikler oluşturma
def create_new_features(df):
    # Örneğin, yaş ve BMI'nin çarpımını yeni bir özellik olarak ekleyebiliriz
    df['Age_BMI'] = df['Age'] * df['BMI']
    return df


# Polynomial Features oluşturma
def add_polynomial_features(df, degree=2):
    poly = PolynomialFeatures(degree, include_bias=False)
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    poly_features = poly.fit_transform(df[numeric_columns])
    poly_feature_names = poly.get_feature_names_out(numeric_columns)
    poly_df = pd.DataFrame(poly_features, columns=poly_feature_names)

    df = df.reset_index(drop=True)
    poly_df = poly_df.reset_index(drop=True)

    df = pd.concat([df, poly_df], axis=1)
    return df


# Ana işlem fonksiyonu
def main(train_file_path, test_file_path, train_output_path, test_output_path, degree=2):
    # Eğitim veri seti için feature engineering
    df_train = pd.read_csv(train_file_path)
    df_train = create_new_features(df_train)
    df_train = add_polynomial_features(df_train, degree)
    df_train.to_csv(train_output_path, index=False)
    print(f"Eğitim veri seti için feature engineering tamamlandı ve {train_output_path} dosyasına kaydedildi.")

    # Test veri seti için feature engineering
    df_test = pd.read_csv(test_file_path)
    df_test = create_new_features(df_test)
    df_test = add_polynomial_features(df_test, degree)
    df_test.to_csv(test_output_path, index=False)
    print(f"Test veri seti için feature engineering tamamlandı ve {test_output_path} dosyasına kaydedildi.")


if __name__ == "__main__":
    train_file_path = "../data/processed/X_train.csv"  # Eğitim veri seti yolu
    test_file_path = "../data/processed/X_test.csv"  # Test veri seti yolu
    train_output_path = "../data/processed/X_train_engineered.csv"  # Kaydedilecek yeni eğitim veri seti yolu
    test_output_path = "../data/processed/X_test_engineered.csv"  # Kaydedilecek yeni test veri seti yolu
    degree = 2  # Polynomial degree
    main(train_file_path, test_file_path, train_output_path, test_output_path, degree)
