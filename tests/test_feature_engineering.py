import pytest
import pandas as pd
import numpy as np


# Yeni özellikler oluşturma
def create_new_features(df):
    df['Age_BMI'] = df['Age'] * df['BMI']
    return df


# Polinom özellikler ekleme
def add_polynomial_features(df, degree=2):
    for column in df.select_dtypes(include=['float64', 'int64']).columns:
        for power in range(2, degree + 1):
            df[f'{column}^{power}'] = np.power(df[column], power)
    return df


@pytest.fixture
def sample_data():
    data = {
        'Age': [25, 35, 45, 55],
        'BMI': [22.5, 24.5, 28.0, 30.0]
    }
    return pd.DataFrame(data)


def test_create_new_features(sample_data):
    df = sample_data
    df = create_new_features(df)
    assert 'Age_BMI' in df.columns
    assert (df['Age_BMI'] == df['Age'] * df['BMI']).all()


def test_add_polynomial_features(sample_data):
    df = sample_data
    df = add_polynomial_features(df, degree=2)
    assert 'Age^2' in df.columns
    assert (df['Age^2'] == df['Age'] ** 2).all()
    assert 'BMI^2' in df.columns
    assert (df['BMI^2'] == df['BMI'] ** 2).all()
