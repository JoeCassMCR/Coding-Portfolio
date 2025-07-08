"""
Pytest tests for data_loader and preprocessing modules.
"""

import pytest
import pandas as pd
from data_loader import load_data
from preprocessing import feature_engineering, preprocess_data, split_data


@pytest.fixture
def sample_df():
    data = {
        'Time': [0.0, 10000.0],
        'V1': [1.0, -1.0],
        'V2': [0.5, -0.5],
        # ... up to V28
        **{f'V{i}': [0.0, 0.0] for i in range(3, 29)},
        'Amount': [100.0, 200.0],
        'Class': [0, 1]
    }
    return pd.DataFrame(data)


def test_load_data(tmp_path, sample_df, monkeypatch):
    # Save a small csv to tmp_path
    file_path = tmp_path / "test.csv"
    sample_df.to_csv(file_path, index=False)
    monkeypatch.setenv('DATA_PATH', str(file_path))
    df = load_data(str(file_path))
    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] == 2


def test_feature_engineering(sample_df):
    df2 = feature_engineering(sample_df)
    assert 'hour_of_day' in df2.columns
    assert 'log_amount' in df2.columns


def test_preprocess_and_split(sample_df):
    X, y = preprocess_data(sample_df)
    assert X.shape[0] == 2
    assert len(y) == 2
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.5)
    assert X_train.shape == (1, X.shape[1])
    assert X_test.shape == (1, X.shape[1])
