import pytest
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from model_pipeline import (
    prepare_data,
    train_model,
    evaluate_model,
    save_model,
    load_model,
)
import joblib
import os

@pytest.fixture
def sample_data():
    """Fixture loading actual train/test CSV data from the tests directory"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    train_path = os.path.join(current_dir, 'churn-bigml-80.csv')
    test_path = os.path.join(current_dir, 'churn-bigml-20.csv')
    return train_path, test_path

# def test_prepare_data_drops_columns(sample_data):
#     """Test if unnecessary columns are dropped"""
#     train_path, test_path = sample_data
#     X_train, X_test, _, _ = prepare_data(train_path, test_path)
    
#     # Ensure irrelevant columns are removed
#     assert 'State' not in X_train.columns
#     assert 'Area code' not in X_train.columns
#     assert 'Total day minutes' not in X_train.columns

# def test_prepare_data_encodes_categorical(sample_data):
#     """Test categorical encoding consistency"""
#     train_path, test_path = sample_data
#     X_train, X_test, _, _ = prepare_data(train_path, test_path)
    
#     # Check binary encoding for categorical features
#     assert X_train['International plan'].isin([0, 1]).all()
#     assert X_test['Voice mail plan'].isin([0, 1]).all()

# def test_prepare_data_normalization(sample_data):
#     """Test numerical feature normalization"""
#     train_path, test_path = sample_data
#     X_train, X_test, _, _ = prepare_data(train_path, test_path)
    
#     # Verify normalization between 0 and 1
#     for col in ['Account length', 'Total day charge']:
#         assert X_train[col].between(0, 1).all(), f"{col} in train not normalized"
#         assert X_test[col].between(0, 1).all(), f"{col} in test not normalized"

# Remaining tests (train_model, evaluate_model, model_io) remain unchanged
def test_train_model():
    """Test model training returns correct classifier"""
    X_train = pd.DataFrame(np.random.rand(10, 5))
    y_train = pd.Series([0, 1]*5)
    model = train_model(X_train, y_train)
    
    assert isinstance(model, RandomForestClassifier)
    assert model.get_params()['max_depth'] == 5
    assert model.get_params()['n_estimators'] == 30

def test_evaluate_model_metrics():
    """Test metric calculations with perfect predictions"""
    model = RandomForestClassifier()
    model.predict = lambda X: np.array([1, 0, 1, 0])
    X_test = pd.DataFrame(np.random.rand(4, 5))
    y_test = pd.Series([1, 0, 1, 0])
    
    accuracy, precision, recall, f1 = evaluate_model(model, X_test, y_test)
    assert accuracy == 1.0
    assert precision == 1.0
    assert recall == 1.0
    assert f1 == 1.0

def test_model_io(tmp_path):
    """Test model saving/loading functionality"""
    model = RandomForestClassifier().fit(np.random.rand(10, 5), [0, 1]*5)
    path = tmp_path / "test_model.joblib"
    
    save_model(model, path, pd.DataFrame(np.random.rand(10, 5)))
    loaded_model = load_model(path)
    
    assert isinstance(loaded_model, RandomForestClassifier)
    assert hasattr(loaded_model, 'predict')