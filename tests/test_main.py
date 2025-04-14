import pytest
from unittest.mock import patch, MagicMock, call
import argparse
import main
from model_pipeline import prepare_data
import mlflow
import numpy as np
import pandas as pd

@pytest.fixture
def mock_mlflow():
    """Fixture mocking MLflow components"""
    with patch('mlflow.start_run') as mock_start, \
         patch('mlflow.log_params') as mock_log_params, \
         patch('mlflow.log_metrics') as mock_log_metrics, \
         patch('mlflow.sklearn.log_model') as mock_log_model, \
         patch('mlflow.register_model') as mock_register:
        yield {
            'start_run': mock_start,
            'log_params': mock_log_params,
            'log_metrics': mock_log_metrics,
            'log_model': mock_log_model,
            'register_model': mock_register
        }

@pytest.fixture
def mock_es():
    """Fixture mocking Elasticsearch client"""
    with patch('main.Elasticsearch') as mock_es:
        client = MagicMock()
        mock_es.return_value = client
        yield client

@pytest.fixture
def sample_args(tmp_path):
    """Fixture providing command-line arguments"""
    data_path = tmp_path / "data.csv"
    pd.DataFrame().to_csv(data_path)  # Empty dummy data
    return argparse.Namespace(
        data=str(data_path),
        train=True,
        save=str(tmp_path / "model.joblib"),
        evaluate=False,
        load=None
    )

# def test_main_workflow(mock_mlflow, mock_es, sample_args, tmp_path):
#     """Integration test for full training workflow"""
#     # Mock data preparation
#     mock_data = (
#         pd.DataFrame(np.random.rand(10, 5)),  # X_train
#         pd.DataFrame(np.random.rand(5, 5)),   # X_test
#         pd.Series([0, 1]*5),                  # y_train
#         pd.Series([0, 1]*2 + [0]),            # y_test
#     )
    
#     with patch('main.prepare_data', return_value=mock_data), \
#          patch('main.train_model') as mock_train:
        
#         # Mock trained model
#         mock_model = MagicMock()
#         mock_model.predict.return_value = np.array([0, 1, 0, 1, 0])
#         mock_train.return_value = mock_model
        
#         # Execute main with mocked arguments
#         with patch('argparse.ArgumentParser.parse_args', return_value=sample_args):
#             main.main()
    
#     # Verify MLflow logging
#     mock_mlflow['log_params'].assert_called_once_with({
#         'data_path': sample_args.data,
#         'model_type': 'RandomForestClassifier'
#     })
    
#     # Verify model registration
#     mock_mlflow['register_model'].assert_called_once()
    
#     # Verify Elasticsearch logging
#     mock_es.index.assert_called_once_with(
#         index="model_registry_with_mlflow_metrics",
#         document={
#             'run_id': mock.ANY,
#             'model_name': 'RandomForestClassifier',
#             'model_version': mock.ANY,
#             'stage': 'Registered',
#             'accuracy': mock.ANY,
#             'precision': mock.ANY,
#             'recall': mock.ANY,
#             'f1_score': mock.ANY,
#             'model_type': 'RandomForestClassifier',
#             'data_path': sample_args.data,
#         }
#     )

# def test_main_without_data():
#     """Test error handling when no data provided"""
#     with patch('argparse.ArgumentParser.parse_args') as mock_args:
#         mock_args.return_value = argparse.Namespace(data=None)
#         with pytest.raises(SystemExit):
#             main.main()