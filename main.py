import argparse
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from model_pipeline import (
    prepare_data,
    train_model,
    evaluate_model,
    save_model,
    load_model,
)
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient
from elasticsearch import Elasticsearch


# Connexion à Elasticsearch
# Start MLflow experiment
mlflow.set_experiment("Churn Prediction Experiment")
with mlflow.start_run() as run:
    run_id = run.info.run_id
es = Elasticsearch(["http://localhost:9200"], meta_header=False)
model_uri = f"runs:/{run_id}/model"


def log_to_elasticsearch(index_name, data):
    es.index(index=index_name, document=data)


def main():
    parser = argparse.ArgumentParser(
        description="Train and Evaluate Churn Prediction Model"
    )
    parser.add_argument("--data", type=str, help="Path to training or testing dataset")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--save", type=str, help="Path to save trained model locally")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate the model")
    parser.add_argument("--load", type=str, help="Path to load trained model")
    args = parser.parse_args()

    # Load Data
    if args.data:
        X_train, X_test, y_train, y_test = prepare_data(args.data,"churn-bigml-20.csv")
        print(args.data)
    else:
        print("❌ Please provide a data file using the --data argument.")
        return

    # Train Model
    if args.train:
        model = train_model(X_train, y_train)
        print("✅ Model trained successfully.")

        # Log model parameters
        mlflow.log_params(
            {"data_path": args.data, "model_type": "RandomForestClassifier"}
        )

        # Evaluate on training data & log metrics
        train_accuracy, train_precision, train_recall, train_f1 = evaluate_model(
            model, X_train, y_train
        )
        mlflow.log_metrics(
            {
                "train_accuracy": train_accuracy,
                "train_precision": train_precision,
                "train_recall": train_recall,
                "train_f1_score": train_f1,
            }
        )
        print(f"📊 Train Accuracy: {train_accuracy}")
        print(f"🎯 Train Precision: {train_precision}")
        print(f"🔁 Train Recall: {train_recall}")
        print(f"🏆 Train F1 Score: {train_f1}")

        # Log the model artifact with a specified artifact path (e.g., "model")
        signature = infer_signature(X_train, model.predict(X_train))
        mlflow.sklearn.log_model(
            sk_model=model, artifact_path="model", signature=signature
        )

        # Construct model URI and register the model in the Model Registry

        registered_model = mlflow.register_model(model_uri, "RandomForestClassifier")
        print(
            f"📦 Model registered in the MLflow Model Registry as '{registered_model.name}' (version {registered_model.version})."
        )
        # logs
        log_to_elasticsearch(
            "model_registry_with_mlflow_metrics",
            {
                "run_id": run_id,
                "model_name": registered_model.name,
                "model_version": registered_model.version,
                "stage": "Registered",
                "accuracy": train_accuracy,
                "precision": train_precision,
                "recall": train_recall,
                "f1_score": train_f1,
                "model_type": "RandomForestClassifier",
                "data_path": args.data,
            },
        )

        # Save model locally if requested
        if args.save:
            save_model(model, args.save, X_train)
            print(f"💾 Model saved locally to {args.save}")

        # Set Model Stage
        def set_model_stage(model_name, model_version, stage):
            client = MlflowClient()
            client.transition_model_version_stage(
                name=model_name, version=model_version, stage=stage
            )
            print(
                f"🚀 Model {model_name} version {model_version} moved to stage '{stage}'."
            )

        # Inside the training block in main.py, after registering the model:
        registered_model = mlflow.register_model(model_uri, "RandomForestClassifier")

        # Move the model to Staging automatically
        set_model_stage("RandomForestClassifier", registered_model.version, "Staging")

    # Load Model for Evaluation
    if args.evaluate:
        if args.load:
            model = load_model(args.load)
            print(f"📂 Loaded model from {args.load}")
        elif args.train:
            print("⚠️ No model loaded, using freshly trained model.")
        else:
            print("❌ No trained model available. Use --load to provide a model file.")
            return

        # Evaluate on test data & log metrics
        accuracy, precision, recall, f1 = evaluate_model(model, X_test, y_test)
        mlflow.log_metrics(
            {
                "test_accuracy": accuracy,
                "test_precision": precision,
                "test_recall": recall,
                "test_f1_score": f1,
            }
        )

        print(f"📊 Test Accuracy: {accuracy}")
        print(f"🎯 Test Precision: {precision}")
        print(f"🔁 Test Recall: {recall}")
        print(f"🏆 Test F1 Score: {f1}")


if __name__ == "__main__":
    main()
