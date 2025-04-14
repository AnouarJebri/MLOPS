import pandas as pd
import numpy as np
import mlflow
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
from mlflow.models.signature import infer_signature


def normalize_data(train, test, columns_to_normalize):
    scaler = MinMaxScaler()
    train[columns_to_normalize] = scaler.fit_transform(train[columns_to_normalize])
    test[columns_to_normalize] = scaler.transform(test[columns_to_normalize])
    return train, test


def drop_columns(data, columns_to_drop):
    return data.drop(columns=columns_to_drop)


def remove_outliers(train, test, num_cols):
    # Compute IQR thresholds based on training data
    thresholds = {}
    for col in num_cols:
        Q1 = train[col].quantile(0.25)
        Q3 = train[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        thresholds[col] = (lower_bound, upper_bound)

    # Apply thresholds to both train and test datasets
    for col in num_cols:
        lower, upper = thresholds[col]
        train = train[(train[col] >= lower) & (train[col] <= upper)]
        test = test[(test[col] >= lower) & (test[col] <= upper)]
    return train, test


def prepare_data(train_path, test_path):
    # Load datasets separately
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    # Drop unnecessary columns
    columns_to_drop = [
        "State",
        "Area code",
        "Total day minutes",
        "Total eve minutes",
        "Total night minutes",
        "Total intl minutes",
    ]
    train_data = drop_columns(train_data, columns_to_drop)
    test_data = drop_columns(test_data, columns_to_drop)

    # Encode categorical variables using LabelEncoder fitted on training data
    label_encoder_international = LabelEncoder()
    label_encoder_vmail = LabelEncoder()
    label_encoder_churn = LabelEncoder()

    train_data["International plan"] = label_encoder_international.fit_transform(
        train_data["International plan"]
    )
    test_data["International plan"] = label_encoder_international.transform(
        test_data["International plan"]
    )

    train_data["Voice mail plan"] = label_encoder_vmail.fit_transform(
        train_data["Voice mail plan"]
    )
    test_data["Voice mail plan"] = label_encoder_vmail.transform(
        test_data["Voice mail plan"]
    )

    train_data["Churn"] = label_encoder_churn.fit_transform(train_data["Churn"])
    test_data["Churn"] = label_encoder_churn.transform(test_data["Churn"])

    # List of numerical columns for outlier removal and normalization
    numerical_columns = [
        "Account length",
        "Number vmail messages",
        "Total day calls",
        "Total day charge",
        "Total eve calls",
        "Total eve charge",
        "Total night calls",
        "Total night charge",
        "Total intl calls",
        "Total intl charge",
        "Customer service calls",
    ]

    # Remove outliers based on training data thresholds and apply to test data as well
    train_data, test_data = remove_outliers(train_data, test_data, numerical_columns)

    # Normalize numerical data: fit scaler on train and apply to test
    train_data, test_data = normalize_data(train_data, test_data, numerical_columns)

    # Split features and target
    X_train = train_data.drop("Churn", axis=1)
    y_train = train_data["Churn"]
    X_test = test_data.drop("Churn", axis=1)
    y_test = test_data["Churn"]

    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train):
    # Updated classifier with restrictions to prevent overfitting and handle class imbalance
    model = RandomForestClassifier(
        n_estimators=50,         # More trees for stability
        max_depth=10,            # Allow some complexity
        min_samples_split=5,      # Prevent small splits
        min_samples_leaf=3,       # Reduce overfitting
        random_state=42,
        class_weight="balanced"
    )
    # cross_val_results = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    # print(f"Cross-validation Accuracy: {cross_val_results.mean()}")
    model.fit(X_train, y_train)

    mlflow.log_param("model_type", "RandomForestClassifier")
    return model


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    mlflow.log_metrics(
        {"accuracy": accuracy, "precision": precision, "recall": recall, "f1_score": f1}
    )

    return accuracy, precision, recall, f1


def save_model(model, filename, X_train):
    joblib.dump(model, filename)

    # Ensure schema consistency by converting integer columns to float64
    X_train_fixed = X_train.copy()
    for col in X_train_fixed.select_dtypes(include=["int"]).columns:
        X_train_fixed[col] = X_train_fixed[col].astype(np.float64)

    signature = infer_signature(X_train_fixed, model.predict(X_train_fixed))
    mlflow.sklearn.log_model(model, "model", signature=signature)

    print(f"💾 Model saved to {filename}")


def load_model(filename):
    return joblib.load(filename)
