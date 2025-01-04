from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor

from src.utils import (
    calculate_shap_values,
    create_train_test_splits,
    load_and_preprocess_data,
)


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_type: str,
    model_parameters: Dict[str, Any],
) -> Any:
    """
    Trains a specified machine learning model.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target variable.
        model_type (str): The type of model to train ('ridge', 'knn', 'random_forest').
        model_parameters (Dict[str, Any]): Model-specific hyperparameters.

    Returns:
        Any: The trained model.
    """

    if model_type == "ridge":
        model = Ridge
    elif model_type == "knn":
        model = KNeighborsRegressor
    elif model_type == "random_forest":
        model = RandomForestRegressor
    else:
        raise ValueError(f"The model_type {model_type} is not defined")

    # Use Grid Search if parameter grid exists.
    if "param_grid" in model_parameters:
        grid_search = GridSearchCV(
            model(),
            model_parameters["param_grid"],
            cv=5,
            scoring="neg_mean_absolute_error",
            n_jobs=-1,
        )
        grid_search.fit(X_train, y_train)
        return grid_search.best_estimator_
    else:
        model = model(**model_parameters)
        model.fit(X_train, y_train)
        return model


def evaluate_model(
    model: any, X_test: pd.DataFrame, y_test: pd.Series
) -> Dict[str, float]:
    """
    Evaluates a trained model using specified metrics.

    Args:
        model (any): A trained model.
        X_test (pd.DataFrame): Testing features.
        y_test (pd.Series): Testing target variable.

    Returns:
        Dict[str, float]: A dictionary containing the performance metrics
        (MAE, RMSE, R-squared, MAPE).
    """

    PRICE_THRESHOLD = 80

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred)).item()
    r2 = r2_score(y_test, y_pred)

    # Calculate MAPE
    y_test_np = np.exp(y_test)
    y_pred_np = np.exp(y_pred)
    mape = (np.mean(np.abs((y_test_np - y_pred_np) / y_test_np)) * 100).item()

    # Calculate RAC
    rac = np.sum(np.abs(y_test_np - y_pred_np) <= PRICE_THRESHOLD) / len(y_test)

    metrics = {"mae": mae, "rmse": rmse, "r2": r2, "mape": mape, "rac @ 80USD": rac}
    return metrics


def report_performance(
    metrics: Dict[str, float], shap_values: np.ndarray = None
) -> None:
    """
    Prints the performance of the model in a nicely formatted string.

    Args:
        metrics (Dict[str, float]): Dictionary containing performance metrics.
        shap_values (np.ndarray): Array containing the shap values.

    Returns:
         None
    """
    print("===== Model Performance =====")
    print(f"  MAE:  {metrics['mae']:.4f}")
    print(f"  RMSE: {metrics['rmse']:.4f}")
    print(f"  R2:   {metrics['r2']:.4f}")
    print(f"  MAPE: {metrics['mape']:.2f}%")
    print(f"  RAC @ 80USD: {metrics['rac @ 80USD']:.2f}%")
    print()
    if shap_values is not None:
        print("===== SHAP values =====")
        print(f"Shape of shap values: {shap_values.shape}")
        print(shap_values)


if __name__ == "__main__":
    X, y = load_and_preprocess_data(file_path="data/cleaned_info.csv")
    X_train, X_test, y_train, y_test = create_train_test_splits(
        X, y, test_size=0.2, random_state=42
    )

    # Example parameters
    ridge_params = {"alpha": 1.0}
    knn_params = {"n_neighbors": 5}
    rf_params = {"n_estimators": 100}

    # Train different models
    ridge_model = train_model(X_train, y_train, "ridge", ridge_params)
    knn_model = train_model(X_train, y_train, "knn", knn_params)
    rf_model = train_model(X_train, y_train, "random_forest", rf_params)

    # Evaluate the models
    ridge_metrics = evaluate_model(ridge_model, X_test, y_test)
    knn_metrics = evaluate_model(knn_model, X_test, y_test)
    rf_metrics = evaluate_model(rf_model, X_test, y_test)
    print("Ridge Metrics:", ridge_metrics)
    print("KNN Metrics:", knn_metrics)
    print("Random Forest Metrics:", rf_metrics)

    # Calculate SHAP values
    rf_shap_values, _ = calculate_shap_values(rf_model, X_test, X_train)
    print(rf_shap_values)
