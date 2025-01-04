from src.csv_extraction import extract_csv
from src.preprocessing import preprocess
from src.trainer import (
    calculate_shap_values,
    create_train_test_splits,
    evaluate_model,
    load_and_preprocess_data,
    report_performance,
    train_model,
)


def main():
    dfs = extract_csv("data/2024-10-24.jsonl")
    _, info = dfs["description"], dfs["info"]

    # Preprocess data
    df = preprocess(info)

    # Train model
    X, y = load_and_preprocess_data(df)
    X_train, X_test, y_train, y_test = create_train_test_splits(
        X, y, test_size=0.2, random_state=42
    )

    # Example parameters
    ridge_params = {"alpha": 1.0, "param_grid": {"alpha": [0.01, 0.1, 1.0, 10.0]}}
    knn_params = {
        "n_neighbors": 5,
        "param_grid": {"n_neighbors": [3, 5, 7], "weights": ["uniform", "distance"]},
    }
    rf_params = {
        "n_estimators": 100,
        "param_grid": {
            "n_estimators": [50, 100, 150],
            "max_depth": [None, 5, 10],
            "min_samples_split": [2, 5],
        },
    }

    # Train different models
    ridge_model = train_model(X_train, y_train, "ridge", ridge_params)
    knn_model = train_model(X_train, y_train, "knn", knn_params)
    rf_model = train_model(X_train, y_train, "random_forest", rf_params)

    # Evaluate the models
    ridge_metrics = evaluate_model(ridge_model, X_test, y_test)
    knn_metrics = evaluate_model(knn_model, X_test, y_test)
    rf_metrics = evaluate_model(rf_model, X_test, y_test)

    # Calculate SHAP values
    rf_shap_values, _ = calculate_shap_values(rf_model, X_test, X_train)
    print("Ridge result:")
    report_performance(ridge_metrics)
    print("KNN result:")
    report_performance(knn_metrics)
    print("Random Forest result:")
    report_performance(rf_metrics, rf_shap_values)


if __name__ == "__main__":
    main()
