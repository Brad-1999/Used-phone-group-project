from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import shap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer, MultiLabelBinarizer


def create_train_test_splits(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify_by: str = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Creates stratified training and testing splits of the data.

    Args:
        X (pd.DataFrame): The feature matrix.
        y (pd.Series): The target variable.
        test_size (float): The proportion of the data to include in the test split.
        random_state (int): The random state for reproducibility.
        stratify_by (str): The column to stratify the data by.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: A tuple containing
        the training feature matrix (X_train), the testing feature matrix (X_test),
        the training target variable (y_train), and the testing target variable (y_test).
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=X[stratify_by] if stratify_by else None,
    )
    return X_train, X_test, y_train, y_test


def load_and_preprocess_data(
    file: str | pl.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Loads the cleaned data from a CSV file, performs necessary preprocessing steps,
    and prepares the feature matrix and target vector.

    Args:
        file (str | pl.DataFrame): The path to the cleaned data CSV file or a Polars DataFrame.

    Returns:
        tuple[pd.DataFrame, pd.Series]: A tuple containing the feature matrix (X)
        and target vector (y).
    """

    # Load data using Polars
    try:
        if isinstance(file, pl.DataFrame):
            df = file
        else:
            df = pl.read_csv(file)
    except Exception as e:
        raise ValueError(f"Error loading file: {e}")

    # Apply log transformation to price
    df = df.with_columns(pl.col("price").log().alias("log_price"))

    # Convert capacity to numeric (GB), handling non-numeric values
    capacity_mapping = {
        "less_than_8": 4,  # Assign a representative value
        "8": 8,
        "16": 16,
        "32": 32,
        "64": 64,
        "128": 128,
        "256": 256,
        "512": 512,
        "1024": 1024,
        "more_than_2048": 2048,  # Assign a representative value
    }

    df = df.with_columns(
        pl.col("capacity").replace(capacity_mapping).cast(pl.Int64).alias("capacity_gb")
    )

    # One-hot encode categorical variables
    categorical_columns = ["condition", "origin", "warranty", "brand", "color", "model"]
    df = df.with_columns(pl.col("color").fill_null("unknown"))
    for col in categorical_columns:
        encoder = LabelBinarizer()
        encoded = encoder.fit_transform(df[col])
        for i, category in enumerate(encoder.classes_):
            df = df.with_columns(
                pl.Series(name=f"{col}_{category}", values=encoded[:, i])
            )
    # Multilabel Encoding for `dominant_colors_by_brand`
    # Create brand-color combination column.
    df = df.with_columns((pl.col("brand") + "_" + pl.col("color")).alias("brand_color"))

    # Find top 2 most popular colors per brand
    top_colors_per_brand = (
        df.group_by("brand_color")
        .len()
        .sort(by=["brand_color", "len"], descending=[False, True])
        .group_by("brand_color")
        .head(2)
        .group_by("brand_color")
        .agg(pl.col("brand_color").alias("brand_color_2"))["brand_color_2"]
        .to_list()
    )
    # Get all the colors, and brands from the top colors per brand
    all_top_colors = [x.split("_")[1] for color in top_colors_per_brand for x in color]

    # Group by brand, and create a list of the colors
    top_colors = (
        df.group_by("brand")
        .agg(pl.col("brand_color").alias("brand_colors"))
        .to_pandas()
    )

    # Create a mapping of top color for brand
    mapping_dict = {
        brand: [
            color.split("_")[1]
            for color in colors
            if color.split("_")[1] in all_top_colors and color.split("_")[0] == brand
        ]
        for brand, colors in zip(top_colors["brand"], top_colors["brand_colors"])
    }

    # Convert the mapping to a list of lists for MultiLabelBinarizer
    encoded_list = []
    for brand in df["brand"]:
        encoded_list.append(mapping_dict[brand])

    # Encode and create the final columns
    mlb = MultiLabelBinarizer()
    encoded_multilabel = mlb.fit_transform(encoded_list)

    for i, color in enumerate(mlb.classes_):
        df = df.with_columns(
            pl.Series(name=f"dominant_color_{color}", values=encoded_multilabel[:, i])
        )

    # Create X and y
    X_columns = [
        col
        for col in df.columns
        if col
        not in (
            "price",
            "log_price",
            "list_time",
            "account_name",
            "account_oid",
            "area_name",
            "region_name",
            "ward_name",
            "brand_color",
            "number_of_images",
            "capacity",
            *categorical_columns,
        )
    ]
    X = df.select(X_columns).to_pandas()
    y = df["log_price"].to_pandas()

    return X, y


def calculate_shap_values(
    model: Any,
    X_test: pd.DataFrame,
    X_train: pd.DataFrame,
    sample_size: int = 100,
    random_state: int = 42,
) -> np.ndarray:
    """
    Calculates SHAP values for the provided model using a subset of data to improve
    performance.

    Args:
        model (object): Trained model.
        X_test (pd.DataFrame): Testing features.
        X_train (pd.DataFrame): Training Features.
        sample_size (int): Number of samples to use for SHAP calculation.
        random_state (int): Random state for reproducibility.

    Returns:
        np.ndarray: Array containing the calculated shap values for each feature.
    """
    # Take a sample of X_test to speed up SHAP calculation
    X_test_sample = X_test.sample(
        n=min(sample_size, len(X_test)), random_state=random_state
    ).astype("float64")
    X_train_sample = X_train.sample(
        n=min(sample_size, len(X_train)), random_state=random_state
    ).astype("float64")

    # Create a KernelExplainer or TreeExplainer for different models.
    if hasattr(model, "estimators_"):
        explainer = shap.TreeExplainer(model, X_train_sample)
    else:
        explainer = shap.KernelExplainer(model.predict, X_train_sample)

    shap_values = explainer.shap_values(X_test_sample)

    if type(shap_values) is list:
        shap_values = np.array(shap_values[1])

    return shap_values, X_test_sample
