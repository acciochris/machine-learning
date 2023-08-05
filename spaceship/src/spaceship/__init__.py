"""Utilities for spaceship titanic"""

import polars as pl

from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder,
)
from sklearn.impute import SimpleImputer
import sklearn

import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    ConfusionMatrixDisplay,
)

def polars_preprocess(df):
    return df.with_columns(
        pl.col("Cabin").str.split("/").list.to_struct(
            fields=["CabinDeck", "CabinNum", "CabinSide"]
        ),
        pl.col("Name").str.split(" ").list.to_struct(
            fields=["FirstName", "LastName"]
        ),
        pl.col("PassengerId").str.split("_").list.to_struct(
            fields=["GroupId", "IdInGroup"]
        ),
    ).unnest("Cabin", "Name", "PassengerId").with_columns(
        pl.col("CabinNum").cast(pl.Int32),
        pl.col("GroupId").cast(pl.Int32),
        pl.col("IdInGroup").cast(pl.Int32),
        pl.col("CryoSleep").cast(pl.Int32),
        pl.col("VIP").cast(pl.Int32),
        # pl.col("Transported").cast(pl.Int32)
    )

imputer_cols = {
    "id": "GroupId IdInGroup".split(),
    "cat": (
        "HomePlanet CryoSleep CabinDeck CabinNum CabinSide "
        "Destination VIP"  # first and last names dropped
    ).split(),
    "num": "Age RoomService FoodCourt ShoppingMall Spa VRDeck".split(),
}

imputer = ColumnTransformer(
    [
        ("id", "passthrough", imputer_cols["id"]),
        ("cat", SimpleImputer(strategy="most_frequent"), imputer_cols["cat"]),
        ("num", SimpleImputer(strategy="median"), imputer_cols["num"]),
    ],
    remainder="drop",  # output column Transported also dropped
    verbose=True,
    verbose_feature_names_out=False,
)

encoder_cols = {
    "one_hot": "HomePlanet CabinDeck CabinSide Destination".split(),
}

encoder = ColumnTransformer(
    [
        ("one_hot", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,  # required for pandas dataframes
        ), encoder_cols["one_hot"])
    ],
    remainder="passthrough",
    verbose=True,
    verbose_feature_names_out=False,
)

scaler_cols = {
    "standard": (
        "GroupId CabinNum Age RoomService FoodCourt ShoppingMall Spa VRDeck"
    ).split(),
}

scaler = ColumnTransformer(
    [
        ("standard", StandardScaler(), scaler_cols["standard"])
    ],
    remainder="passthrough",
    verbose=True,
    verbose_feature_names_out=False,
)

preprocessor = Pipeline(
    [
        ("imputer", imputer),
        ("encoder", encoder),
        ("scaler", scaler),
    ],
    verbose=True,
)

def evaluate(name, y_test, y_pred):
    print(f"Result for {name}:")
    print(f"accuracy: {accuracy_score(y_test, y_pred)}")
    print(f"precision: {precision_score(y_test, y_pred)}")
    print(f"recall: {recall_score(y_test, y_pred)}")
    print(f"f1: {f1_score(y_test, y_pred)}")
    print(f"matthews: {matthews_corrcoef(y_test, y_pred)}")
    plt.close()
    confusion = ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    confusion.plot()
