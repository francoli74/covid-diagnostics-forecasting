from typing import List, Tuple

import gluonts
import pandas as pd
from gluonts.dataset.pandas import PandasDataset
from gluonts.torch.model.deepar.estimator import DeepAREstimator


def model_train(
    df: pd.DataFrame,
    train_split: str,
    prediction_length: int,
) -> Tuple[gluonts.dataset.pandas.PandasDataset,]:
    df_train, _ = train_test_split(df, train_split)
    ds_train = transform_to_ds(df_train)
    static_cat_cardinality = [df["fema_region"].nunique()]
    estimator = training(prediction_length, static_cat_cardinality)
    return ds_train, estimator


def train_test_split(
    df: pd.DataFrame, train_split: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_train = df[df["date"] <= train_split]
    df_test = df[df["date"] > train_split]
    return df_train, df_test


def transform_to_ds(df_train: pd.DataFrame) -> gluonts.dataset.pandas.PandasDataset:
    ds_train = PandasDataset.from_long_dataframe(
        dataframe=df_train,
        target="positive_rate",
        item_id="state_name",
        timestamp="date",
        static_feature_columns=["fema_region"],
        static_features=pd.DataFrame(
            pd.Categorical(sorted(list(df_train["fema_region"].unique()))),
            columns=["fema_region"],
        ),
        feat_dynamic_real=["inconclusive", "negative", "positive"],
    )
    return ds_train


def training(
    prediction_length: int, static_cat_cardinality: List[int]
) -> gluonts.torch.model.deepar.estimator.DeepAREstimator:
    estimator = DeepAREstimator(
        freq="D",
        prediction_length=prediction_length,
        num_layers=2,
        lr=1e-3,
        num_feat_dynamic_real=3,
        num_feat_static_real=0,
        num_feat_static_cat=1,
        cardinality=static_cat_cardinality,
        trainer_kwargs={"accelerator": "mps", "devices": 1, "max_epochs": 2},
    )
    return estimator
