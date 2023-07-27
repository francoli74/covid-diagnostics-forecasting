from typing import List, Tuple
import gluonts
import pandas as pd
from gluonts.dataset.pandas import PandasDataset
from gluonts.torch.model.deepar.estimator import DeepAREstimator


Predictor = gluonts.torch.model.predictor.PyTorchPredictor
GluontsDataset = gluonts.dataset.pandas.PandasDataset


def model_train(
    df: pd.DataFrame,
    ds_train: GluontsDataset,
    prediction_length: int,
) -> Predictor:
    static_cat_cardinality: list = [df["fema_region"].nunique()]
    predictor = training(ds_train, prediction_length, static_cat_cardinality)
    return predictor


def train_test_split(
    df: pd.DataFrame, train_split: str
) -> Tuple[GluontsDataset, GluontsDataset]:
    df_train = df[df["date"] <= train_split]
    df_test = df[df["date"] > train_split]

    ds_train = transform_to_ds(df_train)
    ds_test = transform_to_ds(df_test)
    return ds_train, ds_test


def transform_to_ds(df_train: pd.DataFrame) -> GluontsDataset:
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
    ds_train: GluontsDataset,
    prediction_length: int,
    static_cat_cardinality: List[int],
) -> Predictor:
    estimator = DeepAREstimator(
        freq="D",
        prediction_length=prediction_length,
        context_length=prediction_length * 3,
        num_layers=1,
        lr=1e-3,
        weight_decay=1e-6,  # add a penalty term to the loss function proportional to the sum of the squares of the model's weights multiplied by 1e-6
        dropout_rate=0.1,
        num_feat_dynamic_real=3,
        num_feat_static_real=0,
        # num_feat_static_cat=1,
        cardinality=static_cat_cardinality,
        trainer_kwargs={"accelerator": "mps", "devices": 1, "max_epochs": 2},
    )

    return estimator.train(training_data=ds_train)
