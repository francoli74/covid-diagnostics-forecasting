from typing import List, Tuple

import gluonts
import pandas as pd
from gluonts.dataset.pandas import PandasDataset
from gluonts.evaluation import make_evaluation_predictions
from gluonts.torch.model.predictor import PyTorchPredictor


def predict(
    predictor: PyTorchPredictor, ds_test: PandasDataset
) -> Tuple[List[gluonts.model.forecast.SampleForecast], List[pd.DataFrame]]:
    forecast_it, ts_it = make_evaluation_predictions(
        dataset=ds_test, predictor=predictor, num_samples=0
    )
    return list(forecast_it), list(ts_it)
