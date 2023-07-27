from typing import List, Tuple

import gluonts
import pandas as pd
from gluonts.evaluation import make_evaluation_predictions

Predictor = gluonts.torch.model.predictor.PyTorchPredictor
GluontsDataset = gluonts.dataset.pandas.PandasDataset


def predict(
    predictor: Predictor, ds_test: GluontsDataset
) -> Tuple[List[gluonts.model.forecast.SampleForecast], List[pd.DataFrame]]:
    forecast_it, ts_it = make_evaluation_predictions(
        dataset=ds_test, predictor=predictor, num_samples=0
    )
    return list(forecast_it), list(ts_it)
