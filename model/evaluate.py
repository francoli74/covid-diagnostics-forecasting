from typing import Tuple

import pandas as pd
from gluonts.evaluation import Evaluator


def evaluate(tss: list, forecasts: list) -> Tuple[dict, pd.DataFrame]:
    evaluator = Evaluator(quantiles=[0.1, 0.5, 0.9])
    agg_metrics, item_metrics = evaluator(tss, forecasts)
    return agg_metrics, item_metrics
