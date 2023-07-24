import gluonts


def predict(
    estimator: gluonts.torch.model.deepar.estimator.DeepAREstimator,
    ds_train: gluonts.dataset.pandas.PandasDataset,
) -> gluonts.torch.model.predictor.PyTorchPredictor:
    return estimator.train(training_data=ds_train)
