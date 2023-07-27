from etl.etl import etl
from model.train import train_test_split, model_train
from model.predict import predict
from utils.utils import read_config


def main() -> None:
    config = read_config("model")
    prediction_length = config["model"]["prediction_length"]
    train_split = config["model"]["train_split"]

    df = etl()
    ds_train, ds_test = train_test_split(df, train_split)
    predictor = model_train(df, ds_train, prediction_length)
    forecast_it, ts_it = predict(predictor, ds_test)
    print(forecast_it)
    return forecast_it, ts_it


if __name__ == "__main__":
    main()
