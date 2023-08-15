from etl.etl import etl
from model.evaluate import evaluate
from model.predict import predict
from model.train import model_train, train_test_split
from utils.utils import read_config


def main() -> None:
    config = read_config("model")
    prediction_length = config["model"]["prediction_length"]
    train_split = config["model"]["train_split"]

    df = etl()
    ds_train, ds_test = train_test_split(df, train_split)
    predictor = model_train(df, ds_train, prediction_length)
    forecasts, tss = predict(predictor, ds_test)
    rmse = evaluate(tss, forecasts)
    print(f"RMSE: {rmse}")


if __name__ == "__main__":
    main()
