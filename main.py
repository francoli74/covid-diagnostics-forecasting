from etl.etl import etl
from model.train import model_train
from model.predict import predict
from utils.utils import read_config


def main():
    config = read_config("model")
    prediction_length = config["model"]["prediction_length"]
    train_split = config["model"]["train_split"]

    df = etl()
    ds_train, estimator = model_train(df, train_split, prediction_length)
    predictor = predict(estimator, ds_train)
    return None


if __name__ == "__main__":
    main()
