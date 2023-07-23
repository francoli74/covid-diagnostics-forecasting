from etl.etl import etl
from model.train import training


def main():
    df = etl()
    df_train = training(df)
    return None


if __name__ == "__main__":
    main()
