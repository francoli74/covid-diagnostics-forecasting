import datetime

import numpy as np
import pandas as pd


def etl() -> pd.DataFrame:
    df = extract_data()
    df_transform = transform_data(df)
    return df_transform


def extract_data() -> pd.DataFrame:
    csv_path = (
        "./data/COVID-19_Diagnostic_Laboratory_Testing__PCR_Testing__Time_Series.csv"
    )
    df = pd.read_csv(csv_path, engine="pyarrow")
    return df


def transform_data(df: pd.DataFrame) -> pd.DataFrame:
    df_map = _get_state_region_map(df)
    df_pivot = pivot_data(df)
    df_supp = supplement_missing_date(df_pivot, df_map)
    return df_supp


def _get_state_region_map(df: pd.DataFrame) -> pd.DataFrame:
    df_map = df.groupby(["state_name", "fema_region"]).count().reset_index()
    df_map = df_map[["state_name", "fema_region"]]
    return df_map


def pivot_data(df: pd.DataFrame) -> pd.DataFrame:
    df_trans = (
        pd.pivot_table(
            df,
            values="new_results_reported",
            index=["state_name", "date"],
            columns=["overall_outcome"],
            aggfunc=np.sum,
        )
        .reset_index()
        .fillna(0)
    )

    df_trans.columns = [c.lower() for c in list(df_trans.columns)]
    df_trans["positive_rate"] = df_trans["positive"] / (
        df_trans["inconclusive"] + df_trans["negative"] + df_trans["positive"]
    )
    df_trans["positive_rate"] = df_trans["positive_rate"].fillna(0)
    return df_trans


def supplement_missing_date(df: pd.DataFrame, df_map: pd.DataFrame) -> pd.DataFrame:
    # get all dates from minimum and maximum date
    date_format = "%Y/%m/%d"
    start = datetime.datetime.strptime(df["date"].min(), date_format)
    end = datetime.datetime.strptime(df["date"].max(), date_format)

    date_generated = [
        start + datetime.timedelta(days=x) for x in range(0, (end - start).days)
    ]

    df_dates = pd.DataFrame(
        [date.strftime(date_format) for date in date_generated], columns=["date"]
    )

    # get all states
    df_states = pd.DataFrame(df["state_name"].unique())

    # cross join dates and states as one dataframe
    df_states_date = pd.merge(df_dates, df_states, how="cross")
    df_states_date.columns = ["date", "state_name"]

    df_supp = pd.merge(df_states_date, df, on=["date", "state_name"], how="left")
    df_supp = pd.merge(df_supp, df_map, on="state_name", how="left").fillna(0)
    # df_supp["fema_region"] = df_supp["fema_region"].astype("category")
    return df_supp.sort_values(by=["state_name", "date"]).reset_index(drop=True)
