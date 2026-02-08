import pandas as pd
import numpy as np

FLU_MAP = {"Low": 0, "Moderate": 1, "High": 2}

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes raw input dataframe
    Returns model-ready feature dataframe
    """

    if df is None or df.empty:
        raise ValueError("Empty or invalid dataframe passed to preprocess()")

    df = df.copy()


    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])


    df = df.sort_values(["Hospital_ID", "Department", "Date"])


    df["dayofweek"] = df["Date"].dt.dayofweek #type: ignore
    df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)
    df["weekofyear"] = df["Date"].dt.isocalendar().week.astype(int) #type: ignore
    df["month"] = df["Date"].dt.month #type: ignore
    df["quarter"] = df["Date"].dt.quarter #type: ignore


    if "Precipitation" in df.columns:
        df["is_rain"] = (df["Precipitation"] > 0).astype(int)
        df["Precip_log"] = np.log1p(df["Precipitation"])

    if "Temperature" in df.columns:
        df["temperature_7d_mean"] = (
            df.groupby(["Hospital_ID", "Department"])["Temperature"]
            .shift(1)
            .rolling(7)
            .mean()
        )

    if "Air_Quality_Index" in df.columns:
        df["AQI_7d_mean"] = (
            df.groupby(["Hospital_ID", "Department"])["Air_Quality_Index"]
            .shift(1)
            .rolling(7)
            .mean()
        )


    if "Flu_Activity" in df.columns:
        df["flu_activity"] = df["Flu_Activity"].map(FLU_MAP)

    if "Staffing_Level" in df.columns:
        df["staffing_7d_mean"] = (
            df.groupby(["Hospital_ID", "Department"])["Staffing_Level"]
            .shift(1)
            .rolling(7)
            .mean()
        )


    if "Admissions" in df.columns:
        for lag in [1, 7, 14]:
            df[f"admissions_lag_{lag}"] = (
                df.groupby(["Hospital_ID", "Department"])["Admissions"]
                .shift(lag)
            )

        df["adm_roll_7"] = (
            df.groupby(["Hospital_ID", "Department"])["Admissions"]
            .shift(1)
            .rolling(7)
            .mean()
        )

        df["adm_roll_14"] = (
            df.groupby(["Hospital_ID", "Department"])["Admissions"]
            .shift(1)
            .rolling(14)
            .mean()
        )

    drop_cols = ["Date", "Flu_Activity"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    return df
