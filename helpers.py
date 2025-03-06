import numpy as np
import pandas as pd


def load_experiment_campaign():
    # Load local "Params.xlsx" file
    # study_name = "NARMA5"

    # Load the excel file
    df = pd.read_excel("Params.xlsx", sheet_name="Tabelle1", header=1)

    # Delete rows where all values are NaN
    df = df.dropna(how="all")

    return df


def get_experiment_config(experiment_name):
    df = load_experiment_campaign()
    experiment = df[df["identifier"] == experiment_name]
    return df_to_dict(experiment)


def df_to_dict(df: pd.DataFrame):
    # Convert single-row DataFrame to dictionary
    if len(df) == 1:
        return df.to_dict(orient="records")[0]
    else:
        return df.to_dict()
