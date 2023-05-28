import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def create_full_dataframe(path):
    """
    Creates a full dataframe from the csv files in the given path.
    Args:
        path: Path to the folder containing the csv files.
    Returns:
        df: Dataframe containing all the data from the csv files.
    """

    df = pd.DataFrame()
    files = os.listdir(path)

    for file in files:
        df_temp = pd.read_csv(path + file)
        df = pd.concat([df, df_temp])

    return df

def aggregate_data(df, groupby_column='Selector', columns_to_aggregate=['Performance_score', 'Accuracy']):
    """
    Aggregates the data in the given dataframe.
    Args:
        df: Dataframe containing all the data.
        groupby_column: Column to group the data by.
        columns_to_aggregate: Columns to aggregate.
    Returns:
        df_aggregated: Dataframe containing the aggregated data.
    """
    df_aggregated = df.groupby(groupby_column).agg({columns_to_aggregate[i]: ['mean', 'std'] for i in range(len(columns_to_aggregate))})
    df_aggregated.columns = ['_'.join(col) for col in df_aggregated.columns.values]
    df_aggregated = df_aggregated.reset_index()
    return df_aggregated



if __name__=="__main__":
    df = create_full_dataframe('data/results_artificial/')
    df_aggregated = aggregate_data(df)
    print(df_aggregated)