# -*- coding: utf-8 -*-
"""
Created on Wed Feb 11 16:15:56 2026

@author: srima
"""

import pandas as pd
#import numpy as np

def load_data(path_or_df):
    """
    Modified to handle both a file path string OR an existing DataFrame.
    This prevents the app from breaking if you pass synthetic data into it.
    """
    if isinstance(path_or_df, pd.DataFrame):
        return path_or_df
    return pd.read_csv(path_or_df)

def impute_competitor_prices(df, comp_cols):
    # Check if columns exist before filling to avoid KeyErrors with synthetic data
    existing_cols = [c for c in comp_cols if c in df.columns]
    if existing_cols:
        if len(df) > 1:
            df[existing_cols] = df[existing_cols].ffill().bfill()
        else:
            df[existing_cols] = df[existing_cols].fillna(0)
    return df

def remove_outliers(df, target, percentile):
    # Safety: Don't remove outliers if the dataset is too small (like in probing)
    if len(df) < 20 or target not in df.columns:
        return df
    lower = df[target].quantile(percentile)
    upper = df[target].quantile(1 - percentile)
    return df[(df[target] > lower) & (df[target] < upper)]

def compute_price_deltas(df, own_price, comp_cols):
    for col in comp_cols:
        if own_price in df.columns and col in df.columns:
            df[f"delta_{col}"] = df[own_price] - df[col]
    return df

def compute_delta_summary(df, comp_cols):
    delta_cols = [f"delta_{c}" for c in comp_cols if f"delta_{c}" in df.columns]
    if delta_cols:
        df["delta_min"] = df[delta_cols].min(axis=1)
        df["delta_max"] = df[delta_cols].max(axis=1)
        df["delta_median"] = df[delta_cols].median(axis=1)
    return df