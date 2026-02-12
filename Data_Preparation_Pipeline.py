# -*- coding: utf-8 -*-
"""
Created on Wed Feb 11 16:15:56 2026

@author: srima
"""

import pandas as pd


def load_data(path):
    return pd.read_csv(path)


def impute_competitor_prices(df, comp_cols):
    df[comp_cols] = df[comp_cols].ffill().bfill()
    return df


def remove_outliers(df, target, percentile):
    lower = df[target].quantile(percentile)
    upper = df[target].quantile(1 - percentile)
    return df[(df[target] > lower) & (df[target] < upper)]


def add_time_features(df, date_col):
    df[date_col] = pd.to_datetime(df[date_col])
    df["day_of_week"] = df[date_col].dt.dayofweek
    df["month"] = df[date_col].dt.month
    return df


def compute_price_deltas(df, own_price, comp_cols):
    for col in comp_cols:
        df[f"delta_{col}"] = df[own_price] - df[col]
    return df


def compute_delta_summary(df, comp_cols):
    delta_cols = [f"delta_{c}" for c in comp_cols]
    df["delta_min"] = df[delta_cols].min(axis=1)
    df["delta_max"] = df[delta_cols].max(axis=1)
    df["delta_median"] = df[delta_cols].median(axis=1)
    return df