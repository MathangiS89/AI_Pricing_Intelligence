# -*- coding: utf-8 -*-
"""
Created on Thu Feb 12 06:38:30 2026

@author: srima
"""

import pandas as pd
from sklearn.model_selection import train_test_split

from Data_Preparation_Pipeline import (
    load_data,
    impute_competitor_prices,
    remove_outliers,
    add_time_features,
    compute_price_deltas,
    compute_delta_summary
)

from Pricing_Model import (
    train_linear,
    train_random_forest,
    train_xgboost
)

from Model_Evaluation_Selection import (
    evaluate_model,
    select_best_model
)

from Elasticity import extract_linear_elasticity


# ------------- CONFIG -------------
DATA_PATH = "sample_data.csv"
TARGET = "volume"
OWN_PRICE = "own_price"
COMP_COLS = ["comp_price_1", "comp_price_2"]
OUTLIER_PERCENTILE = 0.05
TEST_SIZE = 0.25
# ----------------------------------


def main():

    print("Loading data...")
    df = load_data(DATA_PATH)
    df = impute_competitor_prices(df, COMP_COLS)
    df = remove_outliers(df, TARGET, OUTLIER_PERCENTILE)
    df = add_time_features(df, "date")

    df = compute_price_deltas(df, OWN_PRICE, COMP_COLS)
    df = compute_delta_summary(df, COMP_COLS)

    features = ["delta_min", "delta_max", "delta_median", "day_of_week", "month"]

    all_results = []

    entities = df["entity_id"].unique()

    for entity in entities:

        print(f"\nProcessing Entity: {entity}")

        entity_df = df[df["entity_id"] == entity]

        if len(entity_df) < 10:
            print("Not enough data. Skipping.")
            continue

        X = entity_df[features]
        y = entity_df[TARGET]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=42
        )

        results = []

        linear_model = train_linear(X_train, y_train)
        linear_metrics = evaluate_model(linear_model, X_test, y_test)
        results.append({
            "name": "Linear",
            "model": linear_model,
            "metrics": linear_metrics
        })

        rf_model = train_random_forest(X_train, y_train)
        rf_metrics = evaluate_model(rf_model, X_test, y_test)
        results.append({
            "name": "Random Forest",
            "model": rf_model,
            "metrics": rf_metrics
        })

        xgb_model = train_xgboost(X_train, y_train)
        xgb_metrics = evaluate_model(xgb_model, X_test, y_test)
        results.append({
            "name": "XGBoost",
            "model": xgb_model,
            "metrics": xgb_metrics
        })

        best_model = select_best_model(results)

        if best_model is None:
            continue

        all_results.append({
            "entity_id": entity,
            "best_model": best_model["name"],
            "r2": best_model["metrics"]["r2"],
            "mape": best_model["metrics"]["mape"]            
        })

    results_df = pd.DataFrame(all_results)
    #print("\nFinal Summary:")
    #print(results_df)
    
    results_df.to_csv("Model_Results.csv", index=False)
    return results_df

if __name__ == "__main__":
    results_df = main()
