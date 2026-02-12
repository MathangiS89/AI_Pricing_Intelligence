# -*- coding: utf-8 -*-
"""
Created on Wed Feb 11 17:08:09 2026

@author: srima
"""

from sklearn.metrics import r2_score, mean_absolute_percentage_error


def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    r2 = r2_score(y_test, preds)
    mape = mean_absolute_percentage_error(y_test, preds)
    return {"r2": r2, "mape": mape}

def select_best_model(results):
    valid_models = [r for r in results if r["metrics"]["r2"] > 0]
    if not valid_models:
        return None
    return sorted(valid_models,
                  key=lambda x: x["metrics"]["r2"],
                  reverse=True)[0]
