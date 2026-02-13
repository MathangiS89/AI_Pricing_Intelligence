# AI Pricing Intelligence Framework

## Overview

This project is a modular, multi-entity pricing elasticity modeling framework designed to evaluate competitive pricing impact at site level.

It supports:

- Multi-entity modeling
- Automated model selection (Linear, Random Forest, XGBoost)
- Sample data for testing
- Batch result output
- Scalable structure for future MLOps integration
> Note: This repository uses a small sample dataset for demonstration purposes only. 
> The framework is designed to scale to larger, production-grade datasets and 
> multi-entity pricing systems.

---

## Project Structure

Data_Preparation_Pipeline.py
Pricing_Model.py
Model_Evaluation_Selection.py
Elasticity.py
run_pipeline.py
generate_synthetic_data.py
sample_data.csv
requirements.txt


---

## Key Features

- Entity-level pricing elasticity modeling
- Automated best-model selection using R² & MAPE
- Sample dataset for algorithm testing
- Batch output file generation
- Production-ready modular structure

---

## How to Run

1. Install dependencies:
pip install -r requirements.txt
2. Run the pricing pipeline:
python run_pipeline.py
3. Output:
Site-level model selection results
Metrics (R², MAPE)
Results saved as Model_Results.csv

---

## Planned Enhancements

- Margin and volume optimisation scenarios
- Elasticity-based pricing recommendations
- MLflow experiment tracking
- Databricks-compatible MLOps architecture
- AI-assisted executive summaries

---

## Version

v1.0 – Multi-Entity Pricing Elasticity Framework

---




