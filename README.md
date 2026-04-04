# Supermarket Sales Forecasting

A time series forecasting application for predicting supermarket sales using statistical and machine learning models, enabling data-driven inventory and demand planning.

---

## Overview

This project presents an interactive forecasting system that models and predicts daily supermarket sales using time series techniques. Accurate sales forecasting is essential in retail operations, as it helps optimise inventory levels, reduce stockouts, and improve overall supply chain efficiency.

The application integrates data preprocessing, time series modelling, and visualisation into a single workflow, allowing users to generate and analyse forecasts in an intuitive interface.

---

## Features

- Upload custom sales datasets or use structured CSV input  
- Automatic preprocessing of time series data:
  - Date parsing and indexing  
  - Daily aggregation of sales  
  - Handling missing values through interpolation  

- Forecasting models:
  - ARIMA (AutoRegressive Integrated Moving Average)  
  - Prophet (trend and seasonality modelling)  

- Visualisation:
  - Historical sales trends  
  - Forecasted values with time horizon  
  - Confidence intervals (Prophet)  
  - Seasonal components and trend decomposition  

- Export:
  - Download forecast results as CSV  

---

## Approach

The system follows a structured time series pipeline:

1. Data cleaning and preparation of raw sales data  
2. Aggregation of sales into daily time series  
3. Handling missing values and ensuring temporal consistency  
4. Model training using ARIMA and Prophet  
5. Generation of multi-step forecasts  
6. Visualisation of predictions and seasonal patterns  

ARIMA is used to capture autoregressive and moving average components, while Prophet provides a flexible framework for modelling trend shifts and seasonality.

---

## Results

- Produced accurate short- and medium-term sales forecasts  
- Captured recurring seasonal patterns in supermarket demand  
- Enabled improved decision-making for inventory planning  
- Delivered interpretable forecasting outputs suitable for business use  

---

## Tech Stack

- Python  
- pandas  
- numpy  
- matplotlib  
- statsmodels (ARIMA)  
- Prophet  
- Streamlit  

---

## Example Use Case

Users can upload historical sales data, select relevant columns, and generate forecasts for future time periods. The system allows comparison of forecasting models and visual inspection of predicted trends, making it useful for retail analytics and operational planning.

---

## Limitations

- Forecast accuracy depends on data quality and historical coverage  
- ARIMA requires parameter tuning for optimal performance  
- Prophet assumptions may not generalise across all datasets  
- External factors such as promotions or holidays may not be fully captured  

---

## Future Work

- Incorporate external regressors (e.g., holidays, promotions)  
- Hyperparameter optimisation for ARIMA and Prophet  
- Comparison with machine learning and deep learning models (e.g., LSTM)  
- Deployment as a full dashboard with real-time updates  

---

## Installation

```bash
pip install -r requirements.txt
streamlit run app.py
