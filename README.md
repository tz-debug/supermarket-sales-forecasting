# Advertising Sales Prediction

A Streamlit machine learning application for predicting sales from advertising spend using regression models.

## Features

- Upload CSV dataset
- Train Linear Regression or Random Forest Regressor
- Evaluate with R², MAE, and RMSE
- View actual vs predicted sales
- Inspect feature coefficients or importance
- Predict sales manually from user input

## Installation

```bash
pip install -r requirements.txt
streamlit run app.py

##Repository Structure
.
├── app.py
├── requirements.txt
├── README.md
└── data/
    ├── sales_data.csv
    └── car_purchasing.csv
