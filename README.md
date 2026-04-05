# Regression Prediction App

An interactive web application built using Streamlit for training, evaluating, and interpreting regression models on structured datasets. The application supports both built-in datasets and user-uploaded CSV files, enabling rapid experimentation and end-to-end machine learning workflows.

---

## Live Demo

The application is deployed and accessible here:

https://supermarket-sales-forecasting.streamlit.app/

---

## Overview

This project provides a streamlined interface for applying regression techniques without requiring extensive coding. It integrates data preprocessing, model training, evaluation, and prediction into a single workflow.

---

## Key Features

- Supports multiple regression models:
  - Linear Regression
  - Random Forest Regressor
- Works with built-in datasets or custom uploaded CSV files
- Automated preprocessing pipeline:
  - Missing value imputation
  - Feature scaling
- Model evaluation metrics:
  - R² Score
  - Mean Absolute Error (MAE)
  - Root Mean Squared Error (RMSE)
- Visualization tools:
  - Actual vs Predicted scatter plot
  - Feature importance for tree-based models
  - Coefficient analysis for linear models
- Manual input interface for real-time predictions
- Export functionality:
  - Predictions as CSV
  - Metrics as JSON
- Robust file handling with support for multiple encodings

---

## Project Structure

supermarket-sales-forecasting/
│
├── app.py
├── requirements.txt
├── car_purchasing.csv
├── sales_data.csv

---

## Datasets

### Sales Dataset
- Features: TV, Radio, Newspaper  
- Target: Sales  

### Car Purchasing Dataset
- Features:
  - Age
  - Annual Salary
  - Credit Card Debt
  - Net Worth  
- Target:
  - Car Purchase Amount  

---

## Installation

Clone the repository:

git clone https://github.com/your-username/supermarket-sales-forecasting.git  
cd supermarket-sales-forecasting  

Install dependencies:

pip install -r requirements.txt  

---

## Running the Application

streamlit run app.py  

---

## Usage Workflow

1. Select a dataset (built-in or upload your own)
2. Choose the target variable and input features
3. Select a regression model
4. Train the model
5. Review evaluation metrics and visualizations
6. Generate predictions using manual inputs

---

## Model Pipeline

- Imputation: Median strategy for missing values  
- Scaling: StandardScaler for normalization  
- Model: User-selected regression algorithm  

---

## Key Achievements

- Developed a fully functional end-to-end machine learning application with an interactive user interface  
- Implemented a modular and reusable preprocessing and modeling pipeline using Scikit-learn  
- Enabled dynamic dataset handling with support for user-uploaded data  
- Integrated model evaluation, visualization, and export features within a single application  
- Deployed the application using Streamlit Cloud for real-time access  

---

## Limitations

- Only numeric features are supported  
- No categorical encoding implemented  
- No model persistence or saving functionality  
- Performance depends on data quality and feature selection  

---

## Potential Enhancements

- Support for categorical variables and encoding techniques  
- Hyperparameter tuning interface  
- Model comparison and benchmarking  
- Model export and reuse  
- Advanced interpretability tools (e.g., SHAP, residual analysis)  

---

## Technology Stack

- Python  
- Streamlit  
- Pandas  
- NumPy  
- Scikit-learn  
- Matplotlib  

---

## Purpose

This project demonstrates the implementation of an end-to-end machine learning workflow, including data preprocessing, model training, evaluation, and deployment using an interactive interface. It is suitable for educational use, prototyping, and showcasing practical machine learning skills.
