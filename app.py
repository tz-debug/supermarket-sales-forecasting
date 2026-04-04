import os
import json
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

st.set_page_config(page_title="Regression Prediction App", layout="wide")
st.title("Regression Prediction App")
st.write(
    "This application supports built-in datasets from the repository or a user-uploaded CSV. "
    "It trains regression models, evaluates performance, and allows interactive prediction."
)

# ----------------------------
# PATH HANDLING
# ----------------------------
def get_data_path(filename):
    possible_paths = [
        os.path.join("data", filename),
        os.path.join(os.getcwd(), "data", filename),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", filename),
    ]
    for path in possible_paths:
        if os.path.exists(path):
            return path
    return None


SALES_PATH = get_data_path("sales_data.csv")
CAR_PATH = get_data_path("car_purchasing.csv")


@st.cache_data
def load_csv_from_path(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data
def load_csv_from_upload(file) -> pd.DataFrame:
    return pd.read_csv(file)


def guess_column(columns, keywords):
    for keyword in keywords:
        for col in columns:
            if keyword in col.lower():
                return col
    return None


def build_model(model_name: str):
    if model_name == "Linear Regression":
        return LinearRegression()
    return RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        max_depth=8,
        min_samples_split=4
    )


def evaluate_model(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return r2, mae, rmse


def get_default_config(dataset_name: str, df: pd.DataFrame):
    columns = df.columns.tolist()

    if dataset_name == "Sales Dataset":
        target = guess_column(columns, ["sales"])
        features = [c for c in columns if c.lower() in ["tv", "radio", "newspaper"]]

    elif dataset_name == "Car Purchasing Dataset":
        target = guess_column(columns, ["car purchase amount", "purchase amount"])
        features = [
            c for c in columns
            if c.lower() in ["age", "annual salary", "credit card debt", "net worth"]
        ]

    else:
        target = guess_column(columns, ["sales", "target", "amount", "price"])
        features = [c for c in columns if c != target]

    return target, features


# ----------------------------
# DATASET SELECTION
# ----------------------------
dataset_option = st.selectbox(
    "Choose dataset source",
    ["Sales Dataset", "Car Purchasing Dataset", "Upload Your Own"]
)

uploaded_file = None
if dataset_option == "Upload Your Own":
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

# ----------------------------
# LOAD DATA
# ----------------------------
try:
    if dataset_option == "Sales Dataset":
        if SALES_PATH is None:
            st.error("sales_data.csv not found in the data folder.")
            st.stop()
        df = load_csv_from_path(SALES_PATH)

    elif dataset_option == "Car Purchasing Dataset":
        if CAR_PATH is None:
            st.error("car_purchasing.csv not found in the data folder.")
            st.stop()
        df = load_csv_from_path(CAR_PATH)

    else:
        if uploaded_file is None:
            st.info("Upload a CSV file to begin.")
            st.stop()
        df = load_csv_from_upload(uploaded_file)

except Exception as e:
    st.error(f"Could not load dataset: {e}")
    st.stop()

if df.empty:
    st.error("The dataset is empty.")
    st.stop()

default_target, default_features = get_default_config(dataset_option, df)

# ----------------------------
# PREVIEW
# ----------------------------
st.subheader("Dataset Preview")
st.dataframe(df.head(), use_container_width=True)

with st.expander("Dataset information"):
    st.write("Shape:", df.shape)
    st.write("Columns:", list(df.columns))

columns = df.columns.tolist()

# ----------------------------
# COLUMN SELECTION
# ----------------------------
st.subheader("Column Selection")

col1, col2 = st.columns(2)

with col1:
    if default_target in columns:
        target_col = st.selectbox(
            "Target column",
            columns,
            index=columns.index(default_target)
        )
    else:
        target_col = st.selectbox("Target column", columns)

with col2:
    default_features = [c for c in default_features if c in columns and c != target_col]
    candidate_features = [c for c in columns if c != target_col]
    feature_cols = st.multiselect(
        "Feature columns",
        candidate_features,
        default=default_features if default_features else candidate_features[: min(3, len(candidate_features))]
    )

if not feature_cols:
    st.warning("Please select at least one feature column.")
    st.stop()

# ----------------------------
# MODEL SETTINGS
# ----------------------------
st.sidebar.header("Model Settings")
model_name = st.sidebar.selectbox(
    "Select Model",
    ["Linear Regression", "Random Forest Regressor"]
)
test_size = st.sidebar.slider("Test Size", 0.1, 0.4, 0.2, 0.05)
random_state = st.sidebar.number_input("Random State", min_value=0, max_value=9999, value=42)

# ----------------------------
# DATA PREP
# ----------------------------
X = df[feature_cols].copy()
y = pd.to_numeric(df[target_col], errors="coerce")

invalid_cols = []
for col in feature_cols:
    X[col] = pd.to_numeric(X[col], errors="coerce")
    if X[col].notna().sum() == 0:
        invalid_cols.append(col)

if invalid_cols:
    st.error(f"These selected features could not be converted to numeric values: {invalid_cols}")
    st.stop()

valid_mask = y.notna()
for col in feature_cols:
    valid_mask &= X[col].notna()

X = X[valid_mask].copy()
y = y[valid_mask].copy()

if len(X) < 10:
    st.error("Not enough valid rows after cleaning. Please use a larger dataset or different columns.")
    st.stop()

# ----------------------------
# PIPELINE
# ----------------------------
preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            ),
            feature_cols
        )
    ]
)

model = build_model(model_name)

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)

# ----------------------------
# TRAIN
# ----------------------------
if st.button("Train Model"):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state
    )

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    r2, mae, rmse = evaluate_model(y_test, y_pred)

    results_df = pd.DataFrame({
        "Actual": y_test.values,
        "Predicted": y_pred
    })

    st.session_state["pipeline"] = pipeline
    st.session_state["feature_cols"] = feature_cols
    st.session_state["target_col"] = target_col
    st.session_state["metrics"] = {"R2": r2, "MAE": mae, "RMSE": rmse}
    st.session_state["X_test"] = X_test
    st.session_state["y_test"] = y_test
    st.session_state["y_pred"] = y_pred
    st.session_state["results_df"] = results_df
    st.session_state["model_name"] = model_name
    st.session_state["reference_X"] = X

# ----------------------------
# RESULTS
# ----------------------------
if "pipeline" in st.session_state:
    st.subheader("Model Performance")

    m1, m2, m3 = st.columns(3)
    m1.metric("R² Score", f"{st.session_state['metrics']['R2']:.3f}")
    m2.metric("MAE", f"{st.session_state['metrics']['MAE']:.3f}")
    m3.metric("RMSE", f"{st.session_state['metrics']['RMSE']:.3f}")

    st.download_button(
        "Download Metrics JSON",
        data=json.dumps(st.session_state["metrics"], indent=2),
        file_name="metrics.json",
        mime="application/json"
    )

    st.subheader("Actual vs Predicted")
    st.dataframe(st.session_state["results_df"].head(25), use_container_width=True)

    csv_results = st.session_state["results_df"].to_csv(index=False)
    st.download_button(
        "Download Predictions CSV",
        data=csv_results,
        file_name="predictions.csv",
        mime="text/csv"
    )

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(st.session_state["y_test"], st.session_state["y_pred"], alpha=0.7)
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title("Actual vs Predicted")
    st.pyplot(fig)

    # ----------------------------
    # MODEL INTERPRETATION
    # ----------------------------
    st.subheader("Model Interpretation")
    trained_model = st.session_state["pipeline"].named_steps["model"]
    used_features = st.session_state["feature_cols"]

    if hasattr(trained_model, "feature_importances_"):
        imp_df = pd.DataFrame({
            "Feature": used_features,
            "Importance": trained_model.feature_importances_
        }).sort_values("Importance", ascending=False)

        st.dataframe(imp_df, use_container_width=True)

        fig2, ax2 = plt.subplots(figsize=(8, 4))
        ax2.bar(imp_df["Feature"], imp_df["Importance"])
        ax2.set_title("Feature Importance")
        st.pyplot(fig2)

    elif hasattr(trained_model, "coef_"):
        coef_df = pd.DataFrame({
            "Feature": used_features,
            "Coefficient": trained_model.coef_
        }).sort_values("Coefficient", ascending=False)

        st.dataframe(coef_df, use_container_width=True)

        fig3, ax3 = plt.subplots(figsize=(8, 4))
        ax3.bar(coef_df["Feature"], coef_df["Coefficient"])
        ax3.set_title("Feature Coefficients")
        st.pyplot(fig3)

    # ----------------------------
    # MANUAL PREDICTION
    # ----------------------------
    st.subheader("Manual Prediction")
    input_data = {}
    reference_X = st.session_state["reference_X"]

    cols_ui = st.columns(len(used_features))
    for i, feature in enumerate(used_features):
        default_val = float(reference_X[feature].median())
        input_data[feature] = cols_ui[i].number_input(
            feature,
            value=default_val,
            format="%.4f"
        )

    if st.button("Predict"):
        input_df = pd.DataFrame([input_data])
        pred = st.session_state["pipeline"].predict(input_df)[0]
        st.success(f"Predicted value: {pred:.2f}")
