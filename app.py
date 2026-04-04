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

st.set_page_config(page_title="Sales Prediction App", layout="wide")
st.title("Sales Prediction App")

st.write(
    "This application predicts target values using regression models. "
    "You can use a built-in dataset from the repository or upload your own CSV file."
)

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


@st.cache_data
def load_data_from_path(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data
def load_data_from_upload(file) -> pd.DataFrame:
    return pd.read_csv(file)


def build_model(model_name: str):
    if model_name == "Linear Regression":
        return LinearRegression()
    return RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        max_depth=6
    )


def evaluate_model(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return r2, mae, rmse


def guess_column(columns, keywords):
    for keyword in keywords:
        for col in columns:
            if keyword in col.lower():
                return col
    return None


# ----------------------------
# LOAD DATA
# ----------------------------
try:
    if dataset_option == "Sales Dataset":
        df = load_data_from_path("data/sales_data.csv")
        default_target = guess_column(df.columns, ["sales"])
        default_features = [
            c for c in df.columns
            if c.lower() in ["tv", "radio", "newspaper"]
        ]

    elif dataset_option == "Car Purchasing Dataset":
        df = load_data_from_path("data/car_purchasing.csv")
        default_target = guess_column(df.columns, ["car purchase amount", "purchase amount"])
        default_features = [
            c for c in df.columns
            if c.lower() in [
                "age",
                "annual salary",
                "credit card debt",
                "net worth"
            ]
        ]

    else:
        if uploaded_file is None:
            st.info("Upload a CSV file to begin.")
            st.stop()

        df = load_data_from_upload(uploaded_file)
        default_target = guess_column(df.columns, ["sales", "target", "amount", "price"])
        default_features = [c for c in df.columns if c != default_target]

except Exception as e:
    st.error(f"Could not load dataset: {e}")
    st.stop()

if df.empty:
    st.error("The dataset is empty.")
    st.stop()

# ----------------------------
# PREVIEW
# ----------------------------
st.subheader("Dataset Preview")
st.dataframe(df.head(), use_container_width=True)

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
    feature_cols = st.multiselect(
        "Feature columns",
        [c for c in columns if c != target_col],
        default=default_features
    )

if not feature_cols:
    st.warning("Please select at least one feature column.")
    st.stop()

# ----------------------------
# MODEL SETTINGS
# ----------------------------
model_name = st.sidebar.selectbox(
    "Select Model",
    ["Linear Regression", "Random Forest Regressor"]
)

test_size = st.sidebar.slider("Test Size", 0.1, 0.4, 0.2, 0.05)

# ----------------------------
# DATA CLEANING
# ----------------------------
X = df[feature_cols].copy()
y = pd.to_numeric(df[target_col], errors="coerce")

for col in feature_cols:
    X[col] = pd.to_numeric(X[col], errors="coerce")

valid_mask = y.notna()
for col in feature_cols:
    valid_mask &= X[col].notna()

X = X[valid_mask]
y = y[valid_mask]

if len(X) < 10:
    st.error("Not enough valid rows after cleaning. Please use a larger dataset.")
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
# TRAIN MODEL
# ----------------------------
if st.button("Train Model"):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    r2, mae, rmse = evaluate_model(y_test, y_pred)

    st.session_state["pipeline"] = pipeline
    st.session_state["feature_cols"] = feature_cols
    st.session_state["X_test"] = X_test
    st.session_state["y_test"] = y_test
    st.session_state["y_pred"] = y_pred
    st.session_state["metrics"] = {"R2": r2, "MAE": mae, "RMSE": rmse}
    st.session_state["model_name"] = model_name

# ----------------------------
# RESULTS
# ----------------------------
if "pipeline" in st.session_state:
    st.subheader("Model Performance")

    m1, m2, m3 = st.columns(3)
    m1.metric("R² Score", f"{st.session_state['metrics']['R2']:.3f}")
    m2.metric("MAE", f"{st.session_state['metrics']['MAE']:.3f}")
    m3.metric("RMSE", f"{st.session_state['metrics']['RMSE']:.3f}")

    st.subheader("Actual vs Predicted")
    results_df = pd.DataFrame({
        "Actual": st.session_state["y_test"].values,
        "Predicted": st.session_state["y_pred"]
    })
    st.dataframe(results_df.head(20), use_container_width=True)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(st.session_state["y_test"], st.session_state["y_pred"])
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title("Actual vs Predicted")
    st.pyplot(fig)

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

    cols_ui = st.columns(len(used_features))
    for i, feature in enumerate(used_features):
        default_val = float(X[feature].median())
        input_data[feature] = cols_ui[i].number_input(
            feature,
            value=default_val
        )

    if st.button("Predict"):
        input_df = pd.DataFrame([input_data])
        pred = st.session_state["pipeline"].predict(input_df)[0]
        st.success(f"Predicted value: {pred:.2f}")
