import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.arima.model import ARIMA

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except Exception:
    PROPHET_AVAILABLE = False


st.set_page_config(page_title="Supermarket Sales Forecasting", layout="wide")
st.title("Supermarket Sales Forecasting")

st.write(
    "Upload a sales dataset, optionally filter by store or product, "
    "and forecast daily sales using ARIMA and Prophet."
)


@st.cache_data
def load_csv(uploaded_file):
    return pd.read_csv(uploaded_file)


def guess_column(columns, candidates):
    lower_map = {c.lower(): c for c in columns}
    for cand in candidates:
        for col in columns:
            if cand in col.lower():
                return col
    return None


@st.cache_data
def prepare_data(df, date_col, sales_col, store_col=None, store_val=None, product_col=None, product_val=None):
    df = df.copy()

    if store_col and store_val and store_val != "All":
        df = df[df[store_col].astype(str) == str(store_val)]

    if product_col and product_val and product_val != "All":
        df = df[df[product_col].astype(str) == str(product_val)]

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df[sales_col] = pd.to_numeric(df[sales_col], errors="coerce")

    df = df.dropna(subset=[date_col, sales_col])

    if df.empty:
        raise ValueError("No valid rows remain after filtering and cleaning.")

    daily = (
        df.groupby(df[date_col].dt.date)[sales_col]
        .sum()
        .reset_index()
    )
    daily.columns = ["Date", "Sales"]

    daily["Date"] = pd.to_datetime(daily["Date"])
    daily = daily.sort_values("Date")
    daily = daily.set_index("Date").asfreq("D")

    daily["Sales"] = daily["Sales"].interpolate(method="linear")
    daily["Sales"] = daily["Sales"].bfill().ffill()

    return daily


def plot_historical(ts_df):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(ts_df.index, ts_df["Sales"], label="Historical Sales")
    ax.set_title("Daily Sales")
    ax.set_xlabel("Date")
    ax.set_ylabel("Sales")
    ax.legend()
    st.pyplot(fig)


def run_arima(ts_df, forecast_days, order):
    model = ARIMA(ts_df["Sales"], order=order)
    fitted = model.fit()
    forecast = fitted.forecast(steps=forecast_days)

    future_index = pd.date_range(
        start=ts_df.index[-1] + pd.Timedelta(days=1),
        periods=forecast_days,
        freq="D"
    )

    return pd.DataFrame({
        "Date": future_index,
        "Forecast": forecast.values
    })


def plot_arima(ts_df, forecast_df):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(ts_df.index, ts_df["Sales"], label="Historical Sales")
    ax.plot(forecast_df["Date"], forecast_df["Forecast"], label="ARIMA Forecast")
    ax.set_title("ARIMA Forecast")
    ax.set_xlabel("Date")
    ax.set_ylabel("Sales")
    ax.legend()
    st.pyplot(fig)


def run_prophet(ts_df, forecast_days):
    prophet_df = ts_df.reset_index().rename(columns={"Date": "ds", "Sales": "y"})

    model = Prophet(
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=True
    )
    model.fit(prophet_df)

    future = model.make_future_dataframe(periods=forecast_days)
    forecast = model.predict(future)
    result = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(forecast_days)

    return model, forecast, result


def plot_prophet(ts_df, forecast_result):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(ts_df.index, ts_df["Sales"], label="Historical Sales")
    ax.plot(forecast_result["ds"], forecast_result["yhat"], label="Prophet Forecast")
    ax.fill_between(
        forecast_result["ds"],
        forecast_result["yhat_lower"],
        forecast_result["yhat_upper"],
        alpha=0.2
    )
    ax.set_title("Prophet Forecast")
    ax.set_xlabel("Date")
    ax.set_ylabel("Sales")
    ax.legend()
    st.pyplot(fig)


uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is None:
    st.info("Upload a CSV file to begin.")
    st.stop()

try:
    df = load_csv(uploaded_file)
except Exception as e:
    st.error(f"Could not read CSV: {e}")
    st.stop()

if df.empty:
    st.error("Uploaded file is empty.")
    st.stop()

st.subheader("Dataset Preview")
st.dataframe(df.head(), use_container_width=True)

columns = df.columns.tolist()

# Auto-detect likely columns
default_date = guess_column(columns, ["date", "order date", "transaction date"])
default_sales = guess_column(columns, ["sales", "revenue", "amount", "total"])
default_store = guess_column(columns, ["store", "location", "branch", "outlet"])
default_product = guess_column(columns, ["product", "category", "item", "type"])

col1, col2 = st.columns(2)
with col1:
    date_col = st.selectbox(
        "Select date column",
        columns,
        index=columns.index(default_date) if default_date in columns else 0
    )
with col2:
    sales_col = st.selectbox(
        "Select sales column",
        columns,
        index=columns.index(default_sales) if default_sales in columns else min(1, len(columns)-1)
    )

col3, col4 = st.columns(2)

with col3:
    store_options = ["None"] + columns
    store_col = st.selectbox(
        "Optional store/location column",
        store_options,
        index=store_options.index(default_store) if default_store in store_options else 0
    )

with col4:
    product_options = ["None"] + columns
    product_col = st.selectbox(
        "Optional product/category column",
        product_options,
        index=product_options.index(default_product) if default_product in product_options else 0
    )

store_val = None
product_val = None

if store_col != "None":
    store_values = ["All"] + sorted(df[store_col].dropna().astype(str).unique().tolist())
    store_val = st.selectbox("Filter by store/location", store_values)

if product_col != "None":
    product_values = ["All"] + sorted(df[product_col].dropna().astype(str).unique().tolist())
    product_val = st.selectbox("Filter by product/category", product_values)

forecast_days = st.slider("Forecast horizon (days)", 7, 60, 14)

st.sidebar.header("ARIMA Settings")
p = st.sidebar.number_input("p", min_value=0, max_value=10, value=5)
d = st.sidebar.number_input("d", min_value=0, max_value=2, value=1)
q = st.sidebar.number_input("q", min_value=0, max_value=10, value=2)

if st.button("Run Forecasting"):
    try:
        ts_df = prepare_data(
            df,
            date_col=date_col,
            sales_col=sales_col,
            store_col=None if store_col == "None" else store_col,
            store_val=store_val,
            product_col=None if product_col == "None" else product_col,
            product_val=product_val
        )

        if len(ts_df) < 20:
            st.error("Please provide at least 20 daily points after filtering.")
            st.stop()

        st.subheader("Prepared Daily Sales Data")
        st.dataframe(ts_df.head(10), use_container_width=True)

        plot_historical(ts_df)

        tab1, tab2 = st.tabs(["ARIMA", "Prophet"])

        with tab1:
            st.subheader("ARIMA Forecast")
            try:
                arima_forecast = run_arima(ts_df, forecast_days, order=(p, d, q))
                st.dataframe(arima_forecast, use_container_width=True)
                plot_arima(ts_df, arima_forecast)

                st.download_button(
                    "Download ARIMA Forecast CSV",
                    data=arima_forecast.to_csv(index=False),
                    file_name="arima_forecast.csv",
                    mime="text/csv"
                )
            except Exception as e:
                st.error(f"ARIMA forecasting failed: {e}")

        with tab2:
            st.subheader("Prophet Forecast")
            if not PROPHET_AVAILABLE:
                st.warning("Prophet is not installed in this environment.")
            else:
                try:
                    prophet_model, prophet_full_forecast, prophet_result = run_prophet(ts_df, forecast_days)
                    st.dataframe(prophet_result, use_container_width=True)
                    plot_prophet(ts_df, prophet_result)

                    st.write("Prophet Components")
                    fig_components = prophet_model.plot_components(prophet_full_forecast)
                    st.pyplot(fig_components)

                    st.download_button(
                        "Download Prophet Forecast CSV",
                        data=prophet_result.to_csv(index=False),
                        file_name="prophet_forecast.csv",
                        mime="text/csv"
                    )
                except Exception as e:
                    st.error(f"Prophet forecasting failed: {e}")

    except Exception as e:
        st.error(f"Data preparation failed: {e}")
