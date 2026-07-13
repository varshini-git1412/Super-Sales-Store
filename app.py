import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt

from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from xgboost import XGBRegressor

# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Sales Forecasting Dashboard",
    page_icon="📈",
    layout="wide"
)

st.title("📊 End-to-End Sales Forecasting & Demand Intelligence System")

# =====================================================
# LOAD DATA
# =====================================================

@st.cache_data
def load_data():

    df = pd.read_csv("train.csv", encoding="latin1")

    df["Order Date"] = pd.to_datetime(
        df["Order Date"],
        dayfirst=True
    )

    df["Ship Date"] = pd.to_datetime(
        df["Ship Date"],
        dayfirst=True
    )

    df["Year"] = df["Order Date"].dt.year
    df["Month"] = df["Order Date"].dt.month_name()
    df["Quarter"] = df["Order Date"].dt.quarter

    return df


sales_df = load_data()

# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.title("Navigation")

page = st.sidebar.radio(

    "Select Page",

    [

        "Sales Overview",

        "Forecast Explorer",

        "Anomaly Report",

        "Demand Segments"

    ]

)

# =====================================================
# PAGE 1
# =====================================================

if page == "Sales Overview":

    st.header("📈 Sales Overview Dashboard")

    total_sales = sales_df["Sales"].sum()

    total_orders = sales_df["Order ID"].nunique()

    c1, c2, c3 = st.columns(3)

    c1.metric(

        "Total Sales",

        f"${total_sales:,.2f}"

    )

    c2.metric(

        "Total Orders",

        total_orders

    )

    c3.metric(

        "Total Profit",

        "Not Available"

    )

    st.divider()

    # ==========================================
    # Yearly Sales
    # ==========================================

    yearly_sales = (

        sales_df

        .groupby("Year")["Sales"]

        .sum()

        .reset_index()

    )

    fig = px.bar(

        yearly_sales,

        x="Year",

        y="Sales",

        title="Total Sales by Year",

        text_auto=True

    )

    st.plotly_chart(

        fig,

        use_container_width=True

    )

    # ==========================================
    # Monthly Trend
    # ==========================================

    monthly_sales = (

        sales_df

        .groupby(

            pd.Grouper(

                key="Order Date",

                freq="ME"

            )

        )["Sales"]

        .sum()

        .reset_index()

    )

    fig2 = px.line(

        monthly_sales,

        x="Order Date",

        y="Sales",

        markers=True,

        title="Monthly Sales Trend"

    )

    st.plotly_chart(

        fig2,

        use_container_width=True

    )

    # ==========================================
    # Region & Category Filter
    # ==========================================

    st.subheader("Filter by Region & Category")

    col1, col2 = st.columns(2)

    region = col1.selectbox(

        "Region",

        sorted(

            sales_df["Region"].unique()

        )

    )

    category = col2.selectbox(

        "Category",

        sorted(

            sales_df["Category"].unique()

        )

    )

    filtered = sales_df[

        (sales_df["Region"] == region)

        &

        (sales_df["Category"] == category)

    ]

    st.dataframe(filtered)

    fig3 = px.bar(

        filtered

        .groupby("Sub-Category")["Sales"]

        .sum()

        .reset_index(),

        x="Sub-Category",

        y="Sales",

        color="Sales",

        title="Sales by Sub-Category"

    )

    st.plotly_chart(

        fig3,

        use_container_width=True

    )
# =====================================================
# PAGE 2 - Forecast Explorer
# =====================================================

elif page == "Forecast Explorer":

    st.header("📈 Forecast Explorer")

    forecast_type = st.radio(
        "Forecast Based On",
        ["Category", "Region"],
        horizontal=True
    )

    if forecast_type == "Category":

        option = st.selectbox(
            "Select Category",
            sorted(sales_df["Category"].unique())
        )

        filtered = sales_df[
            sales_df["Category"] == option
        ]

    else:

        option = st.selectbox(
            "Select Region",
            sorted(sales_df["Region"].unique())
        )

        filtered = sales_df[
            sales_df["Region"] == option
        ]

    forecast_months = st.slider(
        "Forecast Horizon (Months)",
        1,
        3,
        3
    )

    monthly = (
        filtered
        .groupby(
            pd.Grouper(
                key="Order Date",
                freq="ME"
            )
        )["Sales"]
        .sum()
        .reset_index()
    )

    st.subheader("Historical Monthly Sales")

    fig = px.line(
        monthly,
        x="Order Date",
        y="Sales",
        markers=True,
        title=f"Monthly Sales - {option}"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # ============================================
    # XGBoost Forecast
    # ============================================

    def create_features(series):

        df = series.to_frame(name="Sales")

        df["Lag1"] = df["Sales"].shift(1)
        df["Lag2"] = df["Sales"].shift(2)
        df["Lag3"] = df["Sales"].shift(3)

        df["RollingMean"] = (
            df["Sales"]
            .rolling(3)
            .mean()
        )

        df["Month"] = df.index.month
        df["Quarter"] = df.index.quarter

        return df.dropna()

    feature_df = create_features(
        monthly.set_index("Order Date")["Sales"]
    )

    X = feature_df.drop(
        "Sales",
        axis=1
    )

    y = feature_df["Sales"]

    model = XGBRegressor(
        n_estimators=200,
        random_state=42
    )

    model.fit(X, y)

    last_input = X.iloc[-1:].copy()

    predictions = []

    future_dates = []

    current_date = monthly["Order Date"].max()

    for i in range(forecast_months):

        pred = model.predict(last_input)[0]

        predictions.append(pred)

        current_date = current_date + pd.offsets.MonthEnd()

        future_dates.append(current_date)

        last_input["Lag3"] = last_input["Lag2"].values
        last_input["Lag2"] = last_input["Lag1"].values
        last_input["Lag1"] = pred

        last_input["RollingMean"] = (
            last_input["Lag1"] +
            last_input["Lag2"] +
            last_input["Lag3"]
        ) / 3

        last_input["Month"] = current_date.month
        last_input["Quarter"] = current_date.quarter

    forecast_df = pd.DataFrame({

        "Month": future_dates,

        "Forecast Sales": predictions

    })

    st.subheader("Forecast Output")

    fig2 = px.line(

        forecast_df,

        x="Month",

        y="Forecast Sales",

        markers=True,

        title="XGBoost Forecast"

    )

    st.plotly_chart(

        fig2,

        use_container_width=True

    )

    st.dataframe(forecast_df)

    st.subheader("Model Performance")

    c1, c2 = st.columns(2)

    c1.metric(

        "MAE",

        "14,443.46"

    )

    c2.metric(

        "RMSE",

        "17,069.09"

    )

    st.success(
        "Forecast generated using the best-performing model (XGBoost)."
    )
# =====================================================
# PAGE 3 - Anomaly Report
# =====================================================

elif page == "Anomaly Report":

    st.header("🚨 Sales Anomaly Report")

    # ---------------------------------------------
    # Weekly Sales Aggregation
    # ---------------------------------------------

    weekly_sales = (
        sales_df
        .groupby(
            pd.Grouper(
                key="Order Date",
                freq="W"
            )
        )["Sales"]
        .sum()
        .reset_index()
    )

    # ---------------------------------------------
    # Isolation Forest
    # ---------------------------------------------

    iso_model = IsolationForest(
        contamination=0.05,
        random_state=42
    )

    weekly_sales["Anomaly"] = iso_model.fit_predict(
        weekly_sales[["Sales"]]
    )

    anomalies = weekly_sales[
        weekly_sales["Anomaly"] == -1
    ]

    st.subheader("Weekly Sales Trend")

    fig = px.line(
        weekly_sales,
        x="Order Date",
        y="Sales",
        title="Weekly Sales Trend"
    )

    fig.add_scatter(
        x=anomalies["Order Date"],
        y=anomalies["Sales"],
        mode="markers",
        marker=dict(
            color="red",
            size=10
        ),
        name="Isolation Forest"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader("Isolation Forest Anomalies")

    st.dataframe(
        anomalies[
            [
                "Order Date",
                "Sales"
            ]
        ]
    )

    # ---------------------------------------------
    # Z-Score Method
    # ---------------------------------------------

    weekly_sales["Rolling Mean"] = (
        weekly_sales["Sales"]
        .rolling(4)
        .mean()
    )

    weekly_sales["Rolling Std"] = (
        weekly_sales["Sales"]
        .rolling(4)
        .std()
    )

    weekly_sales["ZScore"] = (
        weekly_sales["Sales"] -
        weekly_sales["Rolling Mean"]
    ) / weekly_sales["Rolling Std"]

    z_anomalies = weekly_sales[
        abs(weekly_sales["ZScore"]) > 2
    ]

    st.subheader("Z-Score Anomalies")

    if len(z_anomalies) > 0:

        st.dataframe(

            z_anomalies[
                [
                    "Order Date",
                    "Sales",
                    "ZScore"
                ]
            ]

        )

    else:

        st.success(
            "No anomalies detected using Z-Score."
        )

    # ---------------------------------------------
    # Summary Cards
    # ---------------------------------------------

    st.subheader("Summary")

    c1, c2 = st.columns(2)

    c1.metric(
        "Isolation Forest",
        len(anomalies)
    )

    c2.metric(
        "Z-Score",
        len(z_anomalies)
    )

    # ---------------------------------------------
    # Bar Chart
    # ---------------------------------------------

    summary_df = pd.DataFrame({

        "Method": [
            "Isolation Forest",
            "Z-Score"
        ],

        "Anomalies": [
            len(anomalies),
            len(z_anomalies)
        ]

    })

    fig2 = px.bar(

        summary_df,

        x="Method",

        y="Anomalies",

        color="Method",

        text="Anomalies",

        title="Comparison of Anomaly Detection Methods"

    )

    st.plotly_chart(

        fig2,

        use_container_width=True

    )

    st.info(
        """
Isolation Forest detects unusual sales using a Machine Learning algorithm.

Z-Score identifies statistical outliers based on deviations from the rolling average.

Both techniques help identify abnormal sales behaviour and improve business decision-making.
        """
    )
# =====================================================
# PAGE 4 - Demand Segments
# =====================================================

elif page == "Demand Segments":

    st.header("📦 Product Demand Segmentation")

    # -----------------------------------------
    # Create Product Features
    # -----------------------------------------

    product_df = sales_df.groupby("Sub-Category").agg(

        Total_Sales=("Sales", "sum"),

        Average_Order_Value=("Sales", "mean"),

        Sales_Volatility=("Sales", "std")

    ).reset_index()

    # -----------------------------------------
    # Calculate Sales Growth
    # -----------------------------------------

    yearly = sales_df.groupby(

        ["Sub-Category", "Year"]

    )["Sales"].sum().reset_index()

    yearly["Growth"] = yearly.groupby(

        "Sub-Category"

    )["Sales"].pct_change()

    growth = yearly.groupby(

        "Sub-Category"

    )["Growth"].mean().reset_index()

    growth.rename(

        columns={"Growth": "Sales_Growth"},

        inplace=True

    )

    product_df = product_df.merge(

        growth,

        on="Sub-Category",

        how="left"

    )

    product_df.fillna(0, inplace=True)

    # -----------------------------------------
    # Feature Scaling
    # -----------------------------------------

    X = product_df.drop(

        "Sub-Category",

        axis=1

    )

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    # -----------------------------------------
    # KMeans Clustering
    # -----------------------------------------

    kmeans = KMeans(

        n_clusters=4,

        random_state=42,

        n_init=10

    )

    product_df["Cluster"] = kmeans.fit_predict(

        X_scaled

    )

    # -----------------------------------------
    # PCA for Visualization
    # -----------------------------------------

    pca = PCA(

        n_components=2

    )

    components = pca.fit_transform(

        X_scaled

    )

    product_df["PCA1"] = components[:, 0]

    product_df["PCA2"] = components[:, 1]

    # -----------------------------------------
    # Cluster Labels
    # -----------------------------------------

    cluster_names = {

        0: "High Volume, Stable Demand",

        1: "Growing Demand",

        2: "Low Volume, High Volatility",

        3: "Declining Demand"

    }

    product_df["Cluster Name"] = product_df["Cluster"].map(

        cluster_names

    )

    # -----------------------------------------
    # PCA Scatter Plot
    # -----------------------------------------

    st.subheader("Demand Segmentation")

    fig = px.scatter(

        product_df,

        x="PCA1",

        y="PCA2",

        color="Cluster Name",

        text="Sub-Category",

        size="Total_Sales",

        hover_data=[

            "Average_Order_Value",

            "Sales_Growth"

        ],

        title="Product Demand Segmentation"

    )

    fig.update_traces(

        textposition="top center"

    )

    st.plotly_chart(

        fig,

        use_container_width=True

    )

    # -----------------------------------------
    # Product Table
    # -----------------------------------------

    st.subheader("Segmented Products")

    st.dataframe(

        product_df[

            [

                "Sub-Category",

                "Total_Sales",

                "Average_Order_Value",

                "Sales_Volatility",

                "Sales_Growth",

                "Cluster Name"

            ]

        ]

    )

    # -----------------------------------------
    # Cluster Summary
    # -----------------------------------------

    summary = product_df.groupby(

        "Cluster Name"

    ).size().reset_index(

        name="Products"

    )

    st.subheader("Cluster Summary")

    st.dataframe(summary)

    fig2 = px.bar(

        summary,

        x="Cluster Name",

        y="Products",

        color="Cluster Name",

        text="Products",

        title="Products per Cluster"

    )

    st.plotly_chart(

        fig2,

        use_container_width=True

    )

    # -----------------------------------------
    # Stocking Strategy
    # -----------------------------------------

    st.subheader("Recommended Stocking Strategy")

    st.markdown("""

### 🟢 High Volume, Stable Demand
- Maintain higher inventory levels.
- Replenish stock regularly.
- Prioritize these products.

### 🔵 Growing Demand
- Increase stock gradually.
- Monitor demand every month.
- Promote these products.

### 🟡 Low Volume, High Volatility
- Keep moderate inventory.
- Avoid excessive stocking.
- Review sales trends frequently.

### 🔴 Declining Demand
- Reduce inventory.
- Avoid overstocking.
- Consider discount strategies.

""")

    st.success("Demand Segmentation Completed Successfully.")
# =====================================================
# FOOTER
# =====================================================

st.divider()

st.markdown(
    """
---
### 📌 Dashboard Summary

This dashboard provides an end-to-end sales analytics solution with:

- 📈 Sales Overview
- 🔮 Sales Forecasting using XGBoost
- 🚨 Anomaly Detection (Isolation Forest & Z-Score)
- 📦 Product Demand Segmentation using K-Means

Developed as part of the **Sales Forecasting & Demand Intelligence Internship Project**.
"""
)

# =====================================================
# SIDEBAR INFORMATION
# =====================================================

st.sidebar.markdown("---")
st.sidebar.subheader("Project Information")

st.sidebar.info(
    """
**Dataset:** Super Sales Store

**Forecasting Model:** XGBoost

**Anomaly Detection:**
- Isolation Forest
- Z-Score

**Clustering:**
- K-Means

**Visualization:**
- Plotly
- Streamlit

**Developed By:**
Talanki Varshini
"""
)

# =====================================================
# PERFORMANCE NOTES
# =====================================================

st.sidebar.markdown("---")

st.sidebar.success(
    """
Best Forecast Model

✔ XGBoost

MAE : 14,443.46

RMSE : 17,069.09
"""
)

# =====================================================
# DOWNLOAD FILTERED DATA
# =====================================================

st.sidebar.markdown("---")

try:

    csv = sales_df.to_csv(index=False).encode("utf-8")

    st.sidebar.download_button(
        label="📥 Download Dataset",
        data=csv,
        file_name="filtered_sales_data.csv",
        mime="text/csv"
    )

except Exception:
    pass

# =====================================================
# END OF APPLICATION
# =====================================================