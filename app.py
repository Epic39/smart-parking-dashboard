
import os
import io
import zipfile
import urllib.request
import warnings
from datetime import datetime

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Smart Parking Analytics", page_icon="🅿️", layout="wide")

st.title("🅿️ Smart Parking Availability Analytics & Prediction")
st.caption("Historical Birmingham parking data • PySpark ETL and analytics • Random Forest prediction")

DATA_URLS = [
    "https://archive.ics.uci.edu/static/public/482/parking+birmingham.zip",
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00482/parking_birmingham.zip",
]

@st.cache_data(show_spinner=False)
def load_raw_csv():
    last_error = None
    for url in DATA_URLS:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as response:
                payload = response.read()
            with zipfile.ZipFile(io.BytesIO(payload)) as archive:
                members = [n for n in archive.namelist() if n.lower().endswith(".csv")]
                if not members:
                    raise ValueError("No CSV file found in dataset ZIP.")
                member = next((n for n in members if os.path.basename(n).lower() == "dataset.csv"), members[0])
                with archive.open(member) as f:
                    return pd.read_csv(f)
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"Could not download UCI dataset: {last_error}")

def normalize_columns(df):
    df = df.copy()
    df.columns = [str(c).strip().replace("\ufeff", "") for c in df.columns]
    lookup = {c.lower().replace(" ", "").replace("_", ""): c for c in df.columns}
    aliases = {
        "SystemCode": ["systemcode", "carpark", "carparkid", "parkingid", "site"],
        "Capacity": ["capacity", "spaces", "totalcapacity"],
        "Occupancy": ["occupancy", "occupied", "occupiedspaces", "vehicles"],
        "LastUpdated": ["lastupdated", "timestamp", "datetime", "date", "time"],
    }
    rename = {}
    for target, names in aliases.items():
        for name in names:
            if name in lookup:
                rename[lookup[name]] = target
                break
    df = df.rename(columns=rename)
    required = ["SystemCode", "Capacity", "Occupancy", "LastUpdated"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Unrecognized dataset columns: {list(df.columns)}; missing {missing}")

    df["SystemCode"] = df["SystemCode"].astype(str).str.strip()
    df["Capacity"] = pd.to_numeric(df["Capacity"], errors="coerce")
    df["Occupancy"] = pd.to_numeric(df["Occupancy"], errors="coerce")
    df["LastUpdated"] = pd.to_datetime(df["LastUpdated"], errors="coerce", utc=True)
    df = df.dropna(subset=required)
    df = df[(df["Capacity"] > 0) & (df["Occupancy"] >= 0)]
    df["Occupancy"] = np.minimum(df["Occupancy"], df["Capacity"])
    df = df.drop_duplicates(["SystemCode", "LastUpdated"])
    df = df.sort_values(["SystemCode", "LastUpdated"]).reset_index(drop=True)
    if df.empty:
        raise ValueError("No valid data remains after cleaning.")
    return df

@st.cache_resource(show_spinner="Starting PySpark...")
def get_spark():
    from pyspark.sql import SparkSession
    return (
        SparkSession.builder
        .appName("SmartParkingAnalytics")
        .master("local[2]")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.shuffle.partitions", "4")
        .getOrCreate()
    )

def run_spark_etl(df):
    from pyspark.sql import functions as F
    from pyspark.sql.window import Window

    spark = get_spark()
    rows = [
        (str(r.SystemCode), float(r.Capacity), float(r.Occupancy),
         r.LastUpdated.to_pydatetime().replace(tzinfo=None))
        for r in df[["SystemCode", "Capacity", "Occupancy", "LastUpdated"]].itertuples(index=False)
    ]
    sdf = spark.createDataFrame(
        rows,
        schema="SystemCode string, Capacity double, Occupancy double, LastUpdated timestamp"
    )
    sdf = (
        sdf.dropna(subset=["SystemCode", "Capacity", "Occupancy", "LastUpdated"])
        .filter((F.col("Capacity") > 0) & (F.col("Occupancy") >= 0))
        .dropDuplicates(["SystemCode", "LastUpdated"])
        .withColumn("Occupancy", F.least(F.col("Occupancy"), F.col("Capacity")))
        .withColumn("FreeSpaces", F.greatest(F.col("Capacity") - F.col("Occupancy"), F.lit(0.0)))
        .withColumn("OccupancyRate", F.col("Occupancy") / F.col("Capacity"))
        .withColumn("Hour", F.hour("LastUpdated"))
        .withColumn("DayOfWeek", F.date_format("LastUpdated", "EEEE"))
        .withColumn("Date", F.to_date("LastUpdated"))
        .withColumn("DayNumber", F.dayofweek("LastUpdated"))
    )
    window = Window.partitionBy("SystemCode").orderBy("LastUpdated")
    sdf = sdf.withColumn("PreviousOccupancy", F.lag("Occupancy").over(window))
    sdf = sdf.withColumn("NextOccupancy", F.lead("Occupancy").over(window))

    hotspots = sdf.groupBy("SystemCode").agg(
        F.avg("OccupancyRate").alias("AverageOccupancyRate"),
        F.avg("Occupancy").alias("AverageOccupied"),
        F.avg("FreeSpaces").alias("AverageFreeSpaces"),
        F.max("Capacity").alias("Capacity"),
        F.count("*").alias("Records"),
    ).orderBy(F.desc("AverageOccupancyRate"))

    hourly = sdf.groupBy("Hour").agg(
        F.avg("OccupancyRate").alias("AverageOccupancyRate"),
        F.avg("Occupancy").alias("AverageOccupied"),
        F.avg("FreeSpaces").alias("AverageFreeSpaces"),
    ).orderBy("Hour")

    weekday = sdf.groupBy("DayNumber", "DayOfWeek").agg(
        F.avg("OccupancyRate").alias("AverageOccupancyRate")
    ).orderBy("DayNumber")

    daily = sdf.groupBy("Date").agg(
        F.avg("OccupancyRate").alias("AverageOccupancyRate")
    ).orderBy("Date")

    return {
        "spark": spark,
        "sdf": sdf,
        "rows": sdf.count(),
        "carparks": sdf.select("SystemCode").distinct().count(),
        "hotspots": hotspots.toPandas(),
        "hourly": hourly.toPandas(),
        "weekday": weekday.toPandas(),
        "daily": daily.toPandas(),
    }

def train_model(df):
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error, r2_score

    data = df.copy()
    data["Hour"] = data["LastUpdated"].dt.hour
    data["DayNumber"] = data["LastUpdated"].dt.dayofweek
    data["Month"] = data["LastUpdated"].dt.month
    data["OccupancyRate"] = data["Occupancy"] / data["Capacity"]
    data["PreviousOccupancyRate"] = data.groupby("SystemCode")["OccupancyRate"].shift(1)
    data["PreviousOccupancyRate"] = data["PreviousOccupancyRate"].fillna(data["OccupancyRate"]).clip(0, 1)

    features = ["Hour", "DayNumber", "Month", "Capacity", "PreviousOccupancyRate"]
    X = data[features].replace([np.inf, -np.inf], np.nan).fillna(0)
    y = data["OccupancyRate"].clip(0, 1)
    split = int(len(data) * 0.8)
    if len(data) < 20 or split >= len(data):
        raise ValueError("Not enough data to train and test the model.")

    model = RandomForestRegressor(
        n_estimators=120, max_depth=12, min_samples_leaf=2,
        random_state=42, n_jobs=-1
    )
    model.fit(X.iloc[:split], y.iloc[:split])
    pred = np.clip(model.predict(X.iloc[split:]), 0, 1)
    baseline = np.clip(X.iloc[split:]["PreviousOccupancyRate"].to_numpy(), 0, 1)
    actual = y.iloc[split:]
    return {
        "model": model,
        "features": features,
        "mae": float(mean_absolute_error(actual, pred)),
        "baseline_mae": float(mean_absolute_error(actual, baseline)),
        "r2": float(r2_score(actual, pred)) if len(actual) > 1 else float("nan"),
        "test_count": len(actual),
    }

with st.sidebar:
    st.header("Project details")
    st.write("**Dataset:** UCI Parking Birmingham")
    st.write("**Processing:** PySpark DataFrames and aggregations")
    st.write("**Prediction:** Random Forest regression")
    st.caption("Historical data only — no live sensor feed is connected.")

try:
    with st.spinner("Loading and cleaning historical data..."):
        raw_df = load_raw_csv()
        clean_df = normalize_columns(raw_df)

    with st.spinner("Running PySpark ETL and aggregations..."):
        results = run_spark_etl(clean_df)

    with st.spinner("Training and evaluating Random Forest..."):
        model_results = train_model(clean_df)

    overall_rate = (clean_df["Occupancy"] / clean_df["Capacity"]).mean()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Clean records", f"{results['rows']:,}")
    c2.metric("Car parks", f"{results['carparks']:,}")
    c3.metric("Average occupancy", f"{overall_rate:.1%}")
    c4.metric("Date range", f"{clean_df['LastUpdated'].min():%Y-%m-%d} to {clean_df['LastUpdated'].max():%Y-%m-%d}")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Overview", "Hotspots", "Peak Hours", "Prediction", "BDA Pipeline"
    ])

    with tab1:
        st.subheader("Historical occupancy trend")
        daily = results["daily"].copy()
        if not daily.empty:
            daily["Date"] = pd.to_datetime(daily["Date"])
            fig = px.line(daily, x="Date", y="AverageOccupancyRate",
                          title="Average parking occupancy over time")
            fig.update_yaxes(tickformat=".0%", rangemode="tozero")
            st.plotly_chart(fig, use_container_width=True)

        hot = results["hotspots"].copy()
        hot["AverageOccupancyPercent"] = hot["AverageOccupancyRate"] * 100
        fig = px.bar(hot.sort_values("AverageOccupancyRate", ascending=False),
                     x="SystemCode", y="AverageOccupancyPercent",
                     title="Average occupancy by car park")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("Busiest parking hotspots")
        hot = results["hotspots"].copy()
        hot["Average occupancy (%)"] = (hot["AverageOccupancyRate"] * 100).round(2)
        hot["Average occupied spaces"] = hot["AverageOccupied"].round(1)
        hot["Average free spaces"] = hot["AverageFreeSpaces"].round(1)
        cols = ["SystemCode", "Capacity", "Average occupancy (%)",
                "Average occupied spaces", "Average free spaces", "Records"]
        st.dataframe(hot[cols].head(15), use_container_width=True, hide_index=True)
        top = hot.head(10).sort_values("AverageOccupancyRate")
        fig = px.bar(top, x="AverageOccupancyRate", y="SystemCode", orientation="h",
                     title="Top 10 car parks by average occupancy")
        fig.update_xaxes(tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("Peak-hour analysis")
        hourly = results["hourly"].copy()
        hourly["Average occupancy (%)"] = hourly["AverageOccupancyRate"] * 100
        fig = px.bar(hourly, x="Hour", y="Average occupancy (%)",
                     title="Average occupancy by hour of day")
        fig.update_xaxes(dtick=1)
        st.plotly_chart(fig, use_container_width=True)
        peak = hourly.loc[hourly["AverageOccupancyRate"].idxmax()]
        st.info(f"Highest average occupancy occurs around {int(peak['Hour']):02d}:00 "
                f"({peak['Average occupancy (%)']:.1f}%), based on historical data.")

        weekday = results["weekday"].copy()
        weekday["Average occupancy (%)"] = weekday["AverageOccupancyRate"] * 100
        fig = px.bar(weekday, x="DayOfWeek", y="Average occupancy (%)",
                     title="Average occupancy by weekday")
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.subheader("Estimate occupancy for a selected car park and time")
        carparks = sorted(clean_df["SystemCode"].unique().tolist())
        selected = st.selectbox("Car park", carparks)
        col1, col2 = st.columns(2)
        with col1:
            prediction_date = st.date_input("Prediction date", value=datetime.now().date())
        with col2:
            hour = st.slider("Hour of day", 0, 23, 12)

        park = clean_df[clean_df["SystemCode"] == selected]
        capacity = float(park["Capacity"].median())
        previous_rate = float((park["Occupancy"] / park["Capacity"]).iloc[-1])
        sample = pd.DataFrame([{
            "Hour": hour,
            "DayNumber": prediction_date.weekday(),
            "Month": prediction_date.month,
            "Capacity": capacity,
            "PreviousOccupancyRate": previous_rate,
        }])[model_results["features"]]

        rate = float(np.clip(model_results["model"].predict(sample)[0], 0, 1))
        occupied = int(round(rate * capacity))
        free = max(0, int(round(capacity - occupied)))
        p1, p2, p3 = st.columns(3)
        p1.metric("Predicted occupancy", f"{rate:.1%}")
        p2.metric("Estimated occupied spaces", f"{occupied:,}")
        p3.metric("Estimated free spaces", f"{free:,}")

        st.caption("Model estimate from historical patterns, not live parking availability.")
        st.subheader("Model evaluation on chronological holdout")
        m1, m2, m3 = st.columns(3)
        m1.metric("Random Forest MAE", f"{model_results['mae']:.4f}")
        m2.metric("Baseline MAE", f"{model_results['baseline_mae']:.4f}")
        m3.metric("R² score", f"{model_results['r2']:.4f}" if np.isfinite(model_results["r2"]) else "N/A")
        st.write(f"Test records: {model_results['test_count']:,}")
        if model_results["mae"] < model_results["baseline_mae"]:
            st.success("Random Forest beat the persistence baseline on this test split.")
        else:
            st.warning("Random Forest did not beat the baseline on this split. Report this honestly.")

    with tab5:
        st.subheader("Big Data Analytics workflow")
        st.markdown("""
**1. Ingestion:** Load the UCI Parking Birmingham CSV dataset.  
**2. Cleaning with PySpark:** Remove invalid rows and duplicate car-park/time records.  
**3. Transformation with PySpark:** Derive free spaces, occupancy rate, hour, weekday and date.  
**4. Aggregation with PySpark:** Calculate average occupancy by car park, hour, weekday and date.  
**5. Feature engineering:** Use time features and previous occupancy.  
**6. Machine learning:** Train Random Forest regression and compare with a simple baseline.  
**7. Visualization:** Display trends, hotspots, peak periods and predictions using Streamlit and Plotly.

**Limitations:** This is historical data, not a live sensor feed. The dataset contains occupancy snapshots, so individual vehicle parking duration is not calculated.
""")
        st.write(f"Spark version: {results['spark'].version}")
        st.write(f"Rows processed by Spark: {results['rows']:,}")
        st.write(f"Distinct car parks: {results['carparks']:,}")
        with st.expander("Show Spark execution plan"):
            st.code(results["sdf"]._jdf.queryExecution().executedPlan().toString())

    st.caption("Educational project. Predictions are estimates, not guaranteed real-time availability.")

except Exception as exc:
    st.error("The dashboard could not finish loading.")
    st.exception(exc)
    st.info("Check requirements.txt and the Streamlit Cloud deployment logs for the underlying error.")
