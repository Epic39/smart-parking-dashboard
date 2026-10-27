
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

st.set_page_config(
    page_title="Smart Parking Analytics",
    page_icon="🅿️",
    layout="wide"
)

st.title("🅿️ Smart Parking Availability Analytics & Prediction")
st.caption(
    "Historical Birmingham parking data • PySpark ETL and analytics • Random Forest prediction"
)

DATA_URLS = [
    "https://archive.ics.uci.edu/static/public/482/parking+birmingham.zip",
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00482/parking_birmingham.zip",
]


@st.cache_data(show_spinner=False)
def load_raw_csv():
    last_error = None

    for url in DATA_URLS:
        try:
            request = urllib.request.Request(
                url,
                headers={"User-Agent": "Mozilla/5.0"}
            )

            with urllib.request.urlopen(request, timeout=40) as response:
                content = response.read()

            if content[:2] == b"PK":
                with zipfile.ZipFile(io.BytesIO(content)) as archive:
                    csv_files = [
                        name for name in archive.namelist()
                        if name.lower().endswith(".csv")
                    ]

                    if not csv_files:
                        raise ValueError("No CSV found in downloaded ZIP.")

                    filename = next(
                        (
                            name for name in csv_files
                            if name.lower().endswith("dataset.csv")
                        ),
                        csv_files[0]
                    )

                    with archive.open(filename) as file:
                        return pd.read_csv(file)

            return pd.read_csv(io.BytesIO(content))

        except Exception as error:
            last_error = error

    raise RuntimeError(
        f"Unable to download UCI dataset. Error: {last_error}"
    )


def normalize_columns(df):
    df = df.copy()

    df.columns = [
        str(column).strip().replace("\ufeff", "")
        for column in df.columns
    ]

    # Handles SystemCodeNumber, System Code Number,
    # system_code_number and similar column formats.
    lookup = {
        "".join(char for char in column.lower() if char.isalnum()): column
        for column in df.columns
    }

    aliases = {
        "SystemCode": [
            "systemcode",
            "systemcodenumber",
            "carpark",
            "carparkid",
            "parkingid",
            "site",
            "carparkname"
        ],
        "Capacity": [
            "capacity",
            "spaces",
            "totalcapacity"
        ],
        "Occupancy": [
            "occupancy",
            "occupied",
            "occupiedspaces",
            "vehicles"
        ],
        "LastUpdated": [
            "lastupdated",
            "timestamp",
            "datetime",
            "dateandtime",
            "time"
        ]
    }

    rename = {}

    for target, candidates in aliases.items():
        for candidate in candidates:
            normalized = "".join(
                char for char in candidate.lower() if char.isalnum()
            )

            if normalized in lookup:
                rename[lookup[normalized]] = target
                break

    df = df.rename(columns=rename)

    required = [
        "SystemCode",
        "Capacity",
        "Occupancy",
        "LastUpdated"
    ]

    missing = [
        column for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Dataset columns not recognized. Found: {list(df.columns)}. "
            f"Missing: {missing}"
        )

    df["SystemCode"] = df["SystemCode"].astype(str).str.strip()
    df["Capacity"] = pd.to_numeric(df["Capacity"], errors="coerce")
    df["Occupancy"] = pd.to_numeric(df["Occupancy"], errors="coerce")

    df["LastUpdated"] = pd.to_datetime(
        df["LastUpdated"],
        errors="coerce",
        utc=True
    )

    df = df.dropna(subset=required)

    df = df[
        (df["SystemCode"] != "")
        & (df["Capacity"] > 0)
        & (df["Occupancy"] >= 0)
    ]

    df["Occupancy"] = np.minimum(
        df["Occupancy"],
        df["Capacity"]
    )

    df = df.drop_duplicates(
        subset=["SystemCode", "LastUpdated"],
        keep="last"
    )

    df = df.sort_values(
        ["SystemCode", "LastUpdated"]
    ).reset_index(drop=True)

    if df.empty:
        raise ValueError("No valid records remain after cleaning.")

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

    # Convert to basic Python records to avoid Pandas/Arrow compatibility issues.
    rows = [
        (
            str(row.SystemCode),
            float(row.Capacity),
            float(row.Occupancy),
            row.LastUpdated.to_pydatetime().replace(tzinfo=None)
        )
        for row in df[
            ["SystemCode", "Capacity", "Occupancy", "LastUpdated"]
        ].itertuples(index=False)
    ]

    spark_df = spark.createDataFrame(
        rows,
        schema=(
            "SystemCode string, Capacity double, Occupancy double, "
            "LastUpdated timestamp"
        )
    )

    # PySpark cleaning and feature engineering.
    spark_df = (
        spark_df
        .dropna(
            subset=[
                "SystemCode",
                "Capacity",
                "Occupancy",
                "LastUpdated"
            ]
        )
        .filter(
            (F.col("Capacity") > 0)
            & (F.col("Occupancy") >= 0)
        )
        .dropDuplicates(["SystemCode", "LastUpdated"])
        .withColumn(
            "Occupancy",
            F.least(F.col("Occupancy"), F.col("Capacity"))
        )
        .withColumn(
            "FreeSpaces",
            F.greatest(
                F.col("Capacity") - F.col("Occupancy"),
                F.lit(0.0)
            )
        )
        .withColumn(
            "OccupancyRate",
            F.col("Occupancy") / F.col("Capacity")
        )
        .withColumn("Hour", F.hour("LastUpdated"))
        .withColumn(
            "DayOfWeek",
            F.date_format("LastUpdated", "EEEE")
        )
        .withColumn("Date", F.to_date("LastUpdated"))
        .withColumn(
            "DayNumber",
            F.dayofweek("LastUpdated")
        )
    )

    window = (
        Window
        .partitionBy("SystemCode")
        .orderBy("LastUpdated")
    )

    spark_df = (
        spark_df
        .withColumn(
            "PreviousOccupancy",
            F.lag("Occupancy").over(window)
        )
        .withColumn(
            "NextOccupancy",
            F.lead("Occupancy").over(window)
        )
    )

    # Parking hotspot aggregation.
    hotspots = (
        spark_df
        .groupBy("SystemCode")
        .agg(
            F.avg("OccupancyRate").alias("AverageOccupancyRate"),
            F.avg("Occupancy").alias("AverageOccupied"),
            F.avg("FreeSpaces").alias("AverageFreeSpaces"),
            F.max("Capacity").alias("Capacity"),
            F.count("*").alias("Records")
        )
        .orderBy(F.desc("AverageOccupancyRate"))
    )

    # Hourly aggregation.
    hourly = (
        spark_df
        .groupBy("Hour")
        .agg(
            F.avg("OccupancyRate").alias("AverageOccupancyRate"),
            F.avg("Occupancy").alias("AverageOccupied"),
            F.avg("FreeSpaces").alias("AverageFreeSpaces")
        )
        .orderBy("Hour")
    )

    # Weekday aggregation.
    weekday = (
        spark_df
        .groupBy("DayNumber", "DayOfWeek")
        .agg(
            F.avg("OccupancyRate").alias("AverageOccupancyRate")
        )
        .orderBy("DayNumber")
    )

    # Daily aggregation.
    daily = (
        spark_df
        .groupBy("Date")
        .agg(
            F.avg("OccupancyRate").alias("AverageOccupancyRate")
        )
        .orderBy("Date")
    )

    return {
        "spark": spark,
        "sdf": spark_df,
        "rows": spark_df.count(),
        "carparks": spark_df.select("SystemCode").distinct().count(),
        "hotspots": hotspots.toPandas(),
        "hourly": hourly.toPandas(),
        "weekday": weekday.toPandas(),
        "daily": daily.toPandas()
    }


def train_model(df):
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error, r2_score

    data = df.copy()

    data["Hour"] = data["LastUpdated"].dt.hour
    data["DayNumber"] = data["LastUpdated"].dt.dayofweek
    data["Month"] = data["LastUpdated"].dt.month

    data["OccupancyRate"] = (
        data["Occupancy"] / data["Capacity"]
    ).clip(0, 1)

    data["PreviousOccupancyRate"] = (
        data.groupby("SystemCode")["OccupancyRate"].shift(1)
    )

    data["PreviousOccupancyRate"] = (
        data["PreviousOccupancyRate"]
        .fillna(data["OccupancyRate"])
        .clip(0, 1)
    )

    features = [
        "Hour",
        "DayNumber",
        "Month",
        "Capacity",
        "PreviousOccupancyRate"
    ]

    X = (
        data[features]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )

    y = data["OccupancyRate"]

    # Chronological holdout.
    split = int(len(data) * 0.8)

    if len(data) < 20 or split >= len(data):
        raise ValueError("Not enough data to train and evaluate the model.")

    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=12,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X.iloc[:split], y.iloc[:split])

    predictions = np.clip(
        model.predict(X.iloc[split:]),
        0,
        1
    )

    baseline = np.clip(
        X.iloc[split:]["PreviousOccupancyRate"].to_numpy(),
        0,
        1
    )

    actual = y.iloc[split:]

    return {
        "model": model,
        "features": features,
        "mae": float(
            mean_absolute_error(actual, predictions)
        ),
        "baseline_mae": float(
            mean_absolute_error(actual, baseline)
        ),
        "r2": float(
            r2_score(actual, predictions)
        ) if len(actual) > 1 else float("nan"),
        "test_count": len(actual)
    }


with st.sidebar:
    st.header("Project details")
    st.write("**Dataset:** UCI Parking Birmingham")
    st.write("**Processing:** PySpark DataFrames and aggregations")
    st.write("**Prediction:** Random Forest regression")
    st.caption(
        "Historical dataset. No live sensor feed is connected."
    )


try:
    with st.spinner("Loading and cleaning historical data..."):
        raw_df = load_raw_csv()
        clean_df = normalize_columns(raw_df)

    with st.spinner("Running PySpark ETL and analytics..."):
        results = run_spark_etl(clean_df)

    with st.spinner("Training and evaluating the model..."):
        model_results = train_model(clean_df)

    average_occupancy = (
        clean_df["Occupancy"] / clean_df["Capacity"]
    ).mean()

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Clean records", f"{results['rows']:,}")
    c2.metric("Car parks", f"{results['carparks']:,}")
    c3.metric("Average occupancy", f"{average_occupancy:.1%}")
    c4.metric(
        "Data range",
        f"{clean_df['LastUpdated'].min():%Y-%m-%d} to "
        f"{clean_df['LastUpdated'].max():%Y-%m-%d}"
    )

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Overview",
        "Hotspots",
        "Peak Hours",
        "Prediction",
        "BDA Pipeline"
    ])

    # TAB 1: Overview
    with tab1:
        st.subheader("Historical occupancy trend")

        daily = results["daily"].copy()

        if not daily.empty:
            daily["Date"] = pd.to_datetime(daily["Date"])

            fig = px.line(
                daily,
                x="Date",
                y="AverageOccupancyRate",
                title="Average parking occupancy over time"
            )

            fig.update_yaxes(
                tickformat=".0%",
                rangemode="tozero"
            )

            st.plotly_chart(fig, use_container_width=True)

        hot = results["hotspots"].copy()

        hot["AverageOccupancyPercent"] = (
            hot["AverageOccupancyRate"] * 100
        )

        fig = px.bar(
            hot.sort_values(
                "AverageOccupancyRate",
                ascending=False
            ),
            x="SystemCode",
            y="AverageOccupancyPercent",
            title="Average occupancy by car park"
        )

        st.plotly_chart(fig, use_container_width=True)

    # TAB 2: Hotspots
    with tab2:
        st.subheader("Busiest parking hotspots")

        hot = results["hotspots"].copy()

        hot["Average occupancy (%)"] = (
            hot["AverageOccupancyRate"] * 100
        ).round(2)

        hot["Average occupied spaces"] = (
            hot["AverageOccupied"]
        ).round(1)

        hot["Average free spaces"] = (
            hot["AverageFreeSpaces"]
        ).round(1)

        columns = [
            "SystemCode",
            "Capacity",
            "Average occupancy (%)",
            "Average occupied spaces",
            "Average free spaces",
            "Records"
        ]

        st.dataframe(
            hot[columns].head(15),
            use_container_width=True,
            hide_index=True
        )

        top = hot.head(10).sort_values(
            "AverageOccupancyRate"
        )

        fig = px.bar(
            top,
            x="AverageOccupancyRate",
            y="SystemCode",
            orientation="h",
            title="Top 10 car parks by average occupancy"
        )

        fig.update_xaxes(tickformat=".0%")

        st.plotly_chart(fig, use_container_width=True)

    # TAB 3: Peak hours
    with tab3:
        st.subheader("Peak-hour analysis")

        hourly = results["hourly"].copy()

        hourly["Average occupancy (%)"] = (
            hourly["AverageOccupancyRate"] * 100
        )

        fig = px.bar(
            hourly,
            x="Hour",
            y="Average occupancy (%)",
            title="Average occupancy by hour of day"
        )

        fig.update_xaxes(dtick=1)

        st.plotly_chart(fig, use_container_width=True)

        peak = hourly.loc[
            hourly["AverageOccupancyRate"].idxmax()
        ]

        st.info(
            f"Peak historical occupancy: "
            f"{int(peak['Hour']):02d}:00 "
            f"({peak['Average occupancy (%)']:.1f}%)."
        )

        weekday = results["weekday"].copy()

        weekday["Average occupancy (%)"] = (
            weekday["AverageOccupancyRate"] * 100
        )

        fig = px.bar(
            weekday,
            x="DayOfWeek",
            y="Average occupancy (%)",
            title="Average occupancy by weekday"
        )

        st.plotly_chart(fig, use_container_width=True)

    # TAB 4: Prediction
    with tab4:
        st.subheader("Predict occupancy for a car park and time")

        carparks = sorted(
            clean_df["SystemCode"].unique().tolist()
        )

        selected = st.selectbox(
            "Select car park",
            carparks
        )

        col1, col2 = st.columns(2)

        with col1:
            prediction_date = st.date_input(
                "Prediction date",
                value=datetime.now().date()
            )

        with col2:
            hour = st.slider(
                "Hour of day",
                min_value=0,
                max_value=23,
                value=12
            )

        park = clean_df[
            clean_df["SystemCode"] == selected
        ]

        capacity = float(
            park["Capacity"].median()
        )

        previous_rate = float(
            (
                park["Occupancy"] / park["Capacity"]
            ).iloc[-1]
        )

        sample = pd.DataFrame([{
            "Hour": hour,
            "DayNumber": prediction_date.weekday(),
            "Month": prediction_date.month,
            "Capacity": capacity,
            "PreviousOccupancyRate": previous_rate
        }])[model_results["features"]]

        rate = float(
            np.clip(
                model_results["model"].predict(sample)[0],
                0,
                1
            )
        )

        occupied = int(round(rate * capacity))

        free = max(
            0,
            int(round(capacity - occupied))
        )

        a, b, c = st.columns(3)

        a.metric(
            "Predicted occupancy",
            f"{rate:.1%}"
        )

        b.metric(
            "Estimated occupied spaces",
            f"{occupied:,}"
        )

        c.metric(
            "Estimated free spaces",
            f"{free:,}"
        )

        st.caption(
            "This is a model estimate from historical patterns, "
            "not live parking availability."
        )

        st.subheader("Model evaluation")

        a, b, c = st.columns(3)

        a.metric(
            "Random Forest MAE",
            f"{model_results['mae']:.4f}"
        )

        b.metric(
            "Baseline MAE",
            f"{model_results['baseline_mae']:.4f}"
        )

        c.metric(
            "R² score",
            f"{model_results['r2']:.4f}"
            if np.isfinite(model_results["r2"])
            else "N/A"
        )

        st.write(
            f"Test records: {model_results['test_count']:,}"
        )

        if model_results["mae"] < model_results["baseline_mae"]:
            st.success(
                "Random Forest beat the baseline on this test split."
            )
        else:
            st.warning(
                "Random Forest did not beat the baseline on this split. "
                "Report the result honestly."
            )

    # TAB 5: BDA concepts
    with tab5:
        st.subheader("Big Data Analytics workflow")

        st.markdown("""
**1. Data ingestion:** Load historical UCI Parking Birmingham CSV data.

**2. Data cleaning:** PySpark removes invalid and duplicate records.

**3. Data transformation:** Calculate free spaces, occupancy rate, hour, weekday and date.

**4. Data aggregation:** PySpark computes car park, hourly, weekday and daily averages.

**5. Feature engineering:** Use temporal features and previous occupancy.

**6. Machine learning:** Train Random Forest regression and compare it with a baseline.

**7. Visualization:** Display trends, hotspots, peak periods and estimates using Streamlit and Plotly.

### Limitations

- The source dataset is historical, not a live sensor feed.
- Occupancy snapshots do not provide individual vehicle entry/exit times.
- Individual vehicle parking duration is therefore not calculated.
- Predictions are estimates, not guaranteed availability.
""")

        st.write(
            f"Spark version: {results['spark'].version}"
        )

        st.write(
            f"Rows processed by Spark: {results['rows']:,}"
        )

        st.write(
            f"Distinct car parks: {results['carparks']:,}"
        )

        with st.expander("Show Spark execution plan"):
            st.code(
                results["sdf"]
                ._jdf
                .queryExecution()
                .executedPlan()
                .toString()
            )

    st.caption(
        "Educational project. Predicted availability is an estimate."
    )

except Exception as error:
    st.error("The dashboard could not finish loading.")
    st.exception(error)
    st.info(
        "If this error continues, check the Streamlit Cloud deployment logs."
    )
