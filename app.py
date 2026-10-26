
import io
import zipfile

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score


# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="Smart Parking Analytics",
    page_icon="🅿️",
    layout="wide",
)

st.title("🅿️ Smart Parking Availability Analytics")
st.markdown(
    "A data analytics dashboard combining PySpark processing, "
    "historical parking analysis, and machine-learning predictions."
)
st.caption("Birmingham, UK | Historical dataset from 2016")


# =========================================================
# 2. CREATE PYSPARK SESSION
# =========================================================
@st.cache_resource
def get_spark():
    return (
        SparkSession.builder
        .appName("SmartParkingAnalytics")
        .master("local[2]")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.driver.memory", "1g")
        .getOrCreate()
    )


# =========================================================
# 3. DOWNLOAD AND READ THE UCI ZIP DATASET
# =========================================================
@st.cache_data(ttl=3600, show_spinner="Downloading parking dataset...")
def download_parking_csv():
    urls = [
        "https://archive.ics.uci.edu/ml/"
        "machine-learning-databases/00482/dataset.zip",
        "https://archive.ics.uci.edu/static/public/482/"
        "parking+birmingham.zip",
    ]

    errors = []

    for url in urls:
        try:
            response = requests.get(
                url,
                timeout=60,
                headers={"User-Agent": "SmartParkingAnalytics/1.0"},
            )
            response.raise_for_status()

            with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
                csv_files = [
                    name for name in archive.namelist()
                    if name.lower().endswith(".csv")
                    and not name.startswith("__MACOSX")
                ]

                if not csv_files:
                    raise ValueError("No CSV file found in the ZIP archive.")

                # Prefer the main dataset file when present.
                preferred = [
                    name for name in csv_files
                    if name.lower().endswith("dataset.csv")
                ]
                chosen = preferred[0] if preferred else csv_files[0]

                with archive.open(chosen) as file:
                    raw = pd.read_csv(file, dtype=str)

            def normalise(name):
                return "".join(
                    ch.lower()
                    for ch in str(name)
                    if ch.isalnum()
                )

            rename = {}
            for col in raw.columns:
                key = normalise(col)

                if key in ("systemcodenumber", "carparkid", "facilityid"):
                    rename[col] = "facility_id"
                elif key == "capacity":
                    rename[col] = "capacity"
                elif key == "occupancy":
                    rename[col] = "occupancy"
                elif key in ("lastupdated", "timestamp", "datetime"):
                    rename[col] = "timestamp"

            raw = raw.rename(columns=rename)

            required = [
                "facility_id", "capacity", "occupancy", "timestamp"
            ]
            missing = [col for col in required if col not in raw.columns]

            if missing:
                raise ValueError(
                    f"Expected columns missing: {missing}. "
                    f"Found columns: {list(raw.columns)}"
                )

            return raw[required].copy()

        except Exception as exc:
            errors.append(f"{url}: {exc}")

    raise RuntimeError(
        "Unable to download/read the official UCI dataset.\n"
        + "\n".join(errors)
    )


# =========================================================
# 4. PYSPARK DATA CLEANING AND FEATURE ENGINEERING
# =========================================================
@st.cache_data(ttl=3600, show_spinner="Processing data with PySpark...")
def process_parking_data(raw):
    spark = get_spark()

    # Create a Spark DataFrame from the downloaded records.
    spark_df = spark.createDataFrame(raw)

    # Standardise types and remove invalid rows.
    spark_df = (
        spark_df
        .withColumn("facility_id", F.trim(F.col("facility_id")))
        .withColumn("capacity", F.col("capacity").cast("double"))
        .withColumn("occupancy", F.col("occupancy").cast("double"))
        .withColumn("timestamp", F.to_timestamp(F.col("timestamp")))
        .filter(
            F.col("facility_id").isNotNull()
            & (F.col("facility_id") != "")
            & F.col("capacity").isNotNull()
            & F.col("occupancy").isNotNull()
            & F.col("timestamp").isNotNull()
            & (F.col("capacity") > 0)
            & (F.col("occupancy") >= 0)
        )
        .dropDuplicates(["facility_id", "timestamp"])
        .withColumn(
            "occupancy",
            F.least(F.col("occupancy"), F.col("capacity")),
        )
        .withColumn(
            "free_spaces",
            F.col("capacity") - F.col("occupancy"),
        )
        .withColumn(
            "occupancy_rate",
            F.col("occupancy") / F.col("capacity"),
        )
        .withColumn("hour", F.hour("timestamp"))
        .withColumn("day_number", F.dayofweek("timestamp") - 2)
        .withColumn(
            "day_number",
            F.when(F.col("day_number") < 0, 6)
            .otherwise(F.col("day_number")),
        )
        .withColumn("day_name", F.date_format("timestamp", "EEEE"))
    )

    # PySpark aggregations used by the dashboard.
    facility_summary = (
        spark_df.groupBy("facility_id")
        .agg(
            F.avg("occupancy_rate").alias("average_occupancy"),
            F.max("capacity").alias("capacity"),
            F.count("*").alias("observations"),
        )
        .orderBy("facility_id")
        .toPandas()
    )

    hourly_summary = (
        spark_df.groupBy("hour")
        .agg(F.avg("occupancy_rate").alias("average_occupancy"))
        .orderBy("hour")
        .toPandas()
    )

    daily_summary = (
        spark_df.groupBy("day_number", "day_name")
        .agg(F.avg("occupancy_rate").alias("average_occupancy"))
        .orderBy("day_number")
        .toPandas()
    )

    # Use a Spark window to create lag and next-observation target features.
    window = Window.partitionBy("facility_id").orderBy("timestamp")

    feature_df = (
        spark_df
        .withColumn(
            "previous_1",
            F.lag("occupancy_rate", 1).over(window),
        )
        .withColumn(
            "previous_2",
            F.lag("occupancy_rate", 2).over(window),
        )
        .withColumn(
            "target",
            F.lead("occupancy_rate", 1).over(window),
        )
        .withColumn(
            "hour_sin",
            F.sin(2 * np.pi * F.col("hour") / 24),
        )
        .withColumn(
            "hour_cos",
            F.cos(2 * np.pi * F.col("hour") / 24),
        )
    )

    # Collect only the fields required for the interactive dashboard/model.
    clean_data = spark_df.orderBy(
        "facility_id", "timestamp"
    ).toPandas()

    model_data = (
        feature_df
        .dropna(
            subset=[
                "previous_1", "previous_2", "target",
                "hour_sin", "hour_cos",
            ]
        )
        .orderBy("timestamp")
        .toPandas()
    )

    return clean_data, facility_summary, hourly_summary, daily_summary, model_data


try:
    raw_data = download_parking_csv()
    df, facility_summary, hourly_summary, daily_summary, model_data = (
        process_parking_data(raw_data)
    )
except Exception as exc:
    st.error("The dataset or PySpark pipeline failed.")
    st.exception(exc)
    st.info(
        "Check the Streamlit deployment logs. PySpark requires a "
        "compatible Java runtime and installed dependencies."
    )
    st.stop()


# =========================================================
# 5. SIDEBAR
# =========================================================
st.sidebar.title("Dashboard Controls")

facilities = sorted(df["facility_id"].astype(str).unique().tolist())

selected_facility = st.sidebar.selectbox(
    "Choose a car park",
    facilities,
)

site = (
    df[df["facility_id"].astype(str) == selected_facility]
    .sort_values("timestamp")
    .copy()
)

latest = site.iloc[-1]


# =========================================================
# 6. KEY PERFORMANCE INDICATORS
# =========================================================
st.subheader("Parking Overview")

k1, k2, k3, k4 = st.columns(4)

k1.metric("Car Parks", f"{df['facility_id'].nunique():,}")
k2.metric("Clean Records", f"{len(df):,}")
k3.metric("Selected Capacity", f"{int(latest['capacity']):,}")
k4.metric("Latest Free Spaces", f"{int(latest['free_spaces']):,}")

st.caption(
    f"Selected car park: {selected_facility} | "
    f"Last recorded observation: {latest['timestamp']}"
)


# =========================================================
# 7. LATEST OCCUPANCY GAUGE
# =========================================================
st.subheader("Latest Recorded Occupancy")

left, right = st.columns([1, 1])

with left:
    rate = float(latest["occupancy_rate"]) * 100

    gauge = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=rate,
            number={"suffix": "%"},
            title={"text": "Occupancy Rate"},
            gauge={
                "axis": {"range": [0, 100]},
                "steps": [
                    {"range": [0, 50], "color": "#d1fae5"},
                    {"range": [50, 80], "color": "#fef3c7"},
                    {"range": [80, 100], "color": "#fee2e2"},
                ],
                "bar": {"color": "#2563eb"},
            },
        )
    )
    gauge.update_layout(height=300, margin=dict(t=50, b=20))
    st.plotly_chart(gauge, use_container_width=True)

with right:
    st.metric("Occupied Spaces", f"{int(latest['occupancy']):,}")
    st.metric("Available Spaces", f"{int(latest['free_spaces']):,}")
    st.metric("Total Capacity", f"{int(latest['capacity']):,}")
    st.progress(float(np.clip(latest["occupancy_rate"], 0, 1)))


# =========================================================
# 8. HISTORICAL OCCUPANCY
# =========================================================
st.subheader("1. Historical Occupancy")

history = px.line(
    site,
    x="timestamp",
    y="occupancy_rate",
    title=f"Occupancy Trend — {selected_facility}",
    labels={
        "timestamp": "Date and Time",
        "occupancy_rate": "Occupancy Rate",
    },
    hover_data=["occupancy", "free_spaces", "capacity"],
)
history.update_yaxes(tickformat=".0%", range=[0, 1])
history.update_layout(hovermode="x unified")
st.plotly_chart(history, use_container_width=True)


# =========================================================
# 9. HOURLY AND FACILITY ANALYSIS
# =========================================================
col1, col2 = st.columns(2)

with col1:
    st.subheader("2. Average Occupancy by Hour")

    hourly = hourly_summary.copy()
    hourly["hour"] = pd.to_numeric(hourly["hour"], errors="coerce")
    hourly = hourly.dropna(subset=["hour"]).sort_values("hour")
    hourly["hour"] = hourly["hour"].astype(int)

    hourly_chart = px.bar(
        hourly,
        x="hour",
        y="average_occupancy",
        title="Average Occupancy by Hour of Day",
        labels={
            "hour": "Hour of Day",
            "average_occupancy": "Average Occupancy",
        },
    )
    hourly_chart.update_xaxes(
        tickmode="linear", dtick=1, range=[-0.5, 23.5]
    )
    hourly_chart.update_yaxes(tickformat=".0%", range=[0, 1])
    st.plotly_chart(hourly_chart, use_container_width=True)

with col2:
    st.subheader("3. Car Park Comparison")

    comparison = facility_summary.sort_values(
        "average_occupancy", ascending=False
    )

    facility_chart = px.bar(
        comparison,
        x="facility_id",
        y="average_occupancy",
        title="Average Occupancy by Car Park",
        labels={
            "facility_id": "Car Park",
            "average_occupancy": "Average Occupancy",
        },
        hover_data=["capacity", "observations"],
    )
    facility_chart.update_xaxes(tickangle=-45)
    facility_chart.update_yaxes(tickformat=".0%", range=[0, 1])
    st.plotly_chart(facility_chart, use_container_width=True)


# =========================================================
# 10. DAY-OF-WEEK ANALYSIS
# =========================================================
st.subheader("4. Occupancy by Day of Week")

day_order = [
    "Monday", "Tuesday", "Wednesday", "Thursday",
    "Friday", "Saturday", "Sunday",
]

days = daily_summary.copy()
days["day_number"] = pd.to_numeric(
    days["day_number"], errors="coerce"
)
days = days.dropna(subset=["day_number"])
days["day_number"] = days["day_number"].astype(int)
days = days.sort_values("day_number")

day_chart = px.bar(
    days,
    x="day_name",
    y="average_occupancy",
    category_orders={"day_name": day_order},
    title="Average Occupancy Across Days",
    labels={
        "day_name": "Day",
        "average_occupancy": "Average Occupancy",
    },
)
day_chart.update_yaxes(tickformat=".0%", range=[0, 1])
st.plotly_chart(day_chart, use_container_width=True)


# =========================================================
# 11. MACHINE LEARNING: RANDOM FOREST
# =========================================================
st.subheader("5. Machine Learning Prediction")

st.write(
    "PySpark creates the historical lag features. A Random Forest model "
    "predicts the next recorded occupancy observation."
)

features = [
    "occupancy_rate",
    "previous_1",
    "previous_2",
    "hour_sin",
    "hour_cos",
    "capacity",
]

if len(model_data) < 100:
    st.warning("There are not enough observations to evaluate the model.")
    st.stop()

split_index = int(len(model_data) * 0.8)
train = model_data.iloc[:split_index].copy()
test = model_data.iloc[split_index:].copy()

if train.empty or test.empty:
    st.error("The chronological train/test split is empty.")
    st.stop()

X_train = pd.get_dummies(
    train[features + ["facility_id"]],
    columns=["facility_id"],
)
X_test = pd.get_dummies(
    test[features + ["facility_id"]],
    columns=["facility_id"],
).reindex(columns=X_train.columns, fill_value=0)

y_train = train["target"].astype(float)
y_test = test["target"].astype(float)

model = RandomForestRegressor(
    n_estimators=100,
    max_depth=12,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1,
)

with st.spinner("Training Random Forest..."):
    model.fit(X_train, y_train)
    predictions = np.clip(model.predict(X_test), 0, 1)

model_mae = mean_absolute_error(y_test, predictions)
baseline_mae = mean_absolute_error(
    y_test, test["occupancy_rate"].astype(float)
)
model_r2 = r2_score(y_test, predictions)

m1, m2, m3 = st.columns(3)
m1.metric("Model MAE", f"{model_mae:.4f}")
m2.metric("Baseline MAE", f"{baseline_mae:.4f}")
m3.metric("Model R²", f"{model_r2:.3f}")

if baseline_mae > 0:
    improvement = (baseline_mae - model_mae) / baseline_mae * 100
    st.write(f"MAE change versus baseline: {improvement:.1f}%")

st.caption(
    "The evaluation uses a chronological split. MAE is the average "
    "absolute error in occupancy-rate units. R² may be negative."
)


# =========================================================
# 12. PREDICTION FOR SELECTED CAR PARK
# =========================================================
st.markdown("### Prediction for Selected Car Park")

selected_model_rows = (
    model_data[model_data["facility_id"].astype(str) == selected_facility]
    .sort_values("timestamp")
)

if not selected_model_rows.empty:
    current = selected_model_rows.iloc[[-1]].copy()

    X_current = pd.get_dummies(
        current[features + ["facility_id"]],
        columns=["facility_id"],
    ).reindex(columns=X_train.columns, fill_value=0)

    predicted_rate = float(np.clip(model.predict(X_current)[0], 0, 1))
    capacity = int(current.iloc[0]["capacity"])
    predicted_occupied = int(round(predicted_rate * capacity))
    predicted_free = max(0, capacity - predicted_occupied)

    p1, p2, p3 = st.columns(3)
    p1.metric("Predicted Occupancy", f"{predicted_rate:.1%}")
    p2.metric("Estimated Occupied", f"{predicted_occupied:,}")
    p3.metric("Estimated Free Spaces", f"{predicted_free:,}")

    st.progress(predicted_rate)
else:
    st.warning(
        "Not enough historical records for a prediction for this car park."
    )

st.info(
    "This predicts the next recorded observation, not a guaranteed "
    "time such as exactly one hour later. The dataset is historical, "
    "not a live parking sensor feed."
)


# =========================================================
# 13. DATA PREVIEW AND CSV DOWNLOAD
# =========================================================
with st.expander("View processed parking data"):
    st.dataframe(
        site[
            [
                "timestamp",
                "facility_id",
                "capacity",
                "occupancy",
                "free_spaces",
                "occupancy_rate",
            ]
        ].sort_values("timestamp", ascending=False),
        use_container_width=True,
    )

    st.download_button(
        "Download selected car park CSV",
        data=site.to_csv(index=False).encode("utf-8"),
        file_name="smart_parking_data.csv",
        mime="text/csv",
    )


# =========================================================
# 14. PYSPARK PROCESSING SUMMARY
# =========================================================
with st.expander("PySpark processing details"):
    st.write("This application uses PySpark for:")
    st.markdown(
        """
        - Type conversion and validation
        - Removing duplicate car-park observations
        - Calculating occupied and free spaces
        - Creating hour and day features
        - Aggregating occupancy by hour and car park
        - Window functions (`lag` and `lead`) for prediction features
        """
    )
    st.write(f"Processed Spark records: {len(df):,}")
    st.write(f"Distinct car parks: {df['facility_id'].nunique():,}")


# =========================================================
# 15. SOURCE AND LIMITATIONS
# =========================================================
st.divider()

st.markdown(
    """
    **Dataset:** [UCI Machine Learning Repository — Parking Birmingham]
    (https://archive.ics.uci.edu/dataset/482/parking%2Bbirmingham)

    **Citation:** Stolfi, D. (2017). *Parking Birmingham*.
    UCI Machine Learning Repository. DOI: 10.24432/C51K5Z.
    """
)

st.warning(
    "The data contains historical observations from 2016. The app does "
    "not connect to live parking sensors. Prediction results are estimates."
)
