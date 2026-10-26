
import io
import zipfile
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="Smart Parking Analytics",
    page_icon="🅿️",
    layout="wide"
)

st.title("🅿️ Smart Parking Availability Analytics")
st.markdown(
    "Explore historical parking occupancy, compare car parks, "
    "and estimate future parking availability using machine learning."
)

st.caption(
    "Dataset: Birmingham, UK | Historical observations from 2016"
)

# --------------------------------------------------
# LOAD THE OFFICIAL UCI DATASET DIRECTLY
# --------------------------------------------------
@st.cache_data(ttl=3600, show_spinner="Loading parking dataset...")
def load_data():
    urls = [
        "https://archive.ics.uci.edu/ml/"
        "machine-learning-databases/00482/dataset.zip",
        "https://archive.ics.uci.edu/static/public/482/"
        "parking+birmingham.zip",
    ]

    errors = []

    for url in urls:
        try:
            response = requests.get(url, timeout=45)
            response.raise_for_status()

            with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
                csv_files = [
                    name for name in archive.namelist()
                    if name.lower().endswith(".csv")
                ]

                if not csv_files:
                    raise ValueError("No CSV file was found inside the ZIP.")

                preferred = [
                    name for name in csv_files
                    if name.lower().endswith("dataset.csv")
                ]
                csv_name = preferred[0] if preferred else csv_files[0]

                with archive.open(csv_name) as file:
                    data = pd.read_csv(file)

            # Standardise column names to handle spaces/capitalisation.
            def normalise(name):
                return "".join(
                    char.lower()
                    for char in str(name)
                    if char.isalnum()
                )

            rename = {}
            for column in data.columns:
                key = normalise(column)

                if key == "systemcodenumber":
                    rename[column] = "facility_id"
                elif key == "capacity":
                    rename[column] = "capacity"
                elif key == "occupancy":
                    rename[column] = "occupancy"
                elif key == "lastupdated":
                    rename[column] = "timestamp"

            data = data.rename(columns=rename)

            required = [
                "facility_id", "capacity", "occupancy", "timestamp"
            ]
            missing = [c for c in required if c not in data.columns]

            if missing:
                raise ValueError(
                    f"Missing columns: {missing}. "
                    f"Columns received: {list(data.columns)}"
                )

            data = data[required].copy()
            data["facility_id"] = data["facility_id"].astype(str).str.strip()
            data["capacity"] = pd.to_numeric(
                data["capacity"], errors="coerce"
            )
            data["occupancy"] = pd.to_numeric(
                data["occupancy"], errors="coerce"
            )
            data["timestamp"] = pd.to_datetime(
                data["timestamp"], errors="coerce"
            )

            data = data.dropna(subset=required)
            data = data.drop_duplicates(
                subset=["facility_id", "timestamp"]
            )

            data = data[
                (data["capacity"] > 0)
                & (data["occupancy"] >= 0)
            ].copy()

            # Keep occupancy within the recorded car park capacity.
            data["occupancy"] = np.minimum(
                data["occupancy"], data["capacity"]
            )

            data["free_spaces"] = (
                data["capacity"] - data["occupancy"]
            )
            data["occupancy_rate"] = (
                data["occupancy"] / data["capacity"]
            )
            data["hour"] = data["timestamp"].dt.hour
            data["day_name"] = data["timestamp"].dt.day_name()
            data["day_number"] = data["timestamp"].dt.dayofweek

            data = data.sort_values(
                ["facility_id", "timestamp"]
            ).reset_index(drop=True)

            if data.empty:
                raise ValueError("The dataset has no usable records.")

            return data

        except Exception as error:
            errors.append(f"{url}: {error}")

    raise RuntimeError(
        "Both official dataset download links failed.\n\n"
        + "\n".join(errors)
    )


try:
    with st.spinner("Fetching the official parking dataset..."):
        df = load_data()
except Exception as error:
    st.error("The parking dataset could not be loaded.")
    st.markdown(
        "Check your Streamlit deployment logs and try restarting "
        "the app. The error details are shown below."
    )
    st.exception(error)
    st.stop()


# --------------------------------------------------
# SIDEBAR FILTERS
# --------------------------------------------------
st.sidebar.title("Dashboard Controls")

facilities = sorted(df["facility_id"].unique().tolist())

selected_facility = st.sidebar.selectbox(
    "Select a car park",
    facilities
)

site = (
    df[df["facility_id"] == selected_facility]
    .sort_values("timestamp")
    .copy()
)

latest = site.iloc[-1]

# --------------------------------------------------
# TOP KPI CARDS
# --------------------------------------------------
st.subheader("Parking Overview")

k1, k2, k3, k4 = st.columns(4)

k1.metric("Car Parks", f"{df['facility_id'].nunique():,}")
k2.metric("Data Records", f"{len(df):,}")
k3.metric("Selected Capacity", f"{int(latest['capacity']):,}")
k4.metric("Latest Free Spaces", f"{int(latest['free_spaces']):,}")

st.caption(
    f"Selected car park: {selected_facility} | "
    f"Last recorded observation: {latest['timestamp']}"
)

# --------------------------------------------------
# OCCUPANCY GAUGE
# --------------------------------------------------
st.subheader("Latest Recorded Occupancy")

g1, g2 = st.columns([1, 2])

with g1:
    occupancy_pct = float(latest["occupancy_rate"]) * 100

    fig_gauge = {
        "data": [{
            "type": "indicator",
            "mode": "gauge+number",
            "value": round(occupancy_pct, 1),
            "number": {"suffix": "%"},
            "title": {"text": "Occupancy"},
            "gauge": {
                "axis": {"range": [0, 100]},
                "bar": {"color": "#2563eb"},
                "steps": [
                    {"range": [0, 50], "color": "#dcfce7"},
                    {"range": [50, 80], "color": "#fef3c7"},
                    {"range": [80, 100], "color": "#fee2e2"},
                ],
            },
        }],
        "layout": {"height": 300, "margin": {"t": 60, "b": 10}},
    }

    st.plotly_chart(fig_gauge, use_container_width=True)

with g2:
    st.markdown("#### Latest recorded figures")
    st.metric("Occupied Spaces", f"{int(latest['occupancy']):,}")
    st.metric("Available Spaces", f"{int(latest['free_spaces']):,}")
    st.metric("Capacity", f"{int(latest['capacity']):,}")

    st.progress(float(np.clip(latest["occupancy_rate"], 0, 1)))

# --------------------------------------------------
# HISTORICAL OCCUPANCY CHART
# --------------------------------------------------
st.subheader("1. Historical Occupancy")

history_chart = px.line(
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

history_chart.update_yaxes(
    tickformat=".0%",
    range=[0, 1],
)
history_chart.update_layout(hovermode="x unified")

st.plotly_chart(history_chart, use_container_width=True)

# --------------------------------------------------
# TIME AND CAR PARK COMPARISONS
# --------------------------------------------------
left, right = st.columns(2)

with left:
    st.subheader("2. Average Occupancy by Hour")

    hourly = (
        df.groupby("hour", as_index=False)["occupancy_rate"]
        .mean()
    )

    hourly_chart = px.bar(
        hourly,
        x="hour",
        y="occupancy_rate",
        title="Average Occupancy by Hour of Day",
        labels={
            "hour": "Hour of Day",
            "occupancy_rate": "Average Occupancy",
        },
    )
    hourly_chart.update_yaxes(tickformat=".0%", range=[0, 1])
    st.plotly_chart(hourly_chart, use_container_width=True)

with right:
    st.subheader("3. Car Park Comparison")

    comparison = (
        df.groupby("facility_id", as_index=False)
        .agg(
            average_occupancy=("occupancy_rate", "mean"),
            capacity=("capacity", "max"),
        )
        .sort_values("average_occupancy", ascending=False)
    )

    comparison_chart = px.bar(
        comparison,
        x="facility_id",
        y="average_occupancy",
        title="Average Occupancy by Car Park",
        labels={
            "facility_id": "Car Park",
            "average_occupancy": "Average Occupancy",
        },
        hover_data=["capacity"],
    )
    comparison_chart.update_yaxes(tickformat=".0%", range=[0, 1])
    st.plotly_chart(comparison_chart, use_container_width=True)

# --------------------------------------------------
# DAY-OF-WEEK ANALYSIS
# --------------------------------------------------
st.subheader("4. Occupancy by Day of Week")

day_order = [
    "Monday", "Tuesday", "Wednesday", "Thursday",
    "Friday", "Saturday", "Sunday"
]

day_summary = (
    df.groupby("day_name", as_index=False)["occupancy_rate"]
    .mean()
)
day_summary["day_name"] = pd.Categorical(
    day_summary["day_name"],
    categories=day_order,
    ordered=True,
)
day_summary = day_summary.sort_values("day_name")

day_chart = px.bar(
    day_summary,
    x="day_name",
    y="occupancy_rate",
    labels={
        "day_name": "Day",
        "occupancy_rate": "Average Occupancy",
    },
    title="Average Occupancy Across Days",
)
day_chart.update_yaxes(tickformat=".0%", range=[0, 1])
st.plotly_chart(day_chart, use_container_width=True)

# --------------------------------------------------
# MACHINE LEARNING FEATURE ENGINEERING
# --------------------------------------------------
st.subheader("5. Machine Learning Prediction")

st.write(
    "A Random Forest model predicts the occupancy rate at the "
    "next recorded observation for each car park."
)

model_data = df.copy()

model_data["previous_1"] = (
    model_data.groupby("facility_id")["occupancy_rate"].shift(1)
)
model_data["previous_2"] = (
    model_data.groupby("facility_id")["occupancy_rate"].shift(2)
)
model_data["target"] = (
    model_data.groupby("facility_id")["occupancy_rate"].shift(-1)
)

model_data["hour_sin"] = np.sin(
    2 * np.pi * model_data["hour"] / 24
)
model_data["hour_cos"] = np.cos(
    2 * np.pi * model_data["hour"] / 24
)

feature_columns = [
    "occupancy_rate",
    "previous_1",
    "previous_2",
    "hour_sin",
    "hour_cos",
    "capacity",
]

training_data = (
    model_data.dropna(
        subset=feature_columns + ["target"]
    )
    .sort_values("timestamp")
    .copy()
)

if len(training_data) < 100:
    st.warning(
        "There are not enough records to train a reliable model."
    )
    st.stop()

# Use a chronological split: older observations for training,
# newer observations for evaluation.
split_index = int(len(training_data) * 0.8)

train = training_data.iloc[:split_index].copy()
test = training_data.iloc[split_index:].copy()

if train.empty or test.empty:
    st.error("Not enough records for a train/test split.")
    st.stop()

X_train = pd.get_dummies(
    train[feature_columns + ["facility_id"]],
    columns=["facility_id"],
)
X_test = pd.get_dummies(
    test[feature_columns + ["facility_id"]],
    columns=["facility_id"],
)
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

y_train = train["target"]
y_test = test["target"]

model = RandomForestRegressor(
    n_estimators=120,
    max_depth=12,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1,
)

with st.spinner("Training Random Forest model..."):
    model.fit(X_train, y_train)
    predictions = np.clip(model.predict(X_test), 0, 1)

mae = mean_absolute_error(y_test, predictions)
baseline_mae = mean_absolute_error(
    y_test, test["occupancy_rate"]
)
r2 = r2_score(y_test, predictions)

m1, m2, m3 = st.columns(3)
m1.metric("Model MAE", f"{mae:.4f}")
m2.metric("Persistence Baseline MAE", f"{baseline_mae:.4f}")
m3.metric("Model R²", f"{r2:.3f}")

st.caption(
    "MAE measures average absolute prediction error in occupancy-rate "
    "units. The baseline predicts that the next occupancy rate will "
    "equal the current one. R² can be negative if predictions perform "
    "worse than a constant-mean benchmark."
)

# --------------------------------------------------
# NEXT-OBSERVATION PREDICTION
# --------------------------------------------------
prediction_rows = model_data[
    model_data["facility_id"] == selected_facility
].sort_values("timestamp")

prediction_rows = prediction_rows.dropna(
    subset=feature_columns
)

if not prediction_rows.empty:
    current = prediction_rows.iloc[[-1]].copy()

    X_current = pd.get_dummies(
        current[feature_columns + ["facility_id"]],
        columns=["facility_id"],
    ).reindex(columns=X_train.columns, fill_value=0)

    predicted_rate = float(
        np.clip(model.predict(X_current)[0], 0, 1)
    )

    capacity = int(current.iloc[0]["capacity"])
    predicted_occupied = int(round(predicted_rate * capacity))
    predicted_free = max(0, capacity - predicted_occupied)

    st.markdown("### Prediction for selected car park")

    p1, p2, p3 = st.columns(3)
    p1.metric("Predicted Occupancy", f"{predicted_rate:.1%}")
    p2.metric("Estimated Occupied Spaces", predicted_occupied)
    p3.metric("Estimated Free Spaces", predicted_free)

    st.progress(predicted_rate)

    st.info(
        "This is a model estimate for the next recorded observation, "
        "not a live sensor reading or a guaranteed future availability."
    )
else:
    st.warning(
        "Not enough historical observations for this car park "
        "to produce a prediction."
    )

# --------------------------------------------------
# DATA TABLE AND DOWNLOAD
# --------------------------------------------------
with st.expander("View cleaned parking records"):
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

    csv_data = site.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download selected car park data as CSV",
        data=csv_data,
        file_name="parking_data.csv",
        mime="text/csv",
    )

# --------------------------------------------------
# DATA SOURCE AND LIMITATIONS
# --------------------------------------------------
st.divider()

st.markdown(
    """
    **Data source:** [UCI Machine Learning Repository — Parking Birmingham]
    (https://archive.ics.uci.edu/dataset/482/parking%2Bbirmingham)

    **Dataset citation:** Stolfi, D. (2017). *Parking Birmingham*.
    UCI Machine Learning Repository. DOI: 10.24432/C51K5Z.
    """
)

st.warning(
    "This dashboard uses historical Birmingham parking observations "
    "from October to December 2016. It does not connect to live parking "
    "sensors. Predictions and estimated free-space counts are not "
    "guarantees of current availability."
)
