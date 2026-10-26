
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from ucimlrepo import fetch_ucirepo
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

st.set_page_config(page_title="Smart Parking Analytics", page_icon="🅿️", layout="wide")

st.title("🅿️ Smart Parking Analytics")
st.caption("Historical parking data • Birmingham, UK • Machine Learning")

@st.cache_data
def load_data():
    dataset = fetch_ucirepo(id=482)
    df = pd.concat(
        [x for x in [dataset.data.features, dataset.data.targets] if x is not None],
        axis=1
    )

    df.columns = [
        str(c).strip().replace(" ", "").replace("-", "")
        for c in df.columns
    ]

    rename = {}
    for c in df.columns:
        name = c.lower().replace("_", "")
        if name in ["systemcodenumber", "carpark", "carparkid", "facilityid"]:
            rename[c] = "facility_id"
        elif name == "capacity":
            rename[c] = "capacity"
        elif name == "occupancy":
            rename[c] = "occupancy"
        elif name in ["lastupdated", "timestamp", "datetime", "time"]:
            rename[c] = "timestamp"

    df = df.rename(columns=rename)
    df = df[["facility_id", "capacity", "occupancy", "timestamp"]].copy()

    df["facility_id"] = df["facility_id"].astype(str)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["capacity"] = pd.to_numeric(df["capacity"], errors="coerce")
    df["occupancy"] = pd.to_numeric(df["occupancy"], errors="coerce")

    df = df.dropna().drop_duplicates(["facility_id", "timestamp"])
    df = df[(df["capacity"] > 0) & (df["occupancy"] >= 0)].copy()
    df["occupancy"] = np.minimum(df["occupancy"], df["capacity"])
    df["free_spaces"] = df["capacity"] - df["occupancy"]
    df["occupancy_rate"] = df["occupancy"] / df["capacity"]
    df["hour"] = df["timestamp"].dt.hour
    df["day"] = df["timestamp"].dt.dayofweek

    return df.sort_values(["facility_id", "timestamp"])


try:
    df = load_data()
except Exception as e:
    st.error("Could not load the UCI parking dataset. Check the error below.")
    st.exception(e)
    st.stop()

st.sidebar.header("Dashboard Controls")
facility = st.sidebar.selectbox(
    "Choose a car park", sorted(df["facility_id"].unique())
)
site = df[df["facility_id"] == facility].copy()
latest = site.iloc[-1]

a, b, c, d = st.columns(4)
a.metric("Car parks", df["facility_id"].nunique())
b.metric("Total observations", f"{len(df):,}")
c.metric("Parking capacity", int(latest["capacity"]))
d.metric("Estimated free spaces in latest record", int(latest["free_spaces"]))

st.caption(f"Selected car park: {facility} | Last recorded: {latest['timestamp']}")

st.subheader("1. Historical Occupancy")
fig = px.line(
    site, x="timestamp", y="occupancy_rate",
    title="Occupancy over time",
    labels={"timestamp": "Time", "occupancy_rate": "Occupancy"}
)
fig.update_yaxes(tickformat=".0%", range=[0, 1])
st.plotly_chart(fig, use_container_width=True)

left, right = st.columns(2)

with left:
    hourly = df.groupby("hour", as_index=False)["occupancy_rate"].mean()
    fig = px.line(
        hourly, x="hour", y="occupancy_rate", markers=True,
        title="Average Occupancy by Hour"
    )
    fig.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)

with right:
    summary = df.groupby("facility_id", as_index=False).agg(
        average_occupancy=("occupancy_rate", "mean")
    )
    summary = summary.sort_values("average_occupancy", ascending=False)
    fig = px.bar(
        summary, x="facility_id", y="average_occupancy",
        title="Occupancy Comparison by Car Park"
    )
    fig.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)

st.subheader("2. Machine Learning Prediction")
st.write(
    "A Random Forest model estimates occupancy at the next recorded "
    "observation. This is not a live prediction."
)

groups = []
for _, group in df.groupby("facility_id"):
    group = group.sort_values("timestamp").copy()
    group["previous_1"] = group["occupancy_rate"].shift(1)
    group["previous_2"] = group["occupancy_rate"].shift(2)
    group["target"] = group["occupancy_rate"].shift(-1)
    group["hour_sin"] = np.sin(2 * np.pi * group["hour"] / 24)
    group["hour_cos"] = np.cos(2 * np.pi * group["hour"] / 24)
    groups.append(group)

model_df = pd.concat(groups).dropna(
    subset=["previous_1", "previous_2", "target"]
).sort_values("timestamp")

features = [
    "occupancy_rate", "previous_1", "previous_2",
    "hour_sin", "hour_cos", "capacity"
]

cutoff = model_df["timestamp"].quantile(0.8)
train = model_df[model_df["timestamp"] <= cutoff]
test = model_df[model_df["timestamp"] > cutoff]

X_train = pd.get_dummies(
    train[features + ["facility_id"]], columns=["facility_id"]
)
X_test = pd.get_dummies(
    test[features + ["facility_id"]], columns=["facility_id"]
).reindex(columns=X_train.columns, fill_value=0)

model = RandomForestRegressor(
    n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
)
model.fit(X_train, train["target"])
predictions = np.clip(model.predict(X_test), 0, 1)

mae = mean_absolute_error(test["target"], predictions)
baseline_mae = mean_absolute_error(
    test["target"], test["occupancy_rate"]
)
score = r2_score(test["target"], predictions)

x, y, z = st.columns(3)
x.metric("Model MAE", f"{mae:.4f}")
y.metric("Baseline MAE", f"{baseline_mae:.4f}")
z.metric("Model R²", f"{score:.3f}")

last_row = model_df[model_df["facility_id"] == facility].tail(1)
if not last_row.empty:
    X_last = pd.get_dummies(
        last_row[features + ["facility_id"]], columns=["facility_id"]
    ).reindex(columns=X_train.columns, fill_value=0)

    predicted_rate = float(np.clip(model.predict(X_last)[0], 0, 1))
    capacity = int(last_row.iloc[0]["capacity"])
    predicted_free = round(capacity * (1 - predicted_rate))

    st.success(
        f"Predicted next-observation occupancy: {predicted_rate:.1%} "
        f"| Estimated free spaces: {predicted_free} of {capacity}"
    )

st.divider()
st.markdown(
    "Data source: [UCI Parking Birmingham]"
    "(https://archive.ics.uci.edu/dataset/482/parking+birmingham)"
)
st.warning(
    "This project uses historical 2016 data, not live parking sensors. "
    "Predictions are estimates, not guaranteed availability."
)
