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

st.set_page_config(
    page_title="Smart Parking Analytics",
    page_icon="🅿️",
    layout="wide",
)

st.title("🅿️ Smart Parking Availability Analytics & Prediction")
st.caption("Historical Birmingham parking data • PySpark ETL and analytics • Random Forest prediction")

DATA_URLS = [
    "https://archive.ics.uci.edu/static/public/482/parking+birmingham.zip",
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00482/parking_birmingham.zip",
]
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)
LOCAL_CSV = os.path.join(DATA_DIR, "dataset.csv")


@st.cache_data(show_spinner=False)
def load_raw_csv():
    """Load the UCI CSV; cache successful results."""
    if os.path.exists(LOCAL_CSV):
        return pd.read_csv(LOCAL_CSV)

    last_error = None
    for url in DATA_URLS:
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "Mozilla/5.0 SmartParkingStudentProject"},
            )
            with urllib.request.urlopen(req, timeout=30) as response:
                payload = response.read()

            # UCI publishes this dataset as a ZIP archive.
            if payload[:2] == b"PK":
                with zipfile.ZipFile(io.BytesIO(payload)) as archive:
                    csv_members = [
                        name for name in archive.namelist()
                        if name.lower().endswith(".csv")
                    ]
                    if not csv_members:
                        raise ValueError("The downloaded ZIP contains no CSV file.")
                    # Prefer dataset.csv where available.
                    member = next(
                        (name for name in csv_members
                         if os.path.basename(name).lower() == "dataset.csv"),
                        csv_members[0],
                    )
                    with archive.open(member) as f:
                        df = pd.read_csv(f)
            else:
                df = pd.read_csv(io.BytesIO(payload))

            # Streamlit's app filesystem may be ephemeral; saving is just a cache.
            try:
                df.to_csv(LOCAL_CSV, index=False)
            except Exception:
                pass
            return df
        except Exception as exc:
            last_error = exc

    raise RuntimeError(
        "Could not download the UCI dataset. Check the app's internet access or "
        "upload the dataset CSV as data/dataset.csv. Last error: " + str(last_error)
    )


def normalize_columns(df):
    """Map known variants to stable names without assuming source capitalization."""
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
