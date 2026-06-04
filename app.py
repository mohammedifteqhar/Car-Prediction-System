# app.py — Final production-friendly Streamlit app for model_tabular (FIXED VERSION)
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import re
from tensorflow.keras.models import load_model

# ---------------------------
# PAGE CONFIG
# ---------------------------
st.set_page_config(page_title="Car Price Predictor (Tabular DL)", page_icon="🚗", layout="wide")
st.title("🚗 Car Resale Price Predictor")
st.write("Deep Learning Tabular Model with Embeddings")

# ---------------------------
# PATHS
# ---------------------------
MODEL_DIR = "model_tabular"
MODEL_FILE = os.path.join(MODEL_DIR, "dl_tabular_model_best.keras")

# ---------------------------
# LOAD MODEL & ARTIFACTS
# ---------------------------
if not os.path.exists(MODEL_DIR):
    st.error(f"Model folder not found: {MODEL_DIR}")
    st.stop()

try:
    model = load_model(MODEL_FILE, compile=False)
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

try:
    with open(os.path.join(MODEL_DIR, "encoders.pkl"), "rb") as f:
        encoders = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "metadata.pkl"), "rb") as f:
        metadata = pickle.load(f)
except Exception as e:
    st.error(f"Failed to load encoders/scaler/metadata: {e}")
    st.stop()

cat_cols = metadata["cat_cols"]
num_cols = metadata["num_cols"]
log_target = metadata.get("log_target", True)

# ---------------------------
# FIX: REMOVE binary columns from categorical encoders
# ---------------------------
BINARY_COLS = ["Leather interior", "Accident_history"]

for b in BINARY_COLS:
    if b in cat_cols:
        cat_cols.remove(b)
    if b in encoders:
        del encoders[b]


# ---------------------------
# HELPERS
# ---------------------------
def safe_col_name(col):
    return col.replace(" ", "_").replace("-", "_").replace("/", "_")

def str_to_num_clean(x):
    if pd.isna(x): return 0.0
    s = str(x).strip()
    s = re.sub(r"[^\d\.]", "", s)
    return float(s) if s != "" else 0.0

def clean_input_df(df):
    for col in ["Levy", "Engine volume", "Mileage"]:
        if col in df.columns:
            df[col] = df[col].apply(str_to_num_clean)

    if "Prod. year" in df.columns:
        df["Prod. year"] = pd.to_numeric(df["Prod. year"], errors="coerce")
        df["Prod. year"] = df["Prod. year"].fillna(2020).astype(int)

    for col in ["Cylinders","Airbags","Brand_value_index",
                "Safety_features","Comfort_features","Modification_penalty"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    return df


def map_category_value(col, val):
    """Correct mapping — binary columns bypass encoders"""
    if col in BINARY_COLS:
        return int(val)  # already 0/1

    le = encoders.get(col)
    if le is None:
        return 0

    val_str = str(val)

    if val_str in le.classes_:
        return int(np.where(le.classes_ == val_str)[0][0])

    if "___NA___" in le.classes_:
        return int(np.where(le.classes_ == "___NA___")[0][0])

    return 0


def prepare_model_inputs_from_df(df, debug_mode=False):
    inputs = {}

    # categorical
    for col in cat_cols:
        safe = safe_col_name(col)
        val = df.iloc[0].get(col, "___NA___")
        idx = map_category_value(col, val)
        inputs[f"in_{safe}"] = np.array([[idx]], dtype="int32")

    # numeric
    numeric_df = df[num_cols].astype(float).copy()

    if debug_mode:
        st.write("DEBUG — raw numeric input:")
        st.write({n: numeric_df.iloc[0][i] for i, n in enumerate(num_cols)})

    scaled = scaler.transform(numeric_df).astype("float32")
    inputs["numeric_input"] = scaled

    if debug_mode:
        st.write("DEBUG — scaled numeric_input:")
        st.write(scaled.tolist())

    return inputs


def predict_price(user_dict, debug=False):
    df = pd.DataFrame([user_dict])
    df = clean_input_df(df)

    # force all categorical fields to string
    for c in cat_cols:
        df[c] = df[c].astype(str)

    # binary columns already 0/1 → do not touch
    inputs = prepare_model_inputs_from_df(df, debug_mode=debug)

    if debug:
        st.write("DEBUG — categorical mapping:")
        for c in cat_cols:
            st.write(c, "→", inputs[f"in_{safe_col_name(c)}"].tolist())

    pred_log = float(model.predict(inputs, verbose=0)[0][0])
    price = float(np.expm1(pred_log)) if log_target else float(pred_log)

    return price


# ---------------------------
# UI
# ---------------------------
st.sidebar.markdown("## Debug")
debug_mode = st.sidebar.checkbox("Show Debug Inputs", value=False)

left, right = st.columns(2)

with left:
    manufacturer = st.selectbox("Manufacturer", encoders["Manufacturer"].classes_.tolist())
    model_name = st.selectbox("Model", encoders["Model"].classes_.tolist())
    category = st.selectbox("Category", encoders["Category"].classes_.tolist())
    fuel = st.selectbox("Fuel type", encoders["Fuel type"].classes_.tolist())
    gear = st.selectbox("Gear box type", encoders["Gear box type"].classes_.tolist())
    drive = st.selectbox("Drive wheels", encoders["Drive wheels"].classes_.tolist())

with right:
    doors = st.selectbox("Doors", encoders["Doors"].classes_.tolist())
    wheel = st.selectbox("Wheel Position", encoders["Wheel"].classes_.tolist())
    color = st.selectbox("Color", encoders["Color"].classes_.tolist())

levy = st.text_input("Levy", value="500")
prod_year = st.number_input("Prod. year", 1990, 2025, 2018)
engine = st.text_input("Engine volume", value="1.8")
mileage = st.text_input("Mileage", value="45000")
leather = st.selectbox("Leather interior", ["Yes", "No"])
acc = st.selectbox("Accident History", ["No", "Yes"])
cyl = st.number_input("Cylinders", 2, 16, 4)
airbags = st.number_input("Airbags", 0, 20, 2)
brand = st.slider("Brand Value Index", 0, 10, 5)
safety = st.slider("Safety Features", 0, 10, 5)
comfort = st.slider("Comfort Features", 0, 10, 5)
mod_penalty = st.slider("Modification Penalty", 0, 10, 2)

# ---------------------------
# PACK DATA
# ---------------------------
user_data = {
    "Manufacturer": manufacturer,
    "Model": model_name,
    "Category": category,
    "Fuel type": fuel,
    "Gear box type": gear,
    "Drive wheels": drive,
    "Doors": doors,
    "Wheel": wheel,
    "Color": color,
    "Levy": levy,
    "Prod. year": prod_year,
    "Engine volume": engine,
    "Mileage": mileage,
    "Leather interior": 1 if leather == "Yes" else 0,
    "Accident_history": 1 if acc == "Yes" else 0,
    "Cylinders": cyl,
    "Airbags": airbags,
    "Brand_value_index": brand,
    "Safety_features": safety,
    "Comfort_features": comfort,
    "Modification_penalty": mod_penalty,
}

# ---------------------------
# BUTTON
# ---------------------------
st.markdown("---")
if st.button("🔍 Predict Price", use_container_width=True):
    price = predict_price(user_data, debug=debug_mode)
    st.success(f"💰 Estimated Price: ₹ {price:,.2f}")

