# --- CHANGES MADE ---
# 1. Removed "Modeling" + "Explain (SHAP)" sections.
# 2. Added "Demo" section where trained models are loaded and used for prediction.
# 3. Only EDA + Demo remain.

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns

# ML
import pickle

st.set_page_config(page_title="Mental Health in the Workplace — Demo App", layout="wide")

st.title("🧠 Mental Health in the Workplace — Demo App")
st.caption("EDA + Model Demo (using pre-trained models).")

# ------------------------------
# Data loading
# ------------------------------
@st.cache_data(show_spinner=False)
def load_csv(path_or_buffer: str):
    df = pd.read_csv(path_or_buffer)
    return df

def try_default_paths():
    candidates = [
        "./mental_health_data_final_data.csv",
        "./data/mental_health_data_final_data.csv"
    ]
    for p in candidates:
        if os.path.exists(p):
            return load_csv(p), p
    return None, None

# Sidebar — data source
st.sidebar.header("📂 Data")
uploaded = st.sidebar.file_uploader("Upload CSV (optional)", type=["csv"])

if uploaded is not None:
    df = load_csv(uploaded)
    data_path = "(uploaded file)"
else:
    df, data_path = try_default_paths()
    if df is None:
        st.info("No default CSV found. Please upload your dataset.")
        st.stop()

st.sidebar.success(f"Using data: {data_path}")
raw_df = df.copy()

# ------------------------------
# Helper
# ------------------------------
def infer_categoricals(df):
    cat_cols = list(df.select_dtypes(include=["object", "category", "bool"]).columns)
    low_card_ints = [c for c in df.select_dtypes(include=["int64"]).columns if df[c].nunique() <= 10]
    for c in low_card_ints:
        if c not in cat_cols:
            cat_cols.append(c)
    return sorted(cat_cols)

def safe_plot(fig):
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)

# ------------------------------
# Navigation
# ------------------------------
section = st.sidebar.radio(
    "Navigate",
    ["Overview", "Data Explorer", "Visualizations", "Demo"]
)

# ------------------------------
# Overview
# ------------------------------
if section == "Overview":
    st.subheader("Dataset Preview")
    st.write(df.head(20))
    st.subheader("Descriptive Statistics")
    st.write(df.describe(include="all").T)

# ------------------------------
# Data Explorer
# ------------------------------
elif section == "Data Explorer":
    st.subheader("Column Overview")
    cat_cols = infer_categoricals(df)
    num_cols = [c for c in df.columns if c not in cat_cols]

    st.write("**Categorical Columns:**", cat_cols)
    st.write("**Numeric Columns:**", num_cols)

    st.subheader("Value Counts")
    sel_cat = st.multiselect("Select categorical cols", cat_cols, default=cat_cols[:3])
    for col in sel_cat:
        st.write(f"**{col}**")
        st.write(df[col].value_counts(dropna=False))

# ------------------------------
# Visualizations
# ------------------------------
elif section == "Visualizations":
    st.subheader("Quick Plots")
    cat_cols = infer_categoricals(df)
    num_cols = [c for c in df.columns if c not in cat_cols]

    plot_type = st.selectbox("Plot type", ["Histogram", "Boxplot", "Countplot", "Correlation Heatmap"])

    if plot_type == "Histogram":
        col = st.selectbox("Numeric column", num_cols)
        fig, ax = plt.subplots()
        ax.hist(df[col].dropna(), bins=20)
        ax.set_title(f"Histogram — {col}")
        safe_plot(fig)

    elif plot_type == "Boxplot":
        ycol = st.selectbox("Y (numeric)", num_cols)
        xcol = st.selectbox("X (categorical)", [None] + cat_cols)
        fig, ax = plt.subplots()
        if xcol:
            sns.boxplot(data=df, x=xcol, y=ycol, ax=ax)
        else:
            sns.boxplot(y=df[ycol], ax=ax)
        safe_plot(fig)

    elif plot_type == "Countplot":
        col = st.selectbox("Categorical column", cat_cols)
        fig, ax = plt.subplots()
        sns.countplot(x=df[col], order=df[col].value_counts().index, ax=ax)
        plt.xticks(rotation=30)
        safe_plot(fig)

    elif plot_type == "Correlation Heatmap":
        corr = df.select_dtypes(include=[np.number]).corr()
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(corr, cmap="coolwarm", ax=ax)
        safe_plot(fig)

# ------------------------------
# Demo Section — Use Pretrained Model
# ------------------------------
elif section == "Demo":
    st.subheader("🎯 Model Demo")

    # Load pretrained model (place your .pkl file in same dir)
    model_file = "./trained_model.pkl"
    if not os.path.exists(model_file):
        st.warning("Pre-trained model not found. Save as 'trained_model.pkl' first.")
        st.stop()

    with open(model_file, "rb") as f:
        model = pickle.load(f)

    # Collect inputs dynamically (demo assumes typical features)
    st.write("Enter feature values:")

    sleep = st.slider("Sleep Hours", 3, 10, 7)
    work = st.slider("Work Hours", 4, 15, 9)
    pa = st.slider("Physical Activity Hours", 0, 5, 2)
    sm = st.slider("Social Media Usage (hrs)", 0, 10, 3)
    diet = st.selectbox("Diet Quality", ["Poor", "Average", "Good"])

    # Make prediction
    input_df = pd.DataFrame([{
        "Sleep_Hours": sleep,
        "Work_Hours": work,
        "Physical_Activity_Hours": pa,
        "Social_Media_Usage": sm,
        "Diet_Quality": diet
    }])

    if st.button("Predict Stress Level"):
        pred = model.predict(input_df)[0]
        st.success(f"Predicted Stress Level: **{pred}**")
