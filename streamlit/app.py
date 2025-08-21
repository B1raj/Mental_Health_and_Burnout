

import os
import io
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import streamlit as st

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns

# ML
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Optional XGBoost
try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False

# Optional SHAP
try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False


st.set_page_config(page_title="Mental Health in the Workplace — Streamlit App", layout="wide")

st.title("🧠 Mental Health in the Workplace — Streamlit App")

st.caption("Converted from the uploaded notebook into an interactive Streamlit dashboard.")

# ------------------------------
# Data loading utilities
# ------------------------------

@st.cache_data(show_spinner=False)
def load_csv(path_or_buffer: str):
    df = pd.read_csv(path_or_buffer)
    return df

def try_default_paths():
    # Try common paths used in the notebook/previous messages
    candidates = [
        "./data/mental_health_data_final_data.csv",
        "./mental_health_data_final_data.csv",
        "./data/mental_health_data final data.csv",
        "./mental_health_data final data.csv"
    ]
    for p in candidates:
        if os.path.exists(p):
            try:
                return load_csv(p), p
            except Exception:
                pass
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
        st.info("No default CSV found. Please use the sidebar to upload your dataset.")
        st.stop()

st.sidebar.success(f"Using data: {data_path}")

# Keep a pristine copy
raw_df = df.copy()

# ------------------------------
# Helper Functions
# ------------------------------

def infer_categoricals(df):
    cat_cols = list(df.select_dtypes(include=["object", "category", "bool"]).columns)
    # Also treat low-cardinality integer columns as categorical candidates
    low_card_ints = [c for c in df.select_dtypes(include=["int64", "int32", "int16", "int8"]).columns
                     if df[c].nunique() <= 10]
    for c in low_card_ints:
        if c not in cat_cols:
            cat_cols.append(c)
    return sorted(list(dict.fromkeys(cat_cols)))

def show_missing(df):
    miss = df.isna().sum().to_frame("missing")
    miss["percent"] = (miss["missing"] / len(df) * 100).round(2)
    miss = miss.sort_values("missing", ascending=False)
    st.dataframe(miss)

def safe_plot(fig):
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)

# ------------------------------
# Navigation
# ------------------------------

section = st.sidebar.radio(
    "Navigate",
    [
        "Overview",
        "Data Explorer",
        "Feature Engineering",
        "Visualizations",
        "Modeling",
        "Explain (SHAP)"
    ]
)

# ------------------------------
# Overview
# ------------------------------
if section == "Overview":
    left, right = st.columns([2,1])
    with left:
        st.subheader("Dataset Preview")
        st.write(df.head(20))
    with right:
        st.subheader("Shape & Dtypes")
        st.metric("Rows", len(df))
        st.metric("Columns", df.shape[1])
        st.write(pd.DataFrame({"dtype": df.dtypes.astype(str)}))

    st.subheader("Descriptive Statistics (Numeric)")
    st.write(df.describe(include=[np.number]).T)

    st.subheader("Missing Values")
    show_missing(df)

# ------------------------------
# Data Explorer
# ------------------------------
elif section == "Data Explorer":
    st.subheader("Column Overview")
    cat_cols = infer_categoricals(df)
    num_cols = [c for c in df.columns if c not in cat_cols]

    c1, c2 = st.columns(2)
    with c1:
        st.write("**Categorical-like Columns**")
        st.write(cat_cols if len(cat_cols) else "_None detected_")
    with c2:
        st.write("**Numeric Columns**")
        st.write(num_cols if len(num_cols) else "_None detected_")

    st.markdown("---")
    st.subheader("Value Counts (Categorical-like)")
    sel_cat = st.multiselect("Select columns", cat_cols, default=cat_cols[: min(5, len(cat_cols))])
    for col in sel_cat:
        st.markdown(f"**{col}**")
        st.write(df[col].value_counts(dropna=False).to_frame("count"))

    st.markdown("---")
    st.subheader("Describe (Selected Columns)")
    sel_desc = st.multiselect("Select columns to describe", df.columns.tolist(), default=num_cols[: min(6, len(num_cols))])
    if sel_desc:
        st.write(df[sel_desc].describe(include="all").T)

# ------------------------------
# Feature Engineering
# ------------------------------
elif section == "Feature Engineering":
    st.subheader("Create Derived Features")

    # Example rules for Stress Level (best effort reconstruction; harmless if not applicable)
    col_opts = df.columns.tolist()

    default_cols = {
        "Sleep_Hours": "Sleep_Hours" if "Sleep_Hours" in df.columns else None,
        "Work_Hours": "Work_Hours" if "Work_Hours" in df.columns else None,
        "Physical_Activity_Hours": "Physical_Activity_Hours" if "Physical_Activity_Hours" in df.columns else None,
        "Social_Media_Usage": "Social_Media_Usage" if "Social_Media_Usage" in df.columns else None,
        "Diet_Quality": "Diet_Quality" if "Diet_Quality" in df.columns else None,
    }

    cols = {}
    for k, v in default_cols.items():
        cols[k] = st.selectbox(f"Column for {k}", options=[None] + col_opts, index=(col_opts.index(v)+1 if v in col_opts else 0))

    apply_stress = st.checkbox("Derive Stress_Level (Low/Medium/High) from the selected columns", value=True)

    if apply_stress and all(cols.values()):
        # Heuristic rule-set (non-destructive; you can change thresholds in the UI if needed)
        th_sleep = st.slider("≤ Sleep hours threshold for stress", 5, 8, 7)
        th_work  = st.slider("≥ Work hours threshold for stress", 6, 12, 9)
        th_pa    = st.slider("≤ Physical activity hours threshold", 0, 3, 1)
        th_sm    = st.slider("≥ Social media hours threshold", 0, 6, 3)
        bad_diet = st.multiselect("Diet values considered 'not good'", options=sorted(df[cols["Diet_Quality"]].dropna().astype(str).unique().tolist() if cols["Diet_Quality"] else []),
                                  default=[v for v in ["Poor", "Average"] if v in (df[cols["Diet_Quality"]].astype(str).unique() if cols["Diet_Quality"] else [])])

        cond_high = (
            (df[cols["Sleep_Hours"]] <= th_sleep) &
            (df[cols["Work_Hours"]]  >= th_work)  &
            (df[cols["Physical_Activity_Hours"]] <= th_pa) &
            (df[cols["Social_Media_Usage"]] >= th_sm) &
            (df[cols["Diet_Quality"]].astype(str).isin(bad_diet))
        )

        cond_low = (
            (df[cols["Sleep_Hours"]] > th_sleep) &
            (df[cols["Work_Hours"]]  < th_work)  &
            (df[cols["Physical_Activity_Hours"]] > th_pa) &
            (df[cols["Social_Media_Usage"]] < th_sm) &
            (~df[cols["Diet_Quality"]].astype(str).isin(bad_diet))
        )

        df["Stress_Level"] = np.select(
            [cond_high, cond_low],
            ["High", "Low"],
            default="Medium"
        )
        st.success("Created column: **Stress_Level**")
        st.write(df["Stress_Level"].value_counts())

    st.markdown("#### Save Engineered Dataset")
    if st.button("Save as `engineered_dataset.csv`"):
        out = df.copy()
        out.to_csv("engineered_dataset.csv", index=False)
        st.download_button("Download engineered_dataset.csv", data=out.to_csv(index=False), file_name="engineered_dataset.csv", mime="text/csv")

    st.markdown("#### Current Data Preview")
    st.write(df.head(20))

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
        bins = st.slider("Bins", 5, 60, 30)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(df[col].dropna(), bins=bins)
        ax.set_title(f"Histogram — {col}")
        safe_plot(fig)

    elif plot_type == "Boxplot":
        ycol = st.selectbox("Y (numeric)", num_cols)
        xcol = st.selectbox("X (categorical; optional)", [None] + cat_cols)
        fig, ax = plt.subplots(figsize=(8, 4))
        if xcol:
            sns.boxplot(data=df, x=xcol, y=ycol, ax=ax)
            ax.set_title(f"Boxplot — {ycol} by {xcol}")
        else:
            sns.boxplot(data=df[[ycol]].dropna(), y=ycol, ax=ax)
            ax.set_title(f"Boxplot — {ycol}")
        safe_plot(fig)

    elif plot_type == "Countplot":
        col = st.selectbox("Categorical column", cat_cols)
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.countplot(data=df, x=col, order=df[col].value_counts().index, ax=ax)
        ax.set_title(f"Countplot — {col}")
        ax.tick_params(axis="x", rotation=30)
        safe_plot(fig)

    elif plot_type == "Correlation Heatmap":
        num = df.select_dtypes(include=[np.number])
        if num.shape[1] < 2:
            st.warning("Need at least 2 numeric columns for correlation.")
        else:
            corr = num.corr(numeric_only=True)
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(corr, annot=False, cmap="coolwarm", ax=ax)
            ax.set_title("Correlation Heatmap")
            safe_plot(fig)

# ------------------------------
# Modeling
# ------------------------------
elif section == "Modeling":
    st.subheader("Train a Classifier")

    target = st.selectbox("Target column (classification)", options=[None] + df.columns.tolist(), index=0)
    if not target:
        st.info("Select a target column to continue.")
        st.stop()

    # Drop rows with NA in target
    data = df.dropna(subset=[target]).copy()

    # Ensure binary/multiclass classification
    y = data[target].astype(str)
    X = data.drop(columns=[target])

    cat_cols = infer_categoricals(X)
    num_cols = [c for c in X.columns if c not in cat_cols]

    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    pre = ColumnTransformer(
        transformers=[
            ("cat", enc, cat_cols),
            ("num", "passthrough", num_cols)
        ]
    )

    model_name = st.selectbox("Model", ["Logistic Regression", "Random Forest"] + (["XGBoost"] if XGB_AVAILABLE else []))

    if model_name == "Logistic Regression":
        clf = LogisticRegression(max_iter=200, n_jobs=None)
    elif model_name == "Random Forest":
        clf = RandomForestClassifier(random_state=42, n_estimators=300)
    else:
        clf = XGBClassifier(
            n_estimators=300,
            learning_rate=0.08,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric="mlogloss"
        )

    pipe = Pipeline([("pre", pre), ("clf", clf)])

    test_size = st.slider("Test size", 0.1, 0.4, 0.2, step=0.05)
    random_state = st.number_input("Random state", value=42, step=1)

    if st.button("Train Model"):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)

        st.success("Model trained.")
        acc = accuracy_score(y_test, preds)
        st.metric("Accuracy", f"{acc:.3f}")

        # Classification report
        report = classification_report(y_test, preds, output_dict=True, zero_division=0)
        st.write(pd.DataFrame(report).T)

        # Confusion matrix
        cm = confusion_matrix(y_test, preds, labels=sorted(y.unique()))
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", xticklabels=sorted(y.unique()), yticklabels=sorted(y.unique()), ax=ax)
        ax.set_title("Confusion Matrix")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        safe_plot(fig)

        # Persist trained artifacts in session state
        st.session_state["trained_pipe"] = pipe
        st.session_state["X_columns"] = X.columns.tolist()
        st.session_state["target"] = target
        st.session_state["label_values"] = sorted(y.unique())

# ------------------------------
# Explain (SHAP)
# ------------------------------
elif section == "Explain (SHAP)":
    st.subheader("Model Explainability")

    if not SHAP_AVAILABLE:
        st.warning("SHAP is not installed in your environment. Add `shap` to requirements.txt to use this section.")
        st.stop()

    if "trained_pipe" not in st.session_state:
        st.info("Train a model first in the **Modeling** section.")
        st.stop()

    pipe = st.session_state["trained_pipe"]
    target = st.session_state["target"]

    # Prepare a small sample for faster SHAP computation
    sample_size = st.slider("Sample size for SHAP", 100, 2000, 500, step=100)
    data = df.dropna(subset=[target]).copy()
    y = data[target].astype(str)
    X = data.drop(columns=[target])

    # Use the pipeline's preprocessing to transform X
    X_trans = pipe.named_steps["pre"].fit_transform(X)
    feature_names = []

    pre = pipe.named_steps["pre"]
    # Collect feature names from ColumnTransformer
    cat_cols = pre.transformers_[0][2]
    num_cols = pre.transformers_[1][2]

    # OneHotEncoder feature names
    try:
        ohe_feature_names = pre.named_transformers_["cat"].get_feature_names_out(cat_cols).tolist()
    except Exception:
        ohe_feature_names = [f"cat_{i}" for i in range(len(cat_cols))]

    feature_names = ohe_feature_names + num_cols

    # Convert to dataframe
    X_trans_df = pd.DataFrame(X_trans, columns=feature_names).reset_index(drop=True)

    # Align sample size
    sample = X_trans_df.sample(n=min(sample_size, len(X_trans_df)), random_state=42)

    clf = pipe.named_steps["clf"]

    try:
        if hasattr(clf, "predict_proba"):
            f = lambda A: pipe.named_steps["clf"].predict_proba(A)
        else:
            f = lambda A: pipe.named_steps["clf"].decision_function(A)

        explainer = shap.Explainer(clf, sample, feature_names=feature_names)
        shap_values = explainer(sample)

        st.write("### SHAP Summary Plot")
        fig = plt.figure()
        shap.summary_plot(shap_values, sample, show=False)
        safe_plot(fig)

        # Bar plot average |SHAP|
        st.write("### Mean |SHAP| values")
        shap_abs = np.abs(shap_values.values).mean(axis=0)
        topk = st.slider("Top-k features", 5, min(30, len(feature_names)), min(10, len(feature_names)))
        imp = pd.Series(shap_abs, index=feature_names).sort_values(ascending=False).head(topk)
        st.write(imp.to_frame("mean_abs_SHAP"))

        fig2, ax2 = plt.subplots(figsize=(6, 4))
        imp.iloc[::-1].plot(kind="barh", ax=ax2)
        ax2.set_title("Top Features by |SHAP|")
        safe_plot(fig2)

    except Exception as e:
        st.error(f"SHAP explanation failed: {e}")
        st.info("Tip: Tree models (RandomForest, XGBoost) tend to work best with SHAP TreeExplainer.")
        # Try TreeExplainer fallback for tree-based models
        try:
            if hasattr(clf, "feature_importances_"):
                explainer = shap.TreeExplainer(clf)
                shap_values = explainer(sample)
                fig = plt.figure()
                shap.summary_plot(shap_values, sample, show=False)
                safe_plot(fig)
        except Exception:
            pass

st.markdown("---")
st.caption("© Streamlit conversion — best-effort reconstruction based on your notebook content.")


