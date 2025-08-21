
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- PAGE CONFIG ---
st.set_page_config(page_title="Mental Health Dashboard", layout="wide")

# --- LOAD DATA ---
@st.cache_data
def load_data():
    try:
        return pd.read_csv("./data/mental_health_data_final_data.csv")
    except:
        return None

df = load_data()

# --- SIDEBAR ---
st.sidebar.title("Navigation")
section = st.sidebar.radio("Go to", ["Overview", "Data Explorer", "Visualizations", "Insights"])

# --- OVERVIEW ---
if section == "Overview":
    st.title("📊 Mental Health in the Workplace Dashboard")
    st.markdown(
        "This dashboard highlights key findings from the Mental Health Workplace Survey. "
        "Use the sidebar to explore data, visualizations, and insights."
    )
    if df is not None:
        st.metric("Total Responses", df.shape[0])
        st.metric("Total Features", df.shape[1])
    else:
        st.warning("No dataset found. Please upload it to ./data/")

# --- DATA EXPLORER ---
elif section == "Data Explorer":
    st.title("🔍 Explore Dataset")
    if df is not None:
        st.dataframe(df.head(20))
        st.write("### Column Information")
        st.write(df.dtypes)
    else:
        st.error("Dataset not found.")

# --- VISUALIZATIONS ---
elif section == "Visualizations":
    st.title("📈 Visualizations")
    if df is not None:
        # Stress Level Distribution
        st.subheader("Stress Level Distribution")
        fig, ax = plt.subplots()
        sns.countplot(data=df, x="Stress_Level", ax=ax)
        st.pyplot(fig)

        # Sleep vs Stress
        st.subheader("Average Sleep Hours by Stress Level")
        fig, ax = plt.subplots()
        sns.barplot(data=df, x="Stress_Level", y="Sleep_Hours", ax=ax, estimator="mean")
        st.pyplot(fig)

        # Work Hours vs Stress
        st.subheader("Average Work Hours by Stress Level")
        fig, ax = plt.subplots()
        sns.barplot(data=df, x="Stress_Level", y="Work_Hours", ax=ax, estimator="mean")
        st.pyplot(fig)
    else:
        st.error("Dataset not available.")

# --- INSIGHTS ---
elif section == "Insights":
    st.title("💡 Key Insights")
    st.markdown(
        """
        - High stress is often linked with **longer work hours** and **less sleep**.  
        - Poor **diet quality** and **low physical activity** correlate with higher stress.  
        - Employees reporting **healthy lifestyle choices** tend to have lower stress levels.  
        """
    )

