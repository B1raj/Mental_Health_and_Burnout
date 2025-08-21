import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- PAGE CONFIG ---
st.set_page_config(page_title="Mental Health in the Workplace", layout="wide")

# --- LOAD DATA ---
@st.cache_data
def load_data():
    # Assuming the dataset is stored as CSV in the notebook directory
    df = pd.read_csv("mental_health_data final data.csv")
    return df

df = load_data()

# --- HEADER ---
st.title("🧠 Mental Health in the Workplace - Survey Insights")
st.markdown("An overview of insights from the Workplace Mental Health Survey with interactive EDA.")

# --- DATA PREVIEW ---
with st.expander("📊 View Raw Data"):
    st.dataframe(df.head())

# --- SIDEBAR FILTERS ---
st.sidebar.header("Filters")
selected_country = st.sidebar.multiselect("Select Country", options=df["Country"].unique(), default=df["Country"].unique())
filtered_df = df[df["Country"].isin(selected_country)]

# --- SUMMARY STATS ---
st.subheader("Dataset Overview")
st.write("Number of responses:", len(filtered_df))
st.write("Countries represented:", filtered_df["Country"].nunique())

# --- VISUALIZATIONS ---
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Mental Health Condition vs Gender")
    fig, ax = plt.subplots()
    sns.countplot(data=filtered_df, x="Gender", hue="Mental_Health_Condition", ax=ax)
    plt.xticks(rotation=30)
    st.pyplot(fig)

with col2:
    st.markdown("#### Age Distribution")
    fig, ax = plt.subplots()
    sns.histplot(filtered_df["Age"], kde=True, bins=20, ax=ax, color="teal")
    st.pyplot(fig)

# --- COMPANY FACTORS (adapted to lifestyle/workplace factors) ---
st.subheader("Lifestyle and Work-Related Insights")
col3, col4 = st.columns(2)

with col3:
    st.markdown("#### Stress Level Distribution")
    fig, ax = plt.subplots()
    sns.countplot(x="Stress_Level", data=filtered_df, order=filtered_df["Stress_Level"].value_counts().index, ax=ax)
    st.pyplot(fig)

with col4:
    st.markdown("#### Sleep Hours Distribution")
    fig, ax = plt.subplots()
    sns.histplot(filtered_df["Sleep_Hours"], bins=15, kde=True, ax=ax, color="purple")
    st.pyplot(fig)

# --- ADDITIONAL FACTORS ---
st.subheader("Other Health & Lifestyle Factors")
col5, col6 = st.columns(2)

with col5:
    st.markdown("#### Physical Activity Hours")
    fig, ax = plt.subplots()
    sns.histplot(filtered_df["Physical_Activity_Hours"], bins=15, kde=True, ax=ax, color="green")
    st.pyplot(fig)

with col6:
    st.markdown("#### Social Media Usage")
    fig, ax = plt.subplots()
    sns.histplot(filtered_df["Social_Media_Usage"], bins=15, kde=True, ax=ax, color="orange")
    st.pyplot(fig)

# --- FOOTER ---
st.markdown("---")
st.caption("Developed using Streamlit | Data Source: Workplace Mental Health Survey")
