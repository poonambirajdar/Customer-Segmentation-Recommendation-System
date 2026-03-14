import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

st.set_page_config(page_title="Customer Analytics Dashboard", layout="wide")

st.title("Customer Segmentation & Churn Prediction Dashboard")

# Load data
df = pd.read_csv(r"D:\Customer segmentation project\processed_data/rfm_features.csv")

# Load models
kmeans = joblib.load("model/kmeans_model.pkl")
scaler = joblib.load("model/scaler.pkl")
churn_model = joblib.load("model/churn_model.pkl")

features = df[["recency","frequency","monetary","avg_payment"]]

X = scaler.transform(features)

df["cluster"] = kmeans.predict(X)

# -----------------------------
# KPI SECTION
# -----------------------------

total_customers = df.shape[0]
avg_spending = round(df["monetary"].mean(),2)
avg_frequency = round(df["frequency"].mean(),2)
churn_rate = round((df["recency"] > 180).mean()*100,2)

col1,col2,col3,col4 = st.columns(4)

col1.metric("Total Customers", total_customers)
col2.metric("Average Spending", avg_spending)
col3.metric("Avg Purchase Frequency", avg_frequency)
col4.metric("Churn Risk %", churn_rate)

st.divider()

# -----------------------------
# FILTERS
# -----------------------------

st.sidebar.header("Filters")

cluster_filter = st.sidebar.multiselect(
    "Select Customer Segment",
    df["cluster"].unique(),
    df["cluster"].unique()
)

df_filtered = df[df["cluster"].isin(cluster_filter)]

# -----------------------------
# CLUSTER VISUALIZATION
# -----------------------------

st.subheader("Customer Segmentation")

fig = px.scatter(
    df_filtered,
    x="frequency",
    y="monetary",
    color="cluster",
    size="monetary",
    hover_data=["recency"]
)

st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# DISTRIBUTION CHART
# -----------------------------

col1,col2 = st.columns(2)

with col1:
    st.subheader("Spending Distribution")
    fig = px.box(df_filtered, y="monetary", color="cluster")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Purchase Frequency Distribution")
    fig = px.histogram(df_filtered, x="frequency", color="cluster")
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# -----------------------------
# CUSTOMER PREDICTION
# -----------------------------

st.subheader("Predict Customer Segment & Churn")

col1,col2,col3 = st.columns(3)

with col1:
    recency = st.number_input("Recency (days)",0,365,30)

with col2:
    frequency = st.number_input("Purchase Frequency",1,50,5)

with col3:
    monetary = st.number_input("Total Spending",0,10000,500)

avg_payment = monetary / frequency

cluster = kmeans.predict(
    scaler.transform([[recency,frequency,monetary,avg_payment]])
)[0]

churn = churn_model.predict([[recency,frequency,monetary]])[0]

st.write("### Predicted Customer Segment:", cluster)

if churn == 1:
    st.error("Customer likely to churn")

else:
    st.success("Customer likely to stay")

# -----------------------------
# MARKETING STRATEGY
# -----------------------------

def marketing_strategy(cluster):

    if cluster == 0:
        return "VIP Customers → Offer premium products and exclusive deals."

    elif cluster == 1:
        return "Frequent Buyers → Introduce loyalty rewards."

    elif cluster == 2:
        return "Price Sensitive → Provide discount campaigns."

    else:
        return "Inactive Customers → Send re-engagement emails."

st.subheader("Marketing Recommendation")

st.info(marketing_strategy(cluster))