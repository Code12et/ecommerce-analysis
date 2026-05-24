import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Olist E-Commerce Dashboard",
    page_icon="🛒",
    layout="wide"
)

# ── Load data ─────────────────────────────────────────────────
@st.cache_data
def load_data():
    path = Path("data/processed/olist_features.parquet")
    return pd.read_parquet(path)

df = load_data()

# ── Sidebar filters ───────────────────────────────────────────
st.sidebar.title("Filters")
states = ["All"] + sorted(df["customer_state"].dropna().unique().tolist())
selected_state = st.sidebar.selectbox("Customer State", states)

if selected_state != "All":
    df = df[df["customer_state"] == selected_state]

# ── Title ─────────────────────────────────────────────────────
st.title("🛒 Olist Brazilian E-Commerce Dashboard")
st.markdown("Analysis of **{:,} orders** from Brazil's largest e-commerce platform.".format(len(df)))

# ── KPI Row ───────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Orders", f"{len(df):,}")
k2.metric("OTIF Rate", f"{(1 - df['is_late'].mean()):.1%}")
k3.metric("Bad Review Rate", f"{df['is_bad_review'].mean():.1%}")
k4.metric("Avg Review Score", f"{df['review_score'].mean():.2f} / 5")

st.divider()

# ── Row 1 ─────────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.subheader("Monthly Order Volume")
    monthly = (
        df.groupby(df["order_purchase_timestamp"].dt.to_period("M"))
        .size()
        .reset_index(name="orders")
    )
    monthly["order_purchase_timestamp"] = monthly["order_purchase_timestamp"].astype(str)
    fig = px.bar(monthly, x="order_purchase_timestamp", y="orders",
                 labels={"order_purchase_timestamp": "Month", "orders": "Orders"})
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Review Score Distribution")
    score_counts = df["review_score"].value_counts().sort_index().reset_index()
    score_counts.columns = ["score", "count"]
    fig = px.bar(score_counts, x="score", y="count",
                 color="score",
                 color_continuous_scale=["red", "orange", "yellow", "lightgreen", "green"])
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# ── Row 2 ─────────────────────────────────────────────────────
col3, col4 = st.columns(2)

with col3:
    st.subheader("Late Delivery Rate by State")
    by_state = (
        df.groupby("customer_state")
        .agg(late_rate=("is_late", "mean"), orders=("order_id", "count"))
        .reset_index()
        .sort_values("late_rate", ascending=False)
    )
    fig = px.bar(by_state, x="customer_state", y="late_rate",
                 color="late_rate", color_continuous_scale="Reds",
                 labels={"late_rate": "Late Rate", "customer_state": "State"})
    st.plotly_chart(fig, use_container_width=True)

with col4:
    st.subheader("Delivery Delay vs Review Score")
    fig = px.box(
        df[df["delivery_delay_days"].between(-10, 30)],
        x="review_score",
        y="delivery_delay_days",
        color="review_score",
        labels={"delivery_delay_days": "Delay (days)", "review_score": "Review Score"}
    )
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# ── Row 3 ─────────────────────────────────────────────────────
col5, col6 = st.columns(2)

with col5:
    st.subheader("Top 10 Sellers by Revenue")
    top_sellers = (
        df.groupby("seller_id")
        .agg(revenue=("payment_value", "sum"), orders=("order_id", "count"))
        .nlargest(10, "revenue")
        .reset_index()
    )
    top_sellers["seller_id"] = top_sellers["seller_id"].str[:8]
    fig = px.bar(top_sellers, x="revenue", y="seller_id",
                 orientation="h",
                 labels={"revenue": "Revenue (BRL)", "seller_id": "Seller"})
    st.plotly_chart(fig, use_container_width=True)

with col6:
    st.subheader("Payment Type Breakdown")
    pay_counts = df["payment_type"].value_counts().reset_index()
    pay_counts.columns = ["type", "count"]
    fig = px.pie(pay_counts, names="type", values="count")
    st.plotly_chart(fig, use_container_width=True)

st.divider()
st.caption("Data: Olist Brazilian E-Commerce Dataset (Kaggle) | 2016–2018")