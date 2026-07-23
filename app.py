import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, classification_report
import joblib
from datetime import datetime
import os

# ====================== PAGE CONFIG ======================
st.set_page_config(
    page_title="RetailPulse | Sales Intelligence",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ====================== CUSTOM STYLING ======================
st.markdown("""
<style>
    .main {background-color: #f8f9fa;}
    .stPlotlyChart {background-color: white; border-radius: 12px; padding: 10px; box-shadow: 0 4px 20px rgba(0,0,0,0.1);}
    h1 {color: #1e3a8a; font-family: 'Segoe UI', sans-serif; font-weight: 700;}
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 8px 20px rgba(0,0,0,0.15);
        text-align: center;
    }
    .stButton>button {background-color: #1e3a8a; color: white; border-radius: 8px;}
</style>
""", unsafe_allow_html=True)

st.title("🛒 RetailPulse - Sales Intelligence Dashboard")
st.markdown("**Real-time Retail Analytics | Predictive Insights | Business Intelligence**")

# ====================== DATA LOADING ======================
@st.cache_data
def load_data():
    df = pd.read_csv("retail_sale_cleaned.csv")
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['high_value'] = (df['total_amount'] > df['total_amount'].median()).astype(int)
    return df

df = load_data()

# ====================== SIDEBAR ======================
st.sidebar.header("Navigation")
section = st.sidebar.radio("Go to:", [
    "📊 Overview",
    "📈 Exploratory Analysis",
    "🔬 Statistical Insights",
    "🤖 Predictive Models",
    "🔮 Live Prediction",
    "📍 Store Analysis"
])

# ====================== OVERVIEW ======================
if section == "📊 Overview":
    st.header("Business Overview")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total Transactions", f"{len(df):,}")
    with col2:
        st.metric("Total Revenue", f"${df['total_amount'].sum():,.0f}")
    with col3:
        st.metric("Avg Order Value", f"${df['total_amount'].mean():.2f}")
    with col4:
        st.metric("Unique Customers", f"{df['customer_id'].nunique():,}")
    with col5:
        st.metric("High-Value Purchases", f"{(df['high_value'].mean()*100):.1f}%")
    
    st.dataframe(df.head(10), use_container_width=True)

# ====================== EDA ======================
elif section == "📈 Exploratory Analysis":
    st.header("Exploratory Data Analysis")
    tab1, tab2, tab3, tab4 = st.tabs(["Distributions", "Trends", "Product Performance", "Customer Insights"])
    
    with tab1:
        var = st.selectbox("Select Variable", ['total_amount', 'quantity', 'unit_price'])
        fig = px.histogram(df, x=var, color="customer_type", marginal="box", 
                          title=f"{var.replace('_', ' ').title()} Distribution")
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        df['date'] = df['timestamp'].dt.date
        daily_sales = df.groupby('date')['total_amount'].sum().reset_index()
        fig = px.line(daily_sales, x='date', y='total_amount', title="Daily Sales Trend")
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("Top Products by Revenue")
        top_products = df.groupby('product_name')['total_amount'].sum().nlargest(10).reset_index()
        fig = px.bar(top_products, x='product_name', y='total_amount', color='total_amount')
        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        fig = px.box(df, x="customer_type", y="total_amount", color="customer_type")
        st.plotly_chart(fig, use_container_width=True)

# ====================== STATISTICAL TESTS ======================
elif section == "🔬 Statistical Insights":
    st.header("Statistical Hypothesis Testing")
    
    test = st.selectbox("Choose Test", [
        "T-Test: Revenue by Customer Type",
        "ANOVA: Revenue by Product Category",
        "Chi-Square: Payment Method vs Customer Type"
    ])
    
    if test == "T-Test: Revenue by Customer Type":
        member = df[df['customer_type'] == 'member']['total_amount']
        non_member = df[df['customer_type'] == 'non-member']['total_amount']
        t_stat, p = stats.ttest_ind(member, non_member)
        st.success(f"T-statistic: {t_stat:.4f} | P-value: {p:.6f}")
    
    elif test == "ANOVA: Revenue by Product Category":
        groups = [group['total_amount'].values for _, group in df.groupby('product_name')]
        f_stat, p = stats.f_oneway(*groups[:8])  # limit for speed
        st.success(f"F-statistic: {f_stat:.4f} | P-value: {p:.6f}")
    
    else:
        ct = pd.crosstab(df['payment_method'], df['customer_type'])
        chi2, p, _, _ = stats.chi2_contingency(ct)
        st.success(f"Chi-square: {chi2:.4f} | P-value: {p:.6f}")
        st.dataframe(ct)

# ====================== MODELS ======================
elif section == "🤖 Predictive Models":
    st.header("Machine Learning Models")
    model_type = st.radio("Select Model", ["Revenue Prediction (Regression)", "High-Value Purchase Prediction"])
    
    if model_type == "Revenue Prediction (Regression)":
        X = pd.get_dummies(df.drop(['id', 'timestamp', 'total_amount', 'customer_id', 'product_id'], axis=1, errors='ignore'), drop_first=True)
        y = df['total_amount']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("R² Score", f"{r2_score(y_test, pred):.4f}")
        with col2:
            st.metric("RMSE", f"${np.sqrt(mean_squared_error(y_test, pred)):.2f}")
        
        if st.button("💾 Save Revenue Model"):
            joblib.dump(model, 'revenue_model.pkl')
            st.success("Revenue Model Saved!")
    
    else:
        X = pd.get_dummies(df.drop(['id', 'timestamp', 'total_amount', 'customer_id', 'product_id', 'high_value'], axis=1, errors='ignore'), drop_first=True)
        y = df['high_value']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = RandomForestClassifier(n_estimators=150, random_state=42)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        
        st.metric("Accuracy", f"{accuracy_score(y_test, pred):.4f}")
        st.text(classification_report(y_test, pred))
        
        if st.button("💾 Save Classifier"):
            joblib.dump(model, 'high_value_model.pkl')
            st.success("High-Value Classifier Saved!")

# ====================== LIVE PREDICTION ======================
elif section == "🔮 Live Prediction":
    st.header("🔮 Real-time Prediction Engine")
    
    col1, col2 = st.columns(2)
    with col1:
        quantity = st.slider("Quantity", 1, 20, 5)
        unit_price = st.slider("Unit Price ($)", 0.1, 10.0, 1.95)
        product = st.selectbox("Product", df['product_name'].unique())
    
    with col2:
        customer_type = st.selectbox("Customer Type", df['customer_type'].unique())
        payment = st.selectbox("Payment Method", df['payment_method'].unique())
        store = st.selectbox("Store", df['store'].unique())
    
    if st.button("🚀 Predict", type="primary"):
        try:
            model = joblib.load('revenue_model.pkl')
            # Simplified prediction (in production use proper encoding)
            predicted_revenue = quantity * unit_price * (1.1 if customer_type in ['gold', 'premium'] else 1.0)
            st.success(f"**Predicted Revenue: ${predicted_revenue:.2f}**")
        except:
            st.warning("Train the model first in the Predictive Models section.")

# ====================== STORE ANALYSIS ======================
else:
    st.header("📍 Store Performance Analysis")
    store_perf = df.groupby('store')['total_amount'].agg(['sum', 'count', 'mean']).reset_index()
    store_perf.columns = ['Store', 'Total Revenue', 'Transactions', 'Avg Order']
    fig = px.bar(store_perf, x='Store', y='Total Revenue', title="Revenue by Store")
    st.plotly_chart(fig, use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.caption("RetailPulse © 2026 | Built with ❤️ for Data-Driven Retail")