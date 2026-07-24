import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, classification_report
import joblib
import os
from datetime import datetime

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
    .main {
        background-color: #0f172a;
        color: #e2e8f0;
    }
    .stPlotlyChart {
        background-color: #1e2937;
        border-radius: 16px;
        padding: 16px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    }
    h1, h2, h3 {
        color: #60a5fa;
        font-family: 'Segoe UI', system-ui, sans-serif;
        font-weight: 700;
    }
    .metric-card {
        background: linear-gradient(135deg, #3b82f6 0%, #1e40af 100%);
        color: white;
        padding: 24px;
        border-radius: 16px;
        box-shadow: 0 10px 25px rgba(59, 130, 246, 0.2);
        text-align: center;
        border: 1px solid rgba(255,255,255,0.1);
    }
    .stButton>button {
        background-color: #3b82f6;
        color: white;
        border-radius: 12px;
        border: none;
        padding: 12px 24px;
        font-weight: 600;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #2563eb;
        transform: translateY(-2px);
    }
    .sidebar .css-1d391kg {
        background-color: #1e2937;
    }
    .dataframe {
        border-radius: 12px;
        overflow: hidden;
    }
</style>
""", unsafe_allow_html=True)

st.title("🛒 RetailPulse")
st.markdown("**Modern Retail Analytics Dashboard** | *Data-Driven Insights for Smarter Retail*")

# ====================== DATA LOADING ======================
@st.cache_data(ttl=3600)
def load_data():
    try:
        df = pd.read_csv("retail_sale_cleaned.csv")
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['date'] = df['timestamp'].dt.date
        df['high_value'] = (df['total_amount'] > df['total_amount'].median()).astype(int)
        return df
    except FileNotFoundError:
        st.error("Data file 'retail_sale_cleaned.csv' not found. Please ensure it's in the working directory.")
        return pd.DataFrame()

df = load_data()

if df.empty:
    st.stop()

# ====================== SIDEBAR FILTERS ======================
st.sidebar.header("🔍 Global Filters")
date_range = st.sidebar.date_input(
    "Date Range",
    [df['date'].min(), df['date'].max()],
    min_value=df['date'].min(),
    max_value=df['date'].max()
)

filtered_df = df.copy()
if len(date_range) == 2:
    filtered_df = filtered_df[
        (filtered_df['date'] >= date_range[0]) & 
        (filtered_df['date'] <= date_range[1])
    ]

store_filter = st.sidebar.multiselect(
    "Select Stores",
    options=filtered_df['store'].unique(),
    default=filtered_df['store'].unique()
)

if store_filter:
    filtered_df = filtered_df[filtered_df['store'].isin(store_filter)]

st.sidebar.header("Navigation")
section = st.sidebar.radio("Go to Section:", [
    "📊 Overview",
    "📈 Exploratory Analysis",
    "🔬 Statistical Insights",
    "🤖 Predictive Models",
    "🔮 Live Prediction",
    "📍 Store Analysis"
])

# ====================== HELPER FUNCTIONS ======================
def create_metric_card(title, value, delta=None, icon="📊"):
    st.markdown(f"""
    <div class="metric-card">
        <h4 style="margin:0; opacity:0.9;">{icon} {title}</h4>
        <h2 style="margin:8px 0 0 0; font-size:2.2em;">{value}</h2>
        {f'<p style="margin:0; font-size:0.9em; opacity:0.8;">{delta}</p>' if delta else ''}
    </div>
    """, unsafe_allow_html=True)

# ====================== OVERVIEW ======================
if section == "📊 Overview":
    st.header("Business Overview")
    
    # KPI Cards
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        create_metric_card("Total Transactions", f"{len(filtered_df):,}", icon="🔄")
    with col2:
        total_rev = filtered_df['total_amount'].sum()
        create_metric_card("Total Revenue", f"${total_rev:,.0f}", icon="💰")
    with col3:
        avg_order = filtered_df['total_amount'].mean()
        create_metric_card("Avg Order Value", f"${avg_order:.2f}", icon="📈")
    with col4:
        unique_cust = filtered_df['customer_id'].nunique()
        create_metric_card("Unique Customers", f"{unique_cust:,}", icon="👥")
    with col5:
        hv_pct = (filtered_df['high_value'].mean() * 100)
        create_metric_card("High-Value %", f"{hv_pct:.1f}%", icon="⭐")
    
    # Enhanced Visualizations
    st.subheader("Key Performance Visualizations")
    viz_col1, viz_col2 = st.columns(2)
    
    with viz_col1:
        # Revenue Trend
        daily_sales = filtered_df.groupby('date')['total_amount'].sum().reset_index()
        fig_trend = px.line(
            daily_sales, x='date', y='total_amount',
            title="📈 Daily Sales Trend",
            template="plotly_dark",
            markers=True,
            color_discrete_sequence=["#60a5fa"]
        )
        fig_trend.update_layout(height=400)
        st.plotly_chart(fig_trend, use_container_width=True)
        
        # Store Revenue Breakdown
        store_rev = filtered_df.groupby('store')['total_amount'].sum().reset_index()
        fig_store = px.pie(
            store_rev, names='store', values='total_amount',
            title="💼 Revenue Distribution by Store",
            template="plotly_dark"
        )
        st.plotly_chart(fig_store, use_container_width=True)
    
    with viz_col2:
        # Customer Type Revenue
        cust_rev = filtered_df.groupby('customer_type')['total_amount'].agg(['sum', 'count']).reset_index()
        fig_cust = px.bar(
            cust_rev, x='customer_type', y='sum',
            title="👥 Revenue by Customer Type",
            color='customer_type',
            text='sum',
            template="plotly_dark"
        )
        fig_cust.update_traces(texttemplate='$%{text:,.0f}')
        st.plotly_chart(fig_cust, use_container_width=True)
        
        # High-Value Purchases
        hv_dist = filtered_df['high_value'].value_counts()
        fig_hv = px.pie(
            values=hv_dist.values, names=['Regular', 'High-Value'],
            title="⭐ High-Value vs Regular Purchases",
            template="plotly_dark",
            color_discrete_sequence=["#94a3b8", "#3b82f6"]
        )
        st.plotly_chart(fig_hv, use_container_width=True)
    
    # Recent Transactions
    st.subheader("Recent Transactions")
    st.dataframe(
        filtered_df.sort_values('timestamp', ascending=False).head(10)[
            ['timestamp', 'product_name', 'customer_type', 'total_amount', 'store']
        ],
        use_container_width=True,
        hide_index=True
    )

# ====================== EDA ======================
elif section == "📈 Exploratory Analysis":
    st.header("Exploratory Data Analysis")
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Distributions", 
        "📈 Trends", 
        "🏆 Product Performance", 
        "👥 Customer Insights"
    ])
    
    with tab1:
        st.subheader("Variable Distributions")
        var = st.selectbox("Select Variable", ['total_amount', 'quantity', 'unit_price'])
        color_var = st.selectbox("Color By", ['customer_type', 'payment_method', None], index=0)
        
        fig = px.histogram(
            filtered_df, x=var, color=color_var, 
            marginal="box", nbins=50,
            title=f"{var.replace('_', ' ').title()} Distribution",
            template="plotly_dark"
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Sales Trends")
        daily_sales = filtered_df.groupby('date')['total_amount'].sum().reset_index()
        fig = px.line(
            daily_sales, x='date', y='total_amount',
            title="Daily Sales Trend",
            template="plotly_dark",
            markers=True
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        if st.checkbox("Show Monthly Aggregation"):
            filtered_df['month'] = pd.to_datetime(filtered_df['date']).dt.to_period('M')
            monthly = filtered_df.groupby('month')['total_amount'].sum().reset_index()
            monthly['month'] = monthly['month'].astype(str)
            fig_m = px.bar(monthly, x='month', y='total_amount', title="Monthly Sales", template="plotly_dark")
            st.plotly_chart(fig_m, use_container_width=True)
    
    with tab3:
        st.subheader("Top Products")
        top_products = filtered_df.groupby('product_name')['total_amount'].agg(['sum', 'count']).reset_index()
        top_products = top_products.nlargest(10, 'sum')
        
        fig = px.bar(
            top_products, x='product_name', y='sum',
            title="Top 10 Products by Revenue",
            color='sum',
            text='sum',
            template="plotly_dark"
        )
        fig.update_traces(texttemplate='$%{text:,.0f}', textposition='outside')
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Product Quantity vs Revenue")
        fig_scatter = px.scatter(
            filtered_df.groupby('product_name').agg({
                'quantity': 'sum', 
                'total_amount': 'sum'
            }).reset_index(),
            x='quantity', y='total_amount',
            size='total_amount', color='product_name',
            title="Quantity vs Revenue per Product",
            template="plotly_dark"
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    with tab4:
        st.subheader("Customer Behavior")
        col_a, col_b = st.columns(2)
        with col_a:
            fig_box = px.box(
                filtered_df, x="customer_type", y="total_amount",
                color="customer_type", title="Order Value by Customer Type",
                template="plotly_dark"
            )
            st.plotly_chart(fig_box, use_container_width=True)
        with col_b:
            fig_violin = px.violin(
                filtered_df, x="customer_type", y="total_amount",
                color="customer_type", title="Distribution by Customer Type",
                template="plotly_dark"
            )
            st.plotly_chart(fig_violin, use_container_width=True)
        
        payment_dist = filtered_df['payment_method'].value_counts()
        fig_pie = px.pie(
            values=payment_dist.values, names=payment_dist.index,
            title="Payment Method Distribution",
            template="plotly_dark"
        )
        st.plotly_chart(fig_pie, use_container_width=True)

# ====================== STATISTICAL TESTS ======================
elif section == "🔬 Statistical Insights":
    st.header("Statistical Hypothesis Testing")
    
    test = st.selectbox("Choose Statistical Test", [
        "T-Test: Revenue by Customer Type",
        "ANOVA: Revenue by Product Category",
        "Chi-Square: Payment Method vs Customer Type"
    ])
    
    if test == "T-Test: Revenue by Customer Type":
        member = filtered_df[filtered_df['customer_type'] == 'member']['total_amount']
        non_member = filtered_df[filtered_df['customer_type'] == 'non-member']['total_amount']
        if len(member) > 0 and len(non_member) > 0:
            t_stat, p = stats.ttest_ind(member, non_member)
            st.success(f"**T-statistic:** {t_stat:.4f} | **P-value:** {p:.6f}")
            st.info("P-value < 0.05 indicates significant difference in revenue between customer types.")
        else:
            st.warning("Insufficient data for this test.")
    
    elif test == "ANOVA: Revenue by Product Category":
        groups = [group['total_amount'].values for _, group in filtered_df.groupby('product_name')]
        if len(groups) > 1:
            f_stat, p = stats.f_oneway(*groups[:10])
            st.success(f"**F-statistic:** {f_stat:.4f} | **P-value:** {p:.6f}")
        else:
            st.warning("Insufficient groups for ANOVA.")
    
    else:
        ct = pd.crosstab(filtered_df['payment_method'], filtered_df['customer_type'])
        chi2, p, dof, expected = stats.chi2_contingency(ct)
        st.success(f"**Chi-square statistic:** {chi2:.4f} | **P-value:** {p:.6f}")
        st.subheader("Contingency Table")
        st.dataframe(ct, use_container_width=True)

# ====================== PREDICTIVE MODELS ======================
elif section == "🤖 Predictive Models":
    st.header("Machine Learning Models")
    
    model_type = st.radio("Select Model Type", ["Revenue Prediction (Regression)", "High-Value Purchase Prediction"])
    
    @st.cache_resource
    def train_revenue_model(data):
        X = pd.get_dummies(
            data.drop(['id', 'timestamp', 'total_amount', 'customer_id', 'product_id', 'date', 'high_value'], 
                     axis=1, errors='ignore'), 
            drop_first=True
        )
        y = data['total_amount']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        
        return model, r2_score(y_test, pred), np.sqrt(mean_squared_error(y_test, pred))
    
    @st.cache_resource
    def train_classifier(data):
        X = pd.get_dummies(
            data.drop(['id', 'timestamp', 'total_amount', 'customer_id', 'product_id', 'date', 'high_value'], 
                     axis=1, errors='ignore'), 
            drop_first=True
        )
        y = data['high_value']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        
        return model, accuracy_score(y_test, pred), classification_report(y_test, pred)
    
    if model_type == "Revenue Prediction (Regression)":
        st.subheader("Linear Regression - Revenue Prediction")
        model, r2, rmse = train_revenue_model(filtered_df)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("R² Score", f"{r2:.4f}", help="Higher is better")
        with col2:
            st.metric("RMSE", f"${rmse:.2f}", help="Lower is better")
        
        if st.button("💾 Save Revenue Model", type="primary"):
            joblib.dump(model, 'revenue_model.pkl')
            st.success("✅ Revenue Model saved successfully!")
    
    else:
        st.subheader("Random Forest - High-Value Purchase Classifier")
        model, acc, report = train_classifier(filtered_df)
        
        st.metric("Accuracy", f"{acc:.4f}")
        st.code(report, language="text")
        
        if st.button("💾 Save Classifier", type="primary"):
            joblib.dump(model, 'high_value_model.pkl')
            st.success("✅ High-Value Model saved successfully!")

# ====================== LIVE PREDICTION ======================
elif section == "🔮 Live Prediction":
    st.header("🔮 Real-time Prediction Engine")
    st.markdown("Adjust parameters below for instant predictions")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Transaction Details")
        quantity = st.slider("Quantity", 1, 50, 5)
        unit_price = st.slider("Unit Price ($)", 0.1, 100.0, 1.95, step=0.05)
        product = st.selectbox("Product", options=filtered_df['product_name'].unique())
    
    with col2:
        st.subheader("Customer & Store")
        customer_type = st.selectbox("Customer Type", options=filtered_df['customer_type'].unique())
        payment = st.selectbox("Payment Method", options=filtered_df['payment_method'].unique())
        store = st.selectbox("Store Location", options=filtered_df['store'].unique())
    
    if st.button("🚀 Generate Prediction", type="primary", use_container_width=True):
        try:
            if os.path.exists('revenue_model.pkl'):
                model = joblib.load('revenue_model.pkl')
                base = quantity * unit_price
                multiplier = 1.15 if customer_type in ['gold', 'premium', 'member'] else 1.0
                predicted_revenue = base * multiplier
                st.success(f"**Predicted Total Revenue: ${predicted_revenue:.2f}**")
                st.balloons()
            else:
                st.info("💡 Tip: Train the Revenue Model first for better predictions")
                predicted_revenue = quantity * unit_price * (1.1 if customer_type in ['gold', 'premium'] else 1.0)
                st.success(f"**Estimated Revenue (Heuristic): ${predicted_revenue:.2f}**")
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")

# ====================== STORE ANALYSIS ======================
else:
    st.header("📍 Store Performance Analysis")
    
    store_perf = filtered_df.groupby('store').agg({
        'total_amount': ['sum', 'mean', 'count'],
        'high_value': 'mean'
    }).reset_index()
    store_perf.columns = ['Store', 'Total Revenue', 'Avg Order Value', 'Transactions', 'High-Value Rate']
    store_perf = store_perf.sort_values('Total Revenue', ascending=False)
    
    fig = px.bar(
        store_perf, x='Store', y='Total Revenue',
        title="Total Revenue by Store",
        color='Avg Order Value',
        text='Total Revenue',
        template="plotly_dark"
    )
    fig.update_traces(texttemplate='$%{text:,.0f}')
    st.plotly_chart(fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        fig2 = px.scatter(
            store_perf, x='Transactions', y='Avg Order Value',
            size='Total Revenue', color='Store',
            title="Efficiency: Transactions vs Avg Order",
            template="plotly_dark"
        )
        st.plotly_chart(fig2, use_container_width=True)
    with col2:
        st.subheader("Store Performance Table")
        st.dataframe(
            store_perf.style.format({
                'Total Revenue': '${:,.0f}',
                'Avg Order Value': '${:.2f}',
                'High-Value Rate': '{:.1%}'
            }),
            use_container_width=True,
            hide_index=True
        )

# Footer
st.sidebar.markdown("---")
st.sidebar.caption("RetailPulse © 2026 | Professional Retail Intelligence Platform")
st.markdown("---")
st.markdown("*Built with Streamlit • Plotly • scikit-learn*")