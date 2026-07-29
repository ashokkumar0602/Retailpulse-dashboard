LIVE LINK:https://retailpulse-dashboard-wxtmpxjgnlaagtntnosszg.streamlit.app/

# 🛒 RetailPulse — Sales Intelligence Dashboard

RetailPulse is an interactive, real-time sales intelligence and business analytics dashboard built with **Streamlit**, **Plotly**, and **Scikit-learn**. Designed for data-driven retail management, the application translates transactional raw data into actionable insights, statistical proofs, and predictive revenue forecasts.

---

## 🚀 Dashboard Modules

The dashboard contains 6 modular analytic segments, navigable via the sidebar:
1. **📊 Overview**: Summary metrics of the business (Total Transactions, Revenue, Average Order Value, Customer Base, and High-Value Sales Ratio) along with interactive raw data exploration.
2. **📈 Exploratory Analysis**: Deep dive into transaction amounts, quantities, and price distributions, daily sales trends, product revenues, and customer types.
3. **🔬 Statistical Insights**: Rigorous statistical hypothesis testing including T-Tests, ANOVA, and Chi-Square tests to validate business trends.
4. **🤖 Predictive Models**: Interactive machine learning model training. Train a **Linear Regression** model for revenue forecasting or a **Random Forest Classifier** to identify high-value purchases.
5. **🔮 Live Prediction**: A real-time prediction engine to estimate transaction revenue based on product, store, quantity, customer segment, and price.
6. **📍 Store Analysis**: Visual analytics on store-by-store sales volumes, transaction counts, and average ticket sizes.

---

## 🛠️ Tech Stack & Libraries

- **Frontend & App Framework**: [Streamlit](https://streamlit.io/) (v1.30+)
- **Data Manipulation**: [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/)
- **Data Visualization**: [Plotly Express](https://plotly.com/python/), [Seaborn](https://seaborn.pydata.org/), [Matplotlib](https://matplotlib.org/)
- **Statistical Computation**: [SciPy (Stats)](https://scipy.org/)
- **Machine Learning**: [Scikit-learn](https://scikit-learn.org/) (Linear Regression, Random Forest Classifier)
- **Model Storage**: [Joblib](https://joblib.readthedocs.io/)

---

## 📂 Project Directory Structure

```text
retailpulse-dashboard/
├── app.py                             # Main Streamlit Dashboard Application
├── retail.ipynb                       # Jupyter Notebook for EDA, Testing, and Model Prototyping
├── retail_sale_cleaned.csv            # Cleaned retail dataset (~50.7k transactions)
├── requirements.txt                   # Project package dependencies
├── env/                               # Python Virtual Environment
├── revenue_model.pkl                  # Serialized Linear Regression model for revenue prediction
├── high_value_classifier_model.pkl    # Serialized Random Forest model for high-value classification
└── retail_sales_regression_model.pkl  # Alternative regression model saved from notebook
```

---

## 📊 Dataset Schema (`retail_sale_cleaned.csv`)

The dashboard runs on `retail_sale_cleaned.csv`, which contains **50,783 rows** and **16 columns**:

| Column Name | Data Type | Description |
| :--- | :--- | :--- |
| `id` | String / UUID | Unique transaction identifier |
| `timestamp` | Date / Time | Date and time of the purchase transaction |
| `quantity` | Integer | Total units of the product purchased |
| `product_id` | String / UUID | Unique identifier of the product |
| `product_name` | String | Name of the product (e.g., wheat, etc.) |
| `unit_price` | Float | Price per single unit of the product |
| `total_amount` | Float | Total amount of the transaction (`quantity` × `unit_price`) |
| `store` | String | Store branch/location |
| `payment_method` | String | Method of payment (e.g., cash, credit card, contactless) |
| `customer_id` | String / UUID | Unique identifier of the customer |
| `customer_type` | String | Category of the customer (e.g., member, non-member, gold, corporate) |
| `year` | Integer | Extracted year of transaction |
| `month` | Integer | Extracted month of transaction (1-12) |
| `day` | Integer | Extracted day of the month |
| `day_name` | String | Day of the week (e.g., Monday, Sunday) |
| `hour` | Integer | Hour of the day the transaction occurred (0-23) |

---

## 🔬 Statistical Analysis Details

To back retail assumptions with mathematical certainty, the dashboard computes:

### 1. Two-Sample Independent T-Test
* **Question**: Is there a significant difference in the average transaction value between **Members** and **Non-Members**?
* **Hypothesis**:
  - $H_0$: $\mu_{\text{member}} = \mu_{\text{non-member}}$ (Average ticket sizes are equal)
  - $H_a$: $\mu_{\text{member}} \neq \mu_{\text{non-member}}$ (Average ticket sizes differ)
* **Finding**: P-value $> 0.05$ (e.g., $\approx 0.776$), showing customer type (membership status) alone is not a statistically significant driver of purchase size.

### 2. One-Way ANOVA (Analysis of Variance)
* **Question**: Does transaction revenue vary significantly across different **Product Categories**?
* **Hypothesis**:
  - $H_0$: All product category revenue means are equal.
  - $H_a$: At least one product category mean differs.
* **Finding**: P-value $< 0.001$, indicating product selection is a highly significant driver of transaction totals.

### 3. Chi-Square Contingency Test
* **Question**: Is there an association between the **Payment Method** preferred and the **Customer Type**?
* **Hypothesis**:
  - $H_0$: Payment method and customer type are independent.
  - $H_a$: They are dependent.
* **Finding**: P-value $> 0.05$ (e.g., $\approx 0.747$), showing no strong association between loyalty status and payment preference.

---

## 🤖 Machine Learning Models

### 1. Revenue Forecast (Regression)
- **Model**: Linear Regression
- **Target Variable**: `total_amount`
- **Features**: One-hot encoded categorical features (product, store, customer type, payment method) + continuous features (quantity, unit price, date/time features).
- **Performance**: $R^2 \approx 0.8730$ | $\text{RMSE} \approx 4.08$.

### 2. High-Value Customer Predictor (Classification)
- **Model**: Random Forest Classifier (100–150 estimators)
- **Target Variable**: `high_value` (binary flag: `1` if transaction total is above the median value, `0` otherwise).
- **Features**: Customer demographics, product type, store location, payment method.
- **Performance**: $\text{Accuracy} \approx 99.76\%$, indicating high reliability in predicting transaction status.

---

## ⚙️ Installation & Running Locally

Follow these steps to run the dashboard on your local machine:

### 1. Prerequisites
Ensure you have **Python 3.9+** installed.

### 2. Set Up Virtual Environment (Recommended)
You can use the existing virtual environment `env` or create a new one:

```bash
# Create a new environment (optional)
python -m venv env

# Activate the virtual environment
# On Windows:
.\env\Scripts\activate
# On macOS/Linux:
source env/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Train Models (Optional)
If you want to re-train the models and save them, run the code cells in `retail.ipynb` or go to the **Predictive Models** tab inside the dashboard and click the **Save** buttons.

### 5. Launch the Dashboard
```bash
streamlit run app.py
```
This will start a local server and open the dashboard in your default browser at `http://localhost:8501`.

---

Developed for **Data-Driven Retail Decisions** | 🛒 RetailPulse
