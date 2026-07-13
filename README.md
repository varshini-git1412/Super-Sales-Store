# 📊 End-to-End Sales Forecasting & Demand Intelligence System

## 📌 Project Overview

This project is an interactive **Sales Forecasting & Demand Intelligence Dashboard** developed using **Python** and **Streamlit**.

The dashboard helps businesses analyze historical sales, forecast future demand, detect unusual sales patterns, and segment products based on demand characteristics.

---

## 🎯 Objectives

- Analyze historical sales data.
- Forecast future sales using Machine Learning.
- Detect sales anomalies.
- Segment products based on demand.
- Provide business recommendations for inventory management.

---

## 🚀 Features

### 📈 Sales Overview
- Total Sales
- Total Orders
- Yearly Sales Analysis
- Monthly Sales Trend
- Region & Category Filters
- Sales by Sub-Category

### 🔮 Forecast Explorer
- Forecast by Category
- Forecast by Region
- Forecast Horizon (1–3 Months)
- XGBoost Forecasting
- Model Performance Metrics
  - MAE: **14,443.46**
  - RMSE: **17,069.09**

### 🚨 Anomaly Detection
- Isolation Forest
- Z-Score Method
- Weekly Sales Trend
- Detected Anomalies
- Summary Comparison

### 📦 Demand Segmentation
- K-Means Clustering
- PCA Visualization
- Cluster Summary
- Product Demand Categories
- Stocking Recommendations

---

## 🧠 Machine Learning Models

### Forecasting
- SARIMA
- Prophet
- ✅ XGBoost (Best Performing Model)

### Anomaly Detection
- Isolation Forest
- Z-Score

### Clustering
- K-Means
- PCA (Visualization)

---

## 📂 Dataset

Dataset Used:

- Super Sales Store Dataset (`train.csv`)

Main columns include:

- Order Date
- Ship Date
- Sales
- Region
- Category
- Sub-Category
- Order ID

---

## 🛠️ Technologies Used

- Python
- Streamlit
- Pandas
- NumPy
- Plotly
- Matplotlib
- Scikit-learn
- XGBoost

---

## 📁 Project Structure

```
Super-Sales-Store/
│
├── app.py
├── analysis.ipynb
├── train.csv
├── vgsales.csv
├── requirements.txt
├── README.md
└── venv/
```

---

## ▶️ Installation

Clone the repository:

```bash
git clone https://github.com/varshini-git1412/Super-Sales-Store.git
```

Move into the project directory:

```bash
cd Super-Sales-Store
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

---

## 📊 Dashboard Modules

- Sales Overview
- Forecast Explorer
- Anomaly Report
- Demand Segments

---

## 📈 Results

### Best Forecast Model

| Model | MAE | RMSE |
|------|---------:|---------:|
| SARIMA | 18031.40 | 19009.18 |
| Prophet | 20250.79 | 22318.41 |
| ✅ XGBoost | **14443.46** | **17069.09** |

---

## 💼 Business Recommendations

- Maintain higher inventory for stable demand products.
- Increase stock for growing-demand products.
- Monitor highly volatile products.
- Reduce inventory for declining-demand products.

---

## 🔮 Future Enhancements

- Deep Learning (LSTM) Forecasting
- Real-time Dashboard
- Cloud Deployment
- Automated Alerts
- Interactive Reporting

---

## 👩‍💻 Developed By

**Talanki Varshini**

Artificial Intelligence & Data Science Student

GitHub: https://github.com/varshini-git1412

---

## 📄 License

This project is developed for educational and internship purposes.
