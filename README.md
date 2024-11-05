# 📊 Customer Churn Prediction Project

Predicting customer churn is essential for helping businesses retain customers and grow revenue. This project leverages the Telco Customer Churn dataset to build predictive models that identify which customers are likely to leave. By understanding the factors driving churn, companies can develop targeted retention strategies.

---

## 🌟 Project Highlights

- **Dataset**: [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)  
  - **Size**: 7,000+ records
  - **Features**: Customer demographics, services, contract details, and churn status
- **Objective**: Predict whether a customer will churn based on their demographic and service information.
- **Tools Used**: Python, Pandas, Seaborn, Matplotlib, Scikit-Learn

---

## 🧩 Project Workflow

### 1️⃣ Data Cleaning & Preprocessing
- **Handled Missing Values**: Missing values in `TotalCharges` were replaced with the median for consistency.
- **Data Type Conversions**: Converted `TotalCharges` to numeric format, enabling calculations and analysis.

### 2️⃣ Exploratory Data Analysis (EDA)
- **Churn Distribution Analysis**: Observed the churn distribution to assess dataset balance.
- **Feature Analysis**: Visualized distributions and relationships to understand which factors may influence churn.

### 3️⃣ Feature Engineering
- **Grouped Tenure**: Created tenure categories (e.g., `0-1 year`, `1-2 years`) to simplify analysis.
- **Spending Bins**: Segmented monthly charges into bins (`Low`, `Medium`, `High`, `Very High`) for better insights.
- **Calculated Customer Lifetime Value (CLV)**: Estimated CLV by multiplying tenure and monthly charges to identify high-value customers.

### 4️⃣ Modeling & Evaluation
- **Models Used**: 
  - **Logistic Regression**: Tuned using GridSearchCV for optimal parameters.
  - **Random Forest Classifier**: Captured non-linear relationships for better performance.
- **Evaluation Metrics**: Employed **AUC-ROC** and **Classification Report** to assess model performance on distinguishing between churners and non-churners.

---

## 🔍 Insights from EDA

- **Shorter Contracts**: Customers on month-to-month contracts had higher churn rates than those on longer contracts.
- **Monthly Spending**: Higher monthly charges correlated with increased churn, suggesting that budget-conscious customers may be more at risk.
- **Add-On Services**: Customers without additional services (like online security or tech support) showed a higher tendency to churn.

---

## 🚀 Future Improvements

- **Advanced Models**: Experiment with boosting algorithms like Gradient Boosting or XGBoost to potentially enhance prediction accuracy.
- **Feature Importance Analysis**: Use tools like SHAP or LIME to better interpret which features most influence churn, particularly for individual predictions.
- **Customer Segmentation**: Apply clustering techniques to segment customers by behavior and tailor retention strategies accordingly.
- **Cross-Validation**: Further validate models using cross-validation to ensure consistency and robustness across different data splits.

---

## 💡 Getting Started

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/PranayaKD/churn-prediction.git
2. **Install Dependencies**:
   pip install -r requirements.txt
3.**Run the Code**: Each section is modular to allow step-by-step exploration of data, feature engineering, and modeling.
