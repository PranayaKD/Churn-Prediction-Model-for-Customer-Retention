# Customer Churn Prediction
This project aims to predict customer churn using the Telco Customer Churn dataset. 
The analysis focuses on understanding the drivers of churn, engineering predictive features, 
and developing machine learning models to improve the prediction accuracy. Key steps include data exploration,
feature engineering, model training, and evaluation.

Table of Contents
Project Overview
Data Exploration (EDA)
Feature Engineering
Modeling
Evaluation and Insights
Conclusion
Future Work
Project Overview
Dataset: Telco Customer Churn Dataset
Goal: To predict whether a customer will churn (leave) based on various demographic, account, and service-related features.
Tools: Python, Pandas, Seaborn, Matplotlib, Scikit-Learn

Data Exploration (EDA)
1. Data Loading and Initial Check
python
Copy code
import pandas as pd

# Load the dataset
data = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# View the first few rows
data.head()
Purpose: Load the dataset and review the initial data structure.
Insight: Observing column names, missing values, and data types provides the foundation for preprocessing.
2. Handling Missing Values
python
Copy code
# Convert 'TotalCharges' to numeric, using 'coerce' to set errors to NaN
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')

# Fill missing values with the median
data['TotalCharges'].fillna(data['TotalCharges'].median(), inplace=True)
Purpose: Convert TotalCharges to a numeric type and handle missing values.
Insight: Missing values in TotalCharges were filled with the median to maintain data consistency without distorting the feature distribution.
3. Exploratory Data Analysis (EDA)
Distribution of Churn
python
Copy code
import seaborn as sns
import matplotlib.pyplot as plt

# Plot churn distribution
sns.countplot(x='Churn', data=data)
plt.title("Churn Distribution")
plt.show()
Purpose: Understand the imbalance between churned and non-churned customers.
Insight: Roughly 26% of customers churned, indicating a slight imbalance that must be considered during model evaluation.
Feature Engineering
1. Tenure Grouping
python
Copy code
# Define tenure groups
def tenure_group(tenure):
    if tenure <= 12:
        return '0-1 year'
    elif tenure <= 24:
        return '1-2 years'
    elif tenure <= 48:
        return '2-4 years'
    elif tenure <= 60:
        return '4-5 years'
    else:
        return '5+ years'

data['TenureGroup'] = data['tenure'].apply(tenure_group)
Purpose: Categorize tenure into broader groups to simplify model interpretation.
Insight: Helps identify general trends in churn likelihood for different tenure groups.
2. Monthly Charges Binning
python
Copy code
# Define spending bins
data['MonthlyChargesBin'] = pd.cut(data['MonthlyCharges'], bins=[0, 30, 60, 90, 120], labels=['Low', 'Medium', 'High', 'Very High'])
Purpose: Group customers by monthly spending to observe if higher spending correlates with churn.
Insight: Can reveal if budget-friendly plans could aid retention of high-spend customers.
3. Customer Lifetime Value (CLV)
python
Copy code
# Estimate Customer Lifetime Value (CLV)
data['CLV'] = data['MonthlyCharges'] * data['tenure']
Purpose: Approximate CLV to identify high-value customers whose churn would significantly impact revenue.
Insight: Useful for prioritizing high-value customers in retention efforts.
Modeling
1. Preparing Data for Modeling
python
Copy code
from sklearn.model_selection import train_test_split

# One-hot encoding for categorical variables
data_encoded = pd.get_dummies(data, drop_first=True)

# Define X and y
X = data_encoded.drop(columns=['Churn', 'customerID'])
y = data_encoded['Churn']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
Purpose: Prepare data for modeling, including one-hot encoding of categorical features and train-test split.
Insight: Ensures the model is trained on a well-represented subset of the data.
2. Model Training and Hyperparameter Tuning
Logistic Regression
python
Copy code
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

# Define parameter grid for tuning
param_grid_log_reg = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2']
}

log_reg = LogisticRegression(solver='saga', max_iter=2000, random_state=42)
grid_search_log_reg = GridSearchCV(log_reg, param_grid_log_reg, cv=5, scoring='roc_auc')
grid_search_log_reg.fit(X_train, y_train)

# Best parameters
print("Best parameters for Logistic Regression:", grid_search_log_reg.best_params_)
Purpose: Tune regularization parameters to optimize model performance.
Insight: Logistic Regression provides interpretability, showing how each feature influences churn.
Random Forest
python
Copy code
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
Purpose: Train a more flexible model capable of capturing non-linear relationships.
Insight: Random Forests are robust against overfitting and can reveal feature importance.
Evaluation and Insights
1. Model Evaluation
python
Copy code
from sklearn.metrics import classification_report, roc_auc_score

# Logistic Regression Evaluation
log_reg_best = grid_search_log_reg.best_estimator_
y_pred_log = log_reg_best.predict(X_test)
print("Logistic Regression Performance:")
print(classification_report(y_test, y_pred_log))
print("AUC Score:", roc_auc_score(y_test, log_reg_best.predict_proba(X_test)[:, 1]))
Purpose: Evaluate model performance using classification report and AUC score.
Insight: AUC-ROC indicates how well the model distinguishes between churners and non-churners.
Conclusion
Key Insights:

Shorter contracts (e.g., month-to-month) have higher churn rates.
High monthly charges are associated with increased churn.
Customers without add-on services like tech support are more likely to churn.
Feature Engineering:

Grouping tenure and monthly charges, as well as calculating CLV, improved model interpretability and accuracy.
Future Work
Explore More Algorithms: Consider gradient boosting models for potentially better accuracy.
Interpretability Tools: Use SHAP or LIME to understand feature impact at an individual customer level.
Cross-Validation: Further validate results with cross-validation to ensure model robustness.
