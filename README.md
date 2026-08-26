# 🏦 Predictive Modeling and Risk Scoring for Bank Customer Churn

An end-to-end machine learning project that predicts the probability of bank customer churn and converts predictions into actionable customer risk scores through an interactive Streamlit dashboard.

The project combines **exploratory data analysis, feature engineering, machine learning, probability-based risk scoring, model evaluation, and what-if scenario analysis** to support proactive customer retention strategies.

---

## 📌 Project Overview

Customer churn is a major challenge for retail banks because losing customers can affect:

* Customer Lifetime Value (CLV)
* Revenue stability
* Cross-selling and upselling opportunities
* Long-term customer relationships
* Overall business competitiveness

Traditional churn analysis focuses on understanding customers **after they leave**.

This project takes a predictive approach by identifying customers who are **likely to churn before the churn occurs**.

The system generates a **churn probability between 0 and 1** and assigns each customer a corresponding risk category.

---

## 🎯 Project Objectives

### Primary Objectives

* Predict whether a bank customer is likely to churn.
* Generate customer-level churn probability scores.
* Identify important factors associated with customer churn.
* Compare multiple machine learning algorithms.

### Secondary Objectives

* Reduce false churn predictions.
* Improve model interpretability.
* Support proactive customer retention.
* Provide scenario-based churn analysis.
* Build an interactive web application for business users.

---

## 📊 Dataset

The project uses the supplied `European_Bank.csv` dataset.

### Dataset Statistics

| Property            |  Value |
| ------------------- | -----: |
| Number of Customers | 10,000 |
| Number of Columns   |     14 |
| Churn Rate          | 20.37% |
| Retention Rate      | 79.63% |
| Missing Values      |   None |

### Features

| Feature         | Description                 |
| --------------- | --------------------------- |
| CustomerId      | Unique customer identifier  |
| Surname         | Customer surname            |
| CreditScore     | Customer creditworthiness   |
| Geography       | Customer country            |
| Gender          | Customer gender             |
| Age             | Customer age                |
| Tenure          | Years with the bank         |
| Balance         | Account balance             |
| NumOfProducts   | Number of bank products     |
| HasCrCard       | Credit card ownership       |
| IsActiveMember  | Customer activity indicator |
| EstimatedSalary | Estimated annual salary     |
| Exited          | Target variable             |

### Target Variable

`Exited`

```text
0 → Customer retained
1 → Customer churned
```

---

## 🔬 Project Workflow

```text
                    European Bank Dataset
                            │
                            ▼
                   Data Quality Analysis
                            │
                            ▼
                    Exploratory Data Analysis
                            │
                            ▼
                    Data Preprocessing
                            │
                            ▼
                    Feature Engineering
                            │
                            ▼
                  Stratified Train/Test Split
                            │
                            ▼
                  Machine Learning Models
                            │
             ┌──────────────┼──────────────┐
             ▼              ▼              ▼
        Logistic       Random Forest   Gradient Boosting
        Regression                         │
             │              │              ▼
             └──────────────┼─────────── XGBoost
                            │
                            ▼
                    Model Evaluation
                            │
                            ▼
                  Churn Probability Score
                            │
                            ▼
                      Risk Categorization
                            │
                            ▼
                 Streamlit Web Application
                            │
                            ▼
                  Retention Decision Support
```

---

## 🧹 Data Preprocessing

The following preprocessing steps were implemented:

* Checked for missing values.
* Checked for duplicate records.
* Removed non-predictive identifiers:

  * `CustomerId`
  * `Surname`
* Removed constant metadata such as `Year`.
* Applied one-hot encoding to categorical features.
* Applied numerical imputation where required.
* Applied feature scaling for numerical variables.
* Used a stratified train-test split to preserve the churn distribution.

---

## ⚙️ Feature Engineering

Additional business-oriented features were created to improve predictive capability.

### 1. Balance-to-Salary Ratio

```text
BalanceSalaryRatio = Balance / (EstimatedSalary + 1)
```

This represents the customer's balance relative to estimated salary.

### 2. Product Density

```text
ProductDensity = NumOfProducts / (Tenure + 1)
```

This captures product utilization relative to the customer's relationship duration.

### 3. Engagement-Product Interaction

```text
EngagementProduct = IsActiveMember × NumOfProducts
```

This combines customer activity and product usage.

### 4. Age-Tenure Interaction

```text
AgeTenureInteraction = Age × (Tenure + 1)
```

This captures the interaction between customer age and relationship duration.

---

## 🤖 Machine Learning Models

Four classification algorithms were evaluated.

### Logistic Regression

Used as an interpretable baseline model.

### Random Forest

Used to capture nonlinear relationships and interactions between customer features.

### Gradient Boosting

Used as a powerful ensemble model for structured/tabular data.

### XGBoost

Used as an advanced gradient boosting benchmark.

---

## 📈 Model Performance

The models were evaluated using:

* Accuracy
* Precision
* Recall
* F1-Score
* ROC-AUC

### Results

| Model               |  Accuracy | Precision |    Recall |  F1-Score |   ROC-AUC |
| ------------------- | --------: | --------: | --------: | --------: | --------: |
| Logistic Regression |     71.0% |     38.3% | **69.8%** |     49.4% |     77.6% |
| Random Forest       |     86.7% |     73.9% |     53.6% | **62.1%** |     85.9% |
| Gradient Boosting   | **87.1%** | **78.9%** |     49.6% |     60.9% | **87.0%** |
| XGBoost             |     86.8% |     78.0% |     48.9% |     60.1% |     86.1% |

### Best ROC-AUC Model

**Gradient Boosting — 0.870 ROC-AUC**

ROC-AUC was used as the primary model-selection criterion because the dataset is imbalanced and accuracy alone does not adequately describe churn detection performance.

Random Forest produced the highest F1-score at the default classification threshold.

---

## 🎯 Churn Risk Scoring

Instead of providing only a binary prediction, the system generates a probability score.

Example:

```text
Customer Churn Probability = 73.4%
```

The probability is converted into operational risk bands.

| Churn Probability | Risk Level  |
| ----------------: | ----------- |
|             < 30% | 🟢 Low      |
|        30% – <60% | 🟡 Medium   |
|        60% – <80% | 🟠 High     |
|             ≥ 80% | 🔴 Critical |

> These thresholds are operational examples and should be recalibrated using historical retention outcomes, campaign capacity, intervention costs, and business objectives before production deployment.

---

## 🖥️ Streamlit Dashboard

The project includes an interactive Streamlit application.

### Dashboard Features

#### 👤 Customer Risk Calculator

Users can enter:

* Credit Score
* Geography
* Gender
* Age
* Tenure
* Balance
* Number of Products
* Credit Card ownership
* Active Member status
* Estimated Salary

The application returns:

* Churn probability
* Risk level
* Churn prediction
* Retention recommendation

---

### 📊 Risk Distribution

The dashboard visually represents the predicted churn probability.

---

### 🔄 What-If Scenario Simulator

Users can modify customer characteristics such as:

* Active membership
* Number of products
* Tenure
* Balance

and observe how the predicted churn probability changes.

This enables scenario-based analysis for customer retention planning.

---

## 🏗️ Project Structure

```text
bank-customer-churn-prediction/
│
├── 📄 European_Bank.csv
├── 📓 Bank_Customer_Churn_Analysis.ipynb
├── 🐍 app.py
├── 📄 requirements.txt
├── 📄 Project_Report.md
├── 📄 README.md
└── 🚫 bank_churn_model.joblib
```

`bank_churn_model.joblib` is intentionally excluded from GitHub through `.gitignore`. The model is generated by running the notebook.

---

## 🛠️ Technologies Used

### Programming Language

* Python

### Data Analysis

* Pandas
* NumPy

### Data Visualization

* Matplotlib
* Seaborn

### Machine Learning

* Scikit-learn
* XGBoost

### Model Persistence

* Joblib

### Web Application

* Streamlit

### Development Environment

* Jupyter Notebook
* Git
* GitHub

---

## 🚀 How to Run the Project

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/bank-customer-churn-prediction.git
```

Navigate to the project:

```bash
cd bank-customer-churn-prediction
```

---

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 3. Run the Jupyter Notebook

Open:

```text
Bank_Customer_Churn_Analysis.ipynb
```

Run all cells.

The notebook trains the models and creates:

```text
bank_churn_model.joblib
```

---

### 4. Start the Streamlit Application

```bash
python -m streamlit run app.py
```

The application will normally be available at:

```text
http://localhost:8501
```

---

## 💡 Business Insights

The project demonstrates how predictive analytics can move customer retention from a reactive strategy to a proactive strategy.

Instead of contacting every customer, the bank can prioritize customers according to predicted risk.

### Low-Risk Customers

Maintain normal engagement and relationship-building activities.

### Medium-Risk Customers

Consider targeted communication and product-fit analysis.

### High-Risk Customers

Prioritize proactive retention campaigns and investigate customer engagement.

### Critical-Risk Customers

Consider immediate personalized retention intervention.

---

## 🏦 Business Applications

The system can support:

* Customer retention campaigns
* Personalized offers
* Relationship-manager prioritization
* Product recommendations
* Customer engagement strategies
* Churn monitoring
* Customer segmentation
* Retention campaign optimization

---

## 🔍 Explainability and Responsible AI

For a production banking environment, predictions should be explainable and monitored.

Future versions of this project can include:

* SHAP explanations
* Partial Dependence Plots
* Feature importance dashboards
* Probability calibration
* Fairness analysis
* Model monitoring
* Model drift detection

Predictions should be treated as decision-support information rather than automatic decisions about customers.

---

## 🔮 Future Enhancements

Potential improvements include:

1. SHAP-based individual customer explanations.
2. Partial dependence analysis.
3. Hyperparameter optimization.
4. Cross-validation and probability calibration.
5. Advanced customer segmentation.
6. Automated model retraining.
7. Model monitoring and drift detection.
8. Integration with a production database.
9. Real-time prediction APIs.
10. A cloud deployment of the Streamlit application.
11. Retention campaign ROI optimization.
12. Fairness and responsible-AI monitoring.

---

## 📚 Project Deliverables

This project includes:

* ✅ Exploratory Data Analysis
* ✅ Data preprocessing
* ✅ Feature engineering
* ✅ Multiple ML models
* ✅ Model evaluation
* ✅ Churn probability prediction
* ✅ Risk scoring
* ✅ Interactive Streamlit dashboard
* ✅ What-if scenario analysis
* ✅ Project report
* 🔄 SHAP explainability — planned enhancement
* 🔄 Partial dependence analysis — planned enhancement

---

## 👩‍💻 Author

**Greeshma G Rao**

BCA Student | Data Analytics & Machine Learning

Areas of interest:

* Data Analytics
* Machine Learning
* Business Intelligence
* Predictive Modeling
* Python
* Data Visualization

---

## ⭐ Project Highlights

> **Built an end-to-end machine learning system that predicts bank customer churn, generates probability-based risk scores, compares multiple classification algorithms, and provides an interactive Streamlit interface for proactive retention analysis.**

---

## 📄 License

This project is intended for educational, portfolio, and demonstration purposes.
