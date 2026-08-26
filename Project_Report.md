# Project Report: Predictive Modeling and Risk Scoring for Bank Customer Churn

## Executive Summary
This project develops an end-to-end machine-learning framework for identifying bank customers at risk of churn. The supplied European Bank dataset contains 10,000 customer records and 14 columns. The target variable is `Exited`, where 1 indicates churn and 0 indicates retention.

The dataset contains no missing values. The churn rate is 20.37%, so evaluation must consider class imbalance rather than accuracy alone.

## Methodology
1. Data quality validation and exploratory data analysis.
2. Removal of identifier fields (`CustomerId`, `Surname`) and constant metadata (`Year`).
3. One-hot encoding of `Geography` and `Gender`.
4. Numerical imputation/scaling inside a scikit-learn pipeline.
5. Feature engineering:
   - Balance-to-Salary Ratio
   - Product Density
   - Engagement-Product Interaction
   - Age-Tenure Interaction
6. Stratified 80/20 train-test split.
7. Comparison of Logistic Regression, Random Forest, Gradient Boosting and XGBoost.
8. Evaluation using Accuracy, Precision, Recall, F1 and ROC-AUC.
9. Threshold analysis to balance retention-team workload and missed churners.
10. Probability-based risk bands and a Streamlit what-if simulator.

## Benchmark Results
On the supplied dataset with the provided implementation and a 0.50 classification threshold:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.710 | 0.383 | 0.698 | 0.494 | 0.776 |
| Random Forest | 0.867 | 0.739 | 0.536 | 0.621 | 0.859 |
| Gradient Boosting | 0.871 | 0.789 | 0.496 | 0.609 | 0.870 |
| XGBoost | 0.868 | 0.780 | 0.489 | 0.601 | 0.861 |

Gradient Boosting gives the strongest ROC-AUC in this benchmark, while Random Forest provides a slightly higher F1 at the default threshold. The final notebook automatically selects the model with the highest ROC-AUC and separately analyzes classification thresholds.

## Business Interpretation
A churn score should not be treated as a final business decision. It should prioritize customers for human-reviewed retention action. High-risk customers can receive targeted engagement, product-fit reviews, service recovery or personalized offers.

## Streamlit Application
The dashboard provides:
- customer churn risk calculator;
- churn probability;
- risk band;
- retention interpretation;
- what-if scenario simulation for activity, products, tenure and balance.

## Recommendations
- Use recall-sensitive thresholds when the cost of missing a churner is high.
- Use precision-sensitive thresholds when retention campaigns are expensive.
- Monitor calibration of probabilities over time.
- Validate interventions with controlled experiments.
- Add SHAP explanations and partial-dependence analysis for production explainability.
- Review fairness across geography and gender before deployment.
- Retrain periodically as customer behavior changes.

## Conclusion
The project converts historical churn data into an actionable predictive-risk framework. The combination of probability scoring, threshold analysis, explainability and a Streamlit interface supports proactive rather than reactive retention.
