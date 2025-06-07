
# Project Title

A brief description of what this project does and who it's for

# Loan Approval Prediction System

A machine learning project that predicts whether a loan application will be approved or not based on applicant details. This system is useful for banks and financial institutions to automate and assist in faster decision-making.

---

## 🎯 Objective

To develop a model that classifies loan applications as **approved** or **not approved** using machine learning techniques, with proper data preprocessing and model optimization.

---

## 📊 Dataset Overview

* **Source:** Public Loan Prediction CSV dataset
* **Records:** 614 samples, 13 features

### Features:

* Categorical: Gender, Married, Dependents, Education, Self\_Employed, Property\_Area
* Numerical: ApplicantIncome, CoapplicantIncome, LoanAmount, Loan\_Amount\_Term
* Binary: Credit\_History
* **Target:** Loan\_Status (1 = Approved, 0 = Not Approved)

### Preprocessing:

* Missing values handled using mean/mode
* Categorical features encoded with label encoding
* Applied **StandardScaler** for feature normalization

---

## 🤖 Model Development

### Models Evaluated:

* **Support Vector Machine (SVM)**
* **Random Forest Classifier**

### Final Model: **SVM with GridSearchCV**

### Why SVM?

* Provided the best test accuracy (\~84.6%)
* Generalizes well on small-to-medium datasets
* Performs well after scaling and with tuned parameters

### Hyperparameter Tuning:

Used **GridSearchCV** to find best parameters:

```python
SVC(C=0.1, kernel='linear', gamma='scale')
```

---

## ⚙️ Performance Metrics

| Model         | Train Accuracy | Test Accuracy |
| ------------- | -------------- | ------------- |
| Default SVM   | 83.5%          | 82.4%         |
| Tuned SVM     | 85.1%          | 84.6%         |

### Confusion Matrix & Report

* Evaluated using `classification_report` and `confusion_matrix`
* Focus on **recall** for loan denials

---

## 🧪 Model Testing

* Created a prediction function to check custom inputs
* All test inputs passed through the same scaling as training
* Saved the model using `joblib`

---

## 📌 Challenges & Learnings

### Challenges:

* Handling small and imbalanced dataset
* Model over-reliance on `Credit_History`
* Trade-off between model interpretability and accuracy

### Learnings:

* Proper feature scaling significantly improves SVM performance
* Hyperparameter tuning improves accuracy but not always drastically
* Evaluation metrics beyond accuracy are critical (e.g., precision, recall, F1-score)

---


## 👤 Author

**Nitya Chauhan**
Final Year Computer Engineering Student
Passionate about ML, Data Science, and building intelligent systems

---

## 📜 License

This project is under the MIT License.

---
