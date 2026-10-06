# 🛡️ FraudShield ML
## Intelligent Financial Fraud Detection & Risk Scoring System

FraudShield ML is an end-to-end machine learning project designed to detect potentially fraudulent financial transactions from transaction-level data.

The project focuses on one of the most challenging problems in financial machine learning: **highly imbalanced classification**, where fraudulent transactions represent only a very small percentage of all transactions.

Instead of relying only on accuracy, FraudShield ML evaluates models using **Precision, Recall, F1-Score, ROC-AUC, PR-AUC, and Confusion Matrix analysis** to understand how effectively each model detects fraud while controlling false alarms.

The final **class-weighted Random Forest classifier** achieved approximately:

- 🎯 **97.1% Precision**
- 🔎 **80.9% Recall**
- 🏆 **88.3% F1-Score**
- 📈 **98.37% ROC-AUC in cross-validation**

The trained model is integrated into a **Streamlit application** that allows users to enter transaction information and receive a fraud prediction and model-generated fraud probability.

---

# 📌 Table of Contents

- [Project Overview](#-project-overview)
- [Problem Statement](#-problem-statement)
- [Project Objectives](#-project-objectives)
- [Why Fraud Detection is Difficult](#-why-fraud-detection-is-difficult)
- [Dataset](#-dataset)
- [Dataset Features](#-dataset-features)
- [Exploratory Data Analysis](#-exploratory-data-analysis)
- [Feature Engineering](#-feature-engineering)
- [Data Preprocessing](#-data-preprocessing)
- [Class Imbalance Handling](#-class-imbalance-handling)
- [Machine Learning Models](#-machine-learning-models)
- [Model Evaluation](#-model-evaluation)
- [Model Comparison](#-model-comparison)
- [Cross-Validation](#-cross-validation)
- [Hyperparameter Tuning](#-hyperparameter-tuning)
- [Threshold Analysis](#-threshold-analysis)
- [Feature Importance](#-feature-importance)
- [Final Model](#-final-model)
- [Streamlit Application](#-streamlit-application)
- [Project Architecture](#-project-architecture)
- [Project Structure](#-project-structure)
- [Technologies Used](#-technologies-used)
- [Installation](#-installation)
- [Running the Application](#-running-the-application)
- [Example Prediction](#-example-prediction)
- [Model Limitations](#-model-limitations)
- [Future Improvements](#-future-improvements)
- [Key Learning Outcomes](#-key-learning-outcomes)
- [Author](#-author)

---

# 🚀 Project Overview

Financial institutions process millions of transactions every day. Among these transactions, only a very small percentage may be fraudulent.

This creates a difficult machine learning problem:

> **How can we identify rare fraudulent transactions without incorrectly flagging large numbers of legitimate transactions?**

FraudShield ML addresses this problem by building and comparing multiple classification algorithms.

The complete workflow includes:

```text
Raw Transaction Data
        ↓
Exploratory Data Analysis
        ↓
Data Cleaning
        ↓
Feature Engineering
        ↓
Train-Test Split
        ↓
Categorical Encoding
        ↓
Feature Scaling
        ↓
Class Imbalance Experiments
        ↓
Multiple ML Models
        ↓
Model Evaluation
        ↓
Cross-Validation
        ↓
Hyperparameter Tuning
        ↓
Threshold Analysis
        ↓
Final Random Forest Model
        ↓
Model Serialization
        ↓
Streamlit Application
        ↓
Fraud Prediction
```

---

# 🎯 Problem Statement

Traditional rule-based fraud detection systems can identify known suspicious patterns, but financial fraud can evolve over time.

A machine learning system can learn patterns from historical transaction data and estimate whether a new transaction resembles previously observed fraudulent behavior.

The goal of FraudShield ML is therefore:

> **Build a machine learning classification system that can distinguish fraudulent transactions from legitimate transactions while maintaining a practical balance between fraud detection and false positive rates.**

---

# 🎯 Project Objectives

The main objectives of this project are:

1. Understand transaction-level financial data.
2. Perform detailed exploratory data analysis.
3. Identify patterns associated with fraudulent transactions.
4. Engineer meaningful balance-related features.
5. Handle highly imbalanced target classes.
6. Train multiple machine learning algorithms.
7. Compare models using fraud-focused evaluation metrics.
8. Perform cross-validation.
9. Perform hyperparameter tuning.
10. Analyze probability thresholds.
11. Analyze feature importance.
12. Select a final model for deployment.
13. Save the trained model and preprocessing objects.
14. Build an interactive Streamlit application.

---

# ⚠️ Why Fraud Detection is Difficult

Fraud detection is different from ordinary classification problems because the classes are extremely imbalanced.

In this dataset:

```text
Legitimate Transactions ≈ 99.87%
Fraudulent Transactions ≈ 0.13%
```

Approximately:

```text
774 legitimate transactions
        :
1 fraudulent transaction
```

This means a model could achieve extremely high accuracy by simply predicting almost every transaction as legitimate.

For example:

```text
If 99.87% of transactions are legitimate,

A model predicting "Legitimate" for every transaction
would appear highly accurate,

but it would detect ZERO fraud.
```

Therefore, accuracy alone is not useful enough for this problem.

FraudShield ML focuses on:

- Precision
- Recall
- F1-Score
- ROC-AUC
- PR-AUC
- Confusion Matrix

---

# 📊 Dataset

The project uses the:

**Synthetic Financial Datasets For Fraud Detection**

Dataset shape:

```text
Rows    : 6,362,620
Columns : 11
```

The dataset contains synthetic financial transaction records with transaction type, transaction amount, sender balances, receiver balances, and fraud labels.

---

# 🧾 Dataset Features

| Feature | Description |
|---|---|
| `step` | Simulated time step |
| `type` | Type of financial transaction |
| `amount` | Transaction amount |
| `nameOrig` | Sender account identifier |
| `oldbalanceOrg` | Sender balance before transaction |
| `newbalanceOrig` | Sender balance after transaction |
| `nameDest` | Receiver account identifier |
| `oldbalanceDest` | Receiver balance before transaction |
| `newbalanceDest` | Receiver balance after transaction |
| `isFraud` | Target variable |
| `isFlaggedFraud` | Existing rule-based fraud flag |

### Target Variable

```text
isFraud = 0 → Legitimate
isFraud = 1 → Fraudulent
```

---

# 🔎 Exploratory Data Analysis

Extensive exploratory analysis was performed before model training.

The analysis examined:

- Missing values
- Duplicate records
- Negative values
- Fraud distribution
- Fraud by transaction type
- Transaction amount distribution
- Balance distributions
- Fraud behavior across time steps
- Feature correlations
- Balance changes
- Existing fraud flags

---

## 📌 Fraud Distribution

The target distribution was:

```text
Legitimate : 6,354,407
Fraud      :     8,213
```

Percentage distribution:

```text
Legitimate → 99.870918%
Fraud      → 0.129082%
```

This confirms that the dataset is **extremely imbalanced**.

---

# 💳 Fraud by Transaction Type

The dataset contains five transaction types:

```text
PAYMENT
TRANSFER
CASH_OUT
CASH_IN
DEBIT
```

Fraudulent transactions were observed only in:

```text
TRANSFER
CASH_OUT
```

No fraudulent transactions were observed in:

```text
PAYMENT
CASH_IN
DEBIT
```

This was an important pattern identified during EDA.

---

# 💰 Transaction Amount Analysis

The transaction amount distributions showed that fraudulent transactions tended to have higher amounts in this dataset, but:

> **A high transaction amount does not automatically mean that the transaction is fraudulent.**

For example, legitimate transactions can also have very large amounts.

Therefore, the model uses multiple transaction and balance features instead of relying only on transaction amount.

---

# 🏦 Balance Analysis

Sender and receiver balances were analyzed before and after transactions.

Important observations included:

- Strong relationships between old and new balance features.
- Differences between sender and receiver balance changes.
- Fraudulent transactions showed different balance-change patterns.
- Some transactions contained balance inconsistencies.

These observations motivated additional feature engineering.

---

# 🛠️ Feature Engineering

Two important features were created.

## 1. Organization Balance Change

```python
df["org_balance_change"] = (
    df["oldbalanceOrg"] -
    df["newbalanceOrig"]
)
```

This represents the recorded decrease in the sender's balance.

---

## 2. Destination Balance Change

```python
df["dest_balance_change"] = (
    df["newbalanceDest"] -
    df["oldbalanceDest"]
)
```

This represents the recorded change in the receiver's balance.

These engineered features became important predictors for the Random Forest model.

---

# 🧹 Data Preprocessing

## Removing Identifier Columns

The following columns were removed:

```text
nameOrig
nameDest
```

These are synthetic account identifiers and were not used as predictive features.

An EDA-only feature was also excluded:

```text
time_period
```

---

## Train-Test Split

The data was split into:

```text
80% Training
20% Testing
```

Using:

```python
train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)
```

### Why `stratify=y`?

Because the fraud class is extremely rare, stratification ensures that the training and testing datasets maintain a similar fraud/legitimate distribution.

---

# 🔤 Categorical Encoding

The `type` feature is categorical.

It was converted into numerical features using one-hot encoding:

```python
pd.get_dummies(
    X,
    columns=["type"],
    drop_first=True
)
```

This allows machine learning algorithms to work with transaction types numerically.

---

# 📏 Feature Scaling

Numerical features were standardized using:

```python
StandardScaler()
```

The scaler was fitted only on the training data:

```python
scaler.fit_transform(X_train)
```

and then applied to the test data:

```python
scaler.transform(X_test)
```

This prevents information from the test set from leaking into the training process.

---

# ⚖️ Class Imbalance Handling

Because fraud represents only around 0.13% of the dataset, different strategies were investigated.

Three important approaches were compared:

### 1. Class Weighting

The Random Forest was trained using:

```python
class_weight="balanced"
```

This increases the importance of the minority fraud class during model training.

---

### 2. Random Undersampling

The majority class was reduced to match the minority class.

Training distribution became approximately:

```text
Legitimate → 6,570
Fraud      → 6,570
```

The resulting model achieved approximately:

```text
Precision → 12%
Recall    → 100%
F1        → 21%
```

Although recall was excellent, the large number of false positives made the approach less suitable for the final model.

---

### 3. SMOTE

SMOTE was used to generate synthetic minority-class examples.

The resulting Random Forest achieved approximately:

```text
Precision → 70%
Recall    → 98%
F1        → 82%
```

SMOTE significantly improved fraud recall but produced more false positives than the original class-weighted Random Forest.

---

# 🤖 Machine Learning Models

Several machine learning algorithms were evaluated.

### Models tested

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. HistGradientBoosting
5. XGBoost
6. Random Forest + Random Undersampling
7. Random Forest + SMOTE

This comparison helped determine which approach provided the best balance between fraud detection and false alarms.

---

# 📈 Model Evaluation

The models were evaluated using:

## Precision

Precision answers:

> Of all transactions predicted as fraud, how many were actually fraud?

```text
Precision = TP / (TP + FP)
```

High precision means fewer legitimate transactions are incorrectly flagged.

---

## Recall

Recall answers:

> Of all actual fraudulent transactions, how many did the model detect?

```text
Recall = TP / (TP + FN)
```

For fraud detection, recall is particularly important because missing a fraudulent transaction can be costly.

---

## F1-Score

F1 combines precision and recall:

```text
F1 = 2 × (Precision × Recall)
     -------------------------
       Precision + Recall
```

It is useful when we need a balance between detecting fraud and avoiding excessive false alarms.

---

## ROC-AUC

ROC-AUC measures the model's ability to distinguish between the two classes across different probability thresholds.

---

## Confusion Matrix

The confusion matrix provides:

```text
True Negative
False Positive
False Negative
True Positive
```

This is particularly useful for understanding fraud detection performance.

---

# 🏆 Model Comparison

| Model | Precision | Recall | F1-Score |
|---|---:|---:|---:|
| Logistic Regression | 2.6% | 97.1% | 5.1% |
| Decision Tree | 89% | 87% | 88% |
| **Random Forest** | **97.1%** | **80.9%** | **88.3%** |
| Hist Gradient Boosting | 25% | 99% | 40% |
| XGBoost | 20% | 99% | 34% |
| RF + Undersampling | 12% | 100% | 21% |
| RF + SMOTE | 70% | 98% | 82% |

### Key observation

Different models showed very different precision-recall trade-offs.

For example:

```text
Logistic Regression
High Recall
Very Low Precision
```

while:

```text
Random Forest
High Precision
Strong Recall
High F1
```

The class-weighted Random Forest provided the strongest overall balance among the tested approaches.

---

# 🔄 Cross-Validation

To evaluate model stability, **5-fold Stratified Cross-Validation** was performed on a representative subset of the training data.

Random Forest results:

```text
Mean Precision → 98.75%
Mean Recall    → 72.90%
Mean F1        → 83.84%
Mean ROC-AUC   → 98.37%
```

Cross-validation showed that the model maintained strong precision and ROC-AUC across multiple folds.

---

# ⚙️ Hyperparameter Tuning

Randomized hyperparameter search was performed for the Random Forest.

Parameters explored included:

```python
n_estimators
max_depth
min_samples_leaf
max_features
```

The best configuration found during tuning was:

```text
n_estimators    = 100
max_depth       = 20
min_samples_leaf = 2
max_features    = None
```

However, the tuned model did not outperform the original Random Forest on the final test set.

Therefore, the original class-weighted Random Forest was retained as the final model.

This is an important part of the project:

> **Hyperparameter tuning was not performed simply to increase a metric artificially; the final model was selected based on actual test performance and practical trade-offs.**

---

# 🎚️ Probability Threshold Analysis

The Random Forest probability threshold was evaluated across multiple values:

```text
0.50
0.55
0.60
0.65
0.70
0.75
0.80
0.85
0.90
0.95
```

Results showed that the highest tested F1-score occurred at:

```text
Threshold = 0.50
```

Therefore, the default 0.50 threshold was retained for the deployed application.

---

# 🔍 Feature Importance

The final Random Forest identified the following features as the most important:

| Rank | Feature | Importance |
|---:|---|---:|
| 1 | `org_balance_change` | ~30.42% |
| 2 | `oldbalanceOrg` | ~16.73% |
| 3 | `newbalanceOrig` | ~11.51% |
| 4 | `amount` | ~11.38% |
| 5 | `dest_balance_change` | ~7.56% |
| 6 | `type_TRANSFER` | ~6.14% |
| 7 | `step` | ~4.05% |

### Important Note

These values represent **impurity-based feature importance from the Random Forest**.

They should not be interpreted as causal relationships.

For example:

```text
org_balance_change = 30.42%
```

does **not** mean that this feature causes 30.42% of fraud.

It means that this feature contributed strongly to the model's decision-making according to the Random Forest's impurity-based importance calculation.

---

# 🏆 Final Model

The final model selected for deployment is:

```text
Random Forest Classifier
```

Configuration:

```python
RandomForestClassifier(
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)
```

### Final Test Performance

```text
Precision → 97.1%
Recall    → 80.9%
F1-Score  → 88.3%
```

Confusion Matrix:

```text
                 Predicted
              Legit     Fraud

Actual Legit  1270847      34
Actual Fraud      317    1326
```

Therefore:

```text
True Negatives  = 1,270,847
False Positives = 34
False Negatives = 317
True Positives  = 1,326
```


# 🖥️ Streamlit Application

FraudShield ML includes an interactive Streamlit interface.

Users can enter:

- Transaction Type
- Transaction Amount
- Transaction Step
- Sender Balance Before
- Sender Balance After
- Receiver Balance Before
- Receiver Balance After
- System Fraud Flag

The application then performs the same essential preprocessing used during model training.

---

# 🔄 Application Prediction Pipeline

```text
User Transaction Input
          ↓
Calculate Balance Changes
          ↓
Create DataFrame
          ↓
One-Hot Encode Transaction Type
          ↓
Align Feature Columns
          ↓
Apply Saved StandardScaler
          ↓
Load Saved Random Forest
          ↓
Generate Prediction
          ↓
Calculate Fraud Probability
          ↓
Display Result
```

Example:

```text
Input Transaction
       ↓
Random Forest
       ↓
Prediction = 1
       ↓
🚨 Potential Fraudulent Transaction

Fraud Probability = 54%
```

---

# 🧪 Example Fraud Test

A real fraudulent transaction from the dataset was tested through the Streamlit application.

Example transaction:

```text
Transaction Type      : TRANSFER
Transaction Amount    : ₹7,567,170.36
Transaction Step      : 38
Sender Balance Before : ₹7,567,170.36
Sender Balance After  : ₹0
Receiver Balance Before: ₹0
Receiver Balance After : ₹0
System Flag            : 0
```

The application predicted:

```text
Prediction         : Fraud
Fraud Probability  : 54%
```

This demonstrates that the deployed pipeline can successfully process a known fraudulent transaction and classify it as fraudulent.

---

# 🏗️ Project Architecture

```text
                    ┌───────────────────────┐
                    │ Financial Transactions│
                    └───────────┬───────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │   Exploratory Data    │
                    │       Analysis        │
                    └───────────┬───────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │ Feature Engineering   │
                    │                       │
                    │ Balance Changes       │
                    │ Transaction Features │
                    └───────────┬───────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │    Preprocessing      │
                    │                       │
                    │ Encoding             │
                    │ Scaling              │
                    │ Train/Test Split     │
                    └───────────┬───────────┘
                                │
                                ▼
              ┌─────────────────────────────────┐
              │      Model Experimentation      │
              │                                 │
              │ Logistic Regression             │
              │ Decision Tree                   │
              │ Random Forest                   │
              │ Gradient Boosting               │
              │ XGBoost                         │
              │ SMOTE / Undersampling           │
              └───────────────┬─────────────────┘
                              │
                              ▼
                    ┌───────────────────────┐
                    │   Model Evaluation    │
                    │                       │
                    │ Precision             │
                    │ Recall                │
                    │ F1                    │
                    │ ROC-AUC               │
                    └───────────┬───────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │   Final Random Forest │
                    └───────────┬───────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │    Joblib Models      │
                    └───────────┬───────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │   Streamlit App       │
                    └───────────┬───────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │ Fraud Prediction      │
                    │ + Risk Probability    │
                    └───────────────────────┘
```

---

# 📁 Project Structure

```text
FraudShield-ML/
│
├── data/
│   └── Synthetic Financial Datasets For Fraud Detection.csv
│
├── notebooks/
│   ├── FraudShield_ML.ipynb
│   ├── Random_forest.ipynb
│   ├── Decision_Tree.ipynb
│   ├── High_Gradient.ipynb
│   ├── logistic_Reg.ipynb
│   └── XGBoost.ipynb
│
├── models/
│   ├── final_random_forest.pkl
│   ├── scaler.pkl
│   └── feature_columns.pkl
│
├── reports/
│   └── model_results.csv
│
├── app.py
├── requirements.txt
└── README.md
```

---

# 🛠️ Technologies Used

## Programming

- Python

## Data Analysis

- Pandas
- NumPy

## Visualization

- Matplotlib

## Machine Learning

- Scikit-learn
- XGBoost
- imbalanced-learn

## Model Persistence

- Joblib

## Application

- Streamlit

## Development

- Jupyter Notebook
- Visual Studio Code
- Git
- GitHub

---

# 📦 Installation

## 1. Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

Move into the project directory:

```bash
cd FraudShield-ML
```

---

## 2. Create a Virtual Environment

Windows:

```bash
python -m venv venv
```

Activate it:

```bash
venv\Scripts\activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Run the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

The application will open in your browser.

---

# 📋 Example Workflow

### Step 1

Open the FraudShield ML application.

### Step 2

Enter the transaction details.

Example:

```text
Transaction Type      → TRANSFER
Transaction Amount    → 7567170.36
Transaction Step      → 38
Sender Balance Before → 7567170.36
Sender Balance After  → 0
Receiver Balance Before → 0
Receiver Balance After  → 0
System Flag           → 0
```

### Step 3

Click:

```text
Predict Transaction
```

### Step 4

The model returns:

```text
🚨 Potential Fraudulent Transaction

Fraud Probability: 54%
```

---

# 📊 Why Random Forest Was Selected

Random Forest was selected because it provided a strong balance between:

```text
Fraud Detection
       +
Low False Positives
       +
Strong F1-Score
       +
High Precision
```

Compared with the other tested approaches:

- Logistic Regression produced many false positives.
- Gradient boosting models achieved very high recall but substantially lower precision.
- Random undersampling produced too many false positives.
- SMOTE improved recall but still produced more false positives than the original Random Forest.
- Hyperparameter tuning did not improve the baseline Random Forest's test performance.

Therefore, the class-weighted Random Forest was selected based on **overall test performance and practical trade-offs**.

---

# ⚠️ Model Limitations

This project has several limitations that should be considered.

### 1. Synthetic Dataset

The dataset is synthetic and may not fully represent real-world banking transactions.

### 2. Class Distribution

The extremely imbalanced class distribution makes model evaluation challenging.

### 3. Historical Patterns

The model learns patterns from the available dataset. New fraud strategies may behave differently.

### 4. Probability Interpretation

The Random Forest probability is a model-generated class probability and should not automatically be interpreted as a perfectly calibrated real-world probability of fraud.

### 5. No Live Banking Integration

The Streamlit application currently performs prediction from manually entered transaction information.

It is not connected to a live banking transaction system.

---

# 🚀 Future Improvements

Future versions of FraudShield ML could include:

### Advanced Model Explainability

Integrate SHAP or other explainability techniques to show why a transaction was classified as suspicious.

### Cost-Sensitive Learning

Different costs can be assigned to:

```text
False Positive
False Negative
```

This would allow threshold selection based on actual business requirements.

### Probability Calibration

Calibrate model probabilities to make the displayed risk score more reliable.

### Advanced Anomaly Detection

Explore algorithms such as:

- Isolation Forest
- One-Class SVM
- Autoencoders

These could help identify previously unseen suspicious behavior.

### Real-Time Transaction Processing

The system could be extended to consume transaction streams and perform automated fraud scoring.

### Model Monitoring

A production system could monitor:

- Data drift
- Concept drift
- Model performance
- Fraud detection rate
- False positive rate

### Cloud Deployment

The Streamlit application could be deployed to a cloud platform for public access.

---

# 🎓 Key Machine Learning Concepts Demonstrated

This project demonstrates practical understanding of:

### Data Analysis

- Exploratory Data Analysis
- Distribution analysis
- Correlation analysis
- GroupBy analysis
- Outlier investigation

### Feature Engineering

- Balance change features
- Transaction behavior analysis
- Feature selection

### Preprocessing

- Train-test splitting
- Stratified splitting
- One-hot encoding
- Standardization

### Imbalanced Learning

- Class weighting
- Random undersampling
- SMOTE

### Machine Learning

- Logistic Regression
- Decision Trees
- Random Forest
- Gradient Boosting
- HistGradientBoosting
- XGBoost

### Model Evaluation

- Confusion Matrix
- Precision
- Recall
- F1-Score
- ROC-AUC
- PR-AUC
- Cross-validation
- Threshold analysis

### Model Optimization

- RandomizedSearchCV
- Hyperparameter tuning
- Probability threshold analysis

### Deployment

- Joblib
- Streamlit
- Model persistence
- Reusing preprocessing pipelines

---

# 💡 Key Takeaways

The major lessons from this project are:

1. **Accuracy is not enough for highly imbalanced fraud detection problems.**

2. **Precision and recall must be considered together.**

3. **Feature engineering can significantly improve fraud detection.**

4. **Class imbalance requires careful experimentation.**

5. **SMOTE and undersampling do not automatically produce better models.**

6. **A more complex model is not always better than a simpler one.**

7. **Cross-validation helps evaluate model stability.**

8. **Threshold selection can change the precision-recall trade-off.**

9. **The preprocessing used during inference must match the preprocessing used during training.**

10. **The final model should be selected based on practical performance, not just the highest individual metric.**

---

# 👨‍💻 Author

## Naveen

**Python | Machine Learning | Data Science**

This project was developed as an end-to-end machine learning project covering the complete workflow from raw financial transaction data to model deployment.

---

# ⭐ Project Highlights

```text
✔ 6.3M+ financial transactions
✔ Highly imbalanced fraud classification
✔ Detailed EDA
✔ Feature engineering
✔ Multiple ML algorithms
✔ SMOTE experimentation
✔ Random undersampling
✔ Cross-validation
✔ Hyperparameter tuning
✔ Threshold analysis
✔ Feature importance
✔ Random Forest final model
✔ Model persistence using Joblib
✔ Interactive Streamlit application
✔ Fraud probability scoring
```

---

## 📌 Final Result

FraudShield ML demonstrates a complete machine learning workflow for financial fraud detection:

```text
6.3M+ Transactions
        ↓
Exploratory Data Analysis
        ↓
Feature Engineering
        ↓
Imbalanced Data Handling
        ↓
7 ML Approaches
        ↓
Model Evaluation
        ↓
Cross-Validation
        ↓
Hyperparameter Tuning
        ↓
Threshold Analysis
        ↓
Random Forest
        ↓
97.1% Precision
80.9% Recall
88.3% F1-Score
        ↓
Streamlit Deployment
        ↓
🛡️ FraudShield ML
```