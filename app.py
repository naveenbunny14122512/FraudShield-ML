import streamlit as st
import pandas as pd
import joblib

# Load saved ML objects
model = joblib.load("models/final_random_forest.pkl")
scaler = joblib.load("models/scaler.pkl")
feature_columns = joblib.load("models/feature_columns.pkl")

print("Model and preprocessing objects loaded successfully!")
st.set_page_config(
    page_title="FraudShield ML",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ FraudShield ML")
st.subheader("Intelligent Financial Fraud Detection System")

st.write(
    "Enter the transaction details below to predict whether "
    "the transaction is potentially fraudulent."
)

st.header("💳 Transaction Details")

col1, col2 = st.columns(2)

with col1:
    transaction_type = st.selectbox(
        "Transaction Type",
        ["PAYMENT", "TRANSFER", "CASH_OUT", "CASH_IN", "DEBIT"]
    )

    amount = st.number_input(
        "Transaction Amount",
        min_value=0.0,
        value=10000.0,
        step=100.0
    )

    step = st.number_input(
        "Transaction Step",
        min_value=1,
        value=1,
        step=1
    )

    oldbalance_org = st.number_input(
        "Sender Balance Before",
        min_value=0.0,
        value=50000.0,
        step=100.0
    )

with col2:
    newbalance_orig = st.number_input(
        "Sender Balance After",
        min_value=0.0,
        value=40000.0,
        step=100.0
    )

    oldbalance_dest = st.number_input(
        "Receiver Balance Before",
        min_value=0.0,
        value=100000.0,
        step=100.0
    )

    newbalance_dest = st.number_input(
        "Receiver Balance After",
        min_value=0.0,
        value=110000.0,
        step=100.0
    )

    is_flagged_fraud = st.selectbox(
        "System Flag",
        [0, 1]
    )
    
# st.header("🔍 Fraud Detection")

# if st.button("Predict Transaction", type="primary"):

#     # Create balance-change features
#     org_balance_change = oldbalance_org - newbalance_orig
#     dest_balance_change = newbalance_dest - oldbalance_dest

#     # Create input DataFrame
#     input_data = pd.DataFrame({
#         "step": [step],
#         "amount": [amount],
#         "oldbalanceOrg": [oldbalance_org],
#         "newbalanceOrig": [newbalance_orig],
#         "oldbalanceDest": [oldbalance_dest],
#         "newbalanceDest": [newbalance_dest],
#         "isFlaggedFraud": [is_flagged_fraud],
#         "org_balance_change": [org_balance_change],
#         "dest_balance_change": [dest_balance_change]
#     })

#     # Encode transaction type
#     input_data = pd.get_dummies(
#         input_data,
#         columns=["type"],
#         drop_first=True
#     )

#     # Make columns exactly the same as training
#     input_data = input_data.reindex(
#         columns=feature_columns,
#         fill_value=0
#     )

#     # Scale numeric features
#     numeric_cols = [
#         "step",
#         "amount",
#         "oldbalanceOrg",
#         "newbalanceOrig",
#         "oldbalanceDest",
#         "newbalanceDest",
#         "org_balance_change",
#         "dest_balance_change"
#     ]

#     input_data[numeric_cols] = scaler.transform(
#         input_data[numeric_cols]
#     )

#     # Make prediction
#     prediction = model.predict(input_data)[0]

#     # Fraud probability
#     probability = model.predict_proba(input_data)[0][1]

#     if prediction == 1:
#         st.error("🚨 Potential Fraudulent Transaction")
#         st.metric(
#             "Fraud Probability",
#             f"{probability:.2%}"
#         )

#     else:
#         st.success("✅ Transaction Classified as Legitimate")
#         st.metric(
#             "Fraud Probability",
#             f"{probability:.2%}"
#         )

st.header("🔍 Fraud Detection")

if st.button("Predict Transaction", type="primary"):

    # Calculate balance-change features
    org_balance_change = oldbalance_org - newbalance_orig
    dest_balance_change = newbalance_dest - oldbalance_dest

    # Create input DataFrame
    input_data = pd.DataFrame({
        "step": [step],
        "type": [transaction_type],
        "amount": [amount],
        "oldbalanceOrg": [oldbalance_org],
        "newbalanceOrig": [newbalance_orig],
        "oldbalanceDest": [oldbalance_dest],
        "newbalanceDest": [newbalance_dest],
        "isFlaggedFraud": [is_flagged_fraud],
        "org_balance_change": [org_balance_change],
        "dest_balance_change": [dest_balance_change]
    })

    # One-hot encode transaction type
    input_data = pd.get_dummies(
        input_data,
        columns=["type"],
        drop_first=True
    )

    # Match exactly the columns used during training
    input_data = input_data.reindex(
        columns=feature_columns,
        fill_value=0
    )

    # Numeric columns that were scaled during training
    numeric_cols = [
        "step",
        "amount",
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest",
        "org_balance_change",
        "dest_balance_change"
    ]

    # Apply the saved scaler
    input_data[numeric_cols] = scaler.transform(
        input_data[numeric_cols]
    )

    # Prediction
    prediction = model.predict(input_data)[0]
    
    

    # Fraud probability
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.error("🚨 Potential Fraudulent Transaction")
    else:
        st.success("✅ Transaction Classified as Legitimate")
    st.write(prediction)
    st.metric(
        "Fraud Probability",
        f"{probability:.2%}"
    )