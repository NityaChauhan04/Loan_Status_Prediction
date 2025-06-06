import streamlit as st
import numpy as np
import joblib


model = joblib.load('svm_loan_model.pkl')
scaler = joblib.load('scaler.pkl')


st.set_page_config(page_title="Loan Approval Prediction", layout="centered")


st.markdown("""
    <style>
        body {
            background-color: #f5f7fa;
            font-family: 'Segoe UI', sans-serif;
        }
        .main {
            background-color: #ffffff;
            padding: 2rem;
            border-radius: 15px;
            box-shadow: 0px 0px 10px rgba(0,0,0,0.1);
        }
        .stButton button {
            background-color: #4CAF50;
            color: white;
            border-radius: 8px;
            padding: 10px 20px;
        }
        .stButton button:hover {
            background-color: #45a049;
        }
        h1 {
            color: #0c4b65;
        }
    </style>
""", unsafe_allow_html=True)


st.title("Loan Approval Prediction App")
st.write("Fill in the details below to check your loan approval status.")


with st.container():
    Gender = st.selectbox("Gender", ["Male", "Female"])
    Married = st.selectbox("Married", ["Yes", "No"])
    Dependents = st.selectbox("Number of Dependents", ["0", "1", "2", "3+"])
    Education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    Self_Employed = st.selectbox("Self Employed", ["Yes", "No"])
    ApplicantIncome = st.number_input("Applicant Income", min_value=0)
    CoapplicantIncome = st.number_input("Coapplicant Income", min_value=0.0)
    LoanAmount = st.number_input("Loan Amount (in thousands)", min_value=0)
    Loan_Amount_Term = st.number_input("Loan Amount Term (in days)", min_value=0)
    Credit_History = st.selectbox("Credit History", ["Yes", "No"])
    Property_Area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])


gender_val = 1 if Gender == "Male" else 0
married_val = 1 if Married == "Yes" else 0
education_val = 1 if Education == "Graduate" else 0
self_employed_val = 1 if Self_Employed == "Yes" else 0
credit_val = 1.0 if Credit_History == "Yes" else 0.0
property_val = {"Urban": 2, "Semiurban": 1, "Rural": 0}[Property_Area]
dependents_val = 3 if Dependents == "3+" else int(Dependents)


input_data = np.array([[gender_val, married_val, dependents_val, education_val,
                        self_employed_val, ApplicantIncome, CoapplicantIncome,
                        LoanAmount, Loan_Amount_Term, credit_val, property_val]])


if st.button("Predict"):
    try:
        scaled_data = scaler.transform(input_data)
        prediction = model.predict(scaled_data)[0]

        if prediction == 1:
            st.success("Loan Approved!")
        else:
            st.error("Loan Not Approved.")
    except Exception as e:
        st.error(f"Something went wrong: {e}")
