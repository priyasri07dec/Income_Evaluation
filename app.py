import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle


# Load Saved Files
model = joblib.load("xgb_model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")
model_columns = joblib.load("model_columns.pkl")

# Load the dataset for dropdown options
df = pd.read_csv("income_evaluation.csv")
# Replace ? with NaN
df.replace("?", np.nan, inplace=True)
df.replace(" ?", np.nan, inplace=True)

#remove leading and trailing spaces from column names
df.columns = df.columns.str.strip()

#filling missing categorical columns
categorical_cols = df.select_dtypes(include='object').columns

# Then filling missing values with the mode for categorical columns
for column in categorical_cols:
       df[column] = df[column].fillna(df[column].mode()[0])



#streamlit page configuration

st.set_page_config(page_title="Income Evaluation", page_icon="💰", layout="wide")

# image
st.image("money.png", width=100)

# Title

st.title("Income Evaluation App")
st.write("Predict whether a person's income is <=50K or >50K")

#sidebar
st.sidebar.header("About the Model")
st.sidebar.write("This model predicts whether a person's income is <=50K or >50K based on various features. The model is trained using XGBoost algorithm.")
st.sidebar.write("Please provide the required information in the input fields and click on the 'Predict' button to see the result.")

#background color
st.markdown(
    """
    <style>
    .stApp {
        background-color: #ADD8E6;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# User Input
# Creating two columns

col1, col2, col3 = st.columns(3)

with col1:
  
  workclass = st.selectbox("Workclass",df["workclass"].unique())
  fnlwgt = st.number_input("Final Weight", min_value=0, value=100000)
  education = st.selectbox("Education", df["education"].unique())
  education_num = st.number_input("Education Number", min_value=0, value=10)

with col2:
  marital_status = st.selectbox("Marital Status", df["marital-status"].unique())
  occupation = st.selectbox("Occupation", df["occupation"].unique())
  race = st.selectbox("Race", df["race"].unique())
  sex = st.selectbox("Sex", df["sex"].unique())
  
with col3:
  capital_gain = st.number_input("Capital Gain", min_value=0, value=0)
  capital_loss = st.number_input("Capital Loss", min_value=0, value=0)
  hours_per_week = st.number_input("Hours Per Week", min_value=0, value=40)
  native_country = st.selectbox("Native Country", df["native-country"].unique())

#Create Input Dictionary
input_data = {
    
    "workclass": workclass,
    "fnlwgt": fnlwgt,
    "education": education,
    "education_num": education_num,
    "marital_status": marital_status,
    "occupation": occupation,
    "race": race,
    "sex": sex,
    "capital_gain": capital_gain,
    "capital_loss": capital_loss,
    "hours_per_week": hours_per_week,
    "native_country": native_country
}

# Convert to DataFrame
input_df = pd.DataFrame([input_data])

# Preprocess Input
for col in model_columns:
    if col in input_df.columns:
        if input_df[col].dtype == "object":
            input_df[col] = label_encoder.transform(input_df[col])
    else:
        input_df[col] = 0

#correct column order
input_df = input_df[model_columns]

# Scale Input
input_scaled = scaler.transform(input_df)

# Predict
prediction = model.predict(input_scaled)

# Decode Prediction
result = label_encoder.inverse_transform(prediction)

#show result
if st.button("Predict"):
    st.success(f"Predicted Income: {result[0]}")