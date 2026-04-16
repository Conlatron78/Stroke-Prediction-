import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("healthcare-dataset-stroke-data.csv")
df = df.drop(columns=["id"])

X = df.drop("stroke", axis=1)
y = df["stroke"]

numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
categorical_features = X.select_dtypes(include=["object"]).columns

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_features),
    ("cat", categorical_pipeline, categorical_features)
])

model = Pipeline([
    ("preprocessing", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced"))
])

model.fit(X, y)

st.title("Stroke Prediction System")
st.write("Enter patient details:")

gender = st.selectbox("Gender", ["Male", "Female", "Other"])
age = st.slider("Age", 0, 100, 30)
hypertension = st.selectbox("Hypertension", [0, 1])
heart_disease = st.selectbox("Heart Disease", [0, 1])
ever_married = st.selectbox("Ever Married", ["Yes", "No"])
work_type = st.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
residence_type = st.selectbox("Residence Type", ["Urban", "Rural"])
avg_glucose_level = st.slider("Average Glucose Level", 50.0, 300.0, 100.0)
bmi = st.slider("BMI", 10.0, 50.0, 25.0)
smoking_status = st.selectbox("Smoking Status", ["formerly smoked", "never smoked", "smokes", "Unknown"])

input_data = pd.DataFrame({
    "gender": [gender],
    "age": [age],
    "hypertension": [hypertension],
    "heart_disease": [heart_disease],
    "ever_married": [ever_married],
    "work_type": [work_type],
    "Residence_type": [residence_type],
    "avg_glucose_level": [avg_glucose_level],
    "bmi": [bmi],
    "smoking_status": [smoking_status]
})

if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.error(f"High Stroke Risk (Probability: {probability:.2f})")
    else:
        st.success(f"Low Stroke Risk (Probability: {probability:.2f})")
