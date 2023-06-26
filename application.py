import streamlit as st
import pickle
import numpy as np

scaler = pickle.load(open("Model/standardScalar.pkl", "rb"))
model = pickle.load(open("Model/modelForPrediction.pkl", "rb"))

def predict_datapoint():
    result = ""

    pregnancies = st.text_input("Pregnancies", value="", key="pregnancies")
    glucose = st.text_input("Glucose", value="", key="glucose")
    blood_pressure = st.text_input("Blood Pressure", value="", key="blood_pressure")
    skin_thickness = st.text_input("Skin Thickness", value="", key="skin_thickness")
    insulin = st.text_input("Insulin", value="", key="insulin")
    bmi = st.text_input("BMI", value="", key="bmi")
    diabetes_pedigree_function = st.text_input("Diabetes Pedigree Function", value="", key="diabetes_pedigree_function")
    age = st.text_input("Age", value="", key="age")

    if st.button("Predict"):
        if pregnancies and glucose and blood_pressure and skin_thickness and insulin and bmi and diabetes_pedigree_function and age:
            pregnancies = float(pregnancies)
            glucose = float(glucose)
            blood_pressure = float(blood_pressure)
            skin_thickness = float(skin_thickness)
            insulin = float(insulin)
            bmi = float(bmi)
            diabetes_pedigree_function = float(diabetes_pedigree_function)
            age = float(age)

            new_data = scaler.transform([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age]])
            prediction = model.predict(new_data)

            if prediction[0] == 1:
                result = "Diabetic"
            else:
                result = "Non-Diabetic"

            st.write(f"Prediction: {result}")

def main():
    st.set_page_config(page_title="Diabetes Prediction App")
    st.title("Diabetes Prediction App")

    

    predict_datapoint()

if __name__ == "__main__":
    main()
