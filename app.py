import pandas as pd
import numpy as np
import streamlit as st
import pickle

# -----------------------------
# Unified Prediction Function
# -----------------------------
def make_prediction(df, model, scaler=None, encoder=None):
    try:
        # Encode categorical features using LabelEncoders
        if encoder and isinstance(encoder, dict):
            for col in df.select_dtypes(include='object').columns:
                if col in encoder:
                    # Ensure values are strings and reshape properly
                    df[col] = encoder[col].transform(df[col].astype(str).values.ravel())
                else:
                    raise ValueError(f"Encoder missing for column: {col}")

        # Scale numerical features
        if scaler:
            df_scaled = scaler.transform(df)
        else:
            df_scaled = df

        # Predict
        prediction = model.predict(df_scaled)
        return prediction[0]
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        return None

  

# -----------------------------
# Parkinson's Disease Tab
# -----------------------------
def Parkinson_tab():
    st.subheader("Parkinson's Disease Prediction")

    p_model = pickle.load(open(r"D:\guvi\Projects\.venv\project_4_multiple_disease_prediction\Parkinson\p_final_model.pkl", "rb"))
    p_scaler = pickle.load(open(r"D:\guvi\Projects\.venv\project_4_multiple_disease_prediction\Parkinson\p_scaler.pkl", "rb"))

    # Input features
    features = {
        "MDVP:Fo(Hz)": st.number_input("MDVP:Fo(Hz)", 70.0, 300.0, step=1.0),
        "MDVP:Flo(Hz)": st.number_input("MDVP:Flo(Hz)", 50.0, 270.0, step=1.0),
        "MDVP:Jitter(Abs)": st.number_input("MDVP:Jitter(Abs)", 0.00001, 0.00030, step=0.00001),
        "MDVP:Shimmer": st.number_input("MDVP:Shimmer", 0.005, 0.2, step=0.001),
        "MDVP:Shimmer(dB)": st.number_input("MDVP:Shimmer(dB)", 0.05, 2.0, step=0.01),
        "Shimmer:APQ3": st.number_input("Shimmer:APQ3", 0.05, 2.0, step=0.01),
        "Shimmer:APQ5": st.number_input("Shimmer:APQ5", 0.002, 1.0, step=0.001),
        "MDVP:APQ": st.number_input("MDVP:APQ", 0.005, 0.2, step=0.001),
        "Shimmer:DDA": st.number_input("Shimmer:DDA", 0.01, 0.2, step=0.01),
        "HNR": st.number_input("HNR", 5.0, 40.0, step=0.5),
        "RPDE": st.number_input("RPDE", 0.2, 1.0, step=0.05),
        "spread1": st.number_input("spread1", -9.0, -1.0, step=0.5),
        "spread2": st.number_input("spread2", 0.0, 0.6, step=0.01),
        "D2": st.number_input("D2", 1.0, 4.0, step=0.5),
        "PPE": st.number_input("PPE", 0.0, 0.7, step=0.01)
    }

    user_df = pd.DataFrame([features])

    if st.button("Predict", key="predict1"):
        result = make_prediction(user_df, model=p_model, scaler=p_scaler)
        if result is not None:
            st.success(f"Parkinson's Disease Prediction: {'Yes' if result == 1 else 'No'}")

# -----------------------------
# Kidney Disease Tab
# -----------------------------
def Kidney_tab():
    st.subheader("Kidney Disease Prediction")

    k_model = pickle.load(open(r"D:\guvi\Projects\.venv\project_4_multiple_disease_prediction\kidney\k_final_model.pkl", "rb"))
    k_scaler = pickle.load(open(r"D:\guvi\Projects\.venv\project_4_multiple_disease_prediction\kidney\k_scaler.pkl", "rb"))
    k_encoder = pickle.load(open(r"D:\guvi\Projects\.venv\project_4_multiple_disease_prediction\kidney\k_encoder.pkl", "rb"))

    features = {
        "blood_pressure": st.number_input("blood_pressure", 50, 150, step=1),
        "specific_gravity": st.number_input("specific_gravity", 0.8, 1.2, step=0.01),
        "albumin": st.number_input("albumin", 0, 5, step=1),
        "sugar": st.number_input("sugar", 0, 5, step=1),
        "red_blood_cells": st.selectbox("red_blood_cells", ["normal", "abnormal"]),
        "pus_cell": st.selectbox("pus_cell", ["normal", "abnormal"]),
        "pus_cell_clumps": st.selectbox("pus_cell_clumps", ["present", "notpresent"]),
        "blood_glucose_random": st.number_input("blood_glucose_random", 0, 600, step=5),
        "blood_urea": st.number_input("blood_urea", 0, 500, step=5),
        "serum_creatinine": st.number_input("serum_creatinine", 0.0, 80.0, step=0.1),
        "sodium": st.number_input("sodium", 0, 200, step=5),
        "haemoglobin": st.number_input("haemoglobin", 0.0, 20.0, step=0.1),
        "packed_cell_volume": st.number_input("packed_cell_volume", 0, 60, step=1),
        "red_blood_cell_count": st.number_input("red_blood_cell_count", 0.0, 10.0, step=0.1),
        "hypertension": st.selectbox("hypertension", ["yes", "no"]),
        "diabetes_mellitus": st.selectbox("diabetes_mellitus", ["yes", "no"]),
        "appetite": st.selectbox("appetite", ["good", "poor"]),
        "peda_edema": st.selectbox("peda_edema", ["yes", "no"]),
        "anemia": st.selectbox("anemia", ["yes", "no"])
    }

    user_df = pd.DataFrame([features])

    if st.button("Predict", key="predict2"):
        result = make_prediction(user_df, model=k_model, scaler=k_scaler, encoder=k_encoder)
        if result is not None:
            st.success(f"Kidney Disease Prediction: {'No' if result == 1 else 'Yes'}")

# -----------------------------
# Liver Disease Tab
# -----------------------------
def Liver_tab():
    st.subheader("Liver Disease Prediction")

    l_model = pickle.load(open(r"D:\guvi\Projects\.venv\project_4_multiple_disease_prediction\Liver\l_final_model.pkl", "rb"))
    l_scaler = pickle.load(open(r"D:\guvi\Projects\.venv\project_4_multiple_disease_prediction\Liver\l_scaler.pkl", "rb"))

    features = {
        "Age": st.number_input("Age", 1, 100, step=1),
        "Total_Bilirubin": st.number_input("Total_Bilirubin", 0.0, 80.0, step=0.1),
        "Direct_Bilirubin": st.number_input("Direct_Bilirubin", 0.0, 25.0, step=0.1),
        "Alkaline_Phosphotase": st.number_input("Alkaline_Phosphotase", 50, 2500, step=5),
        "Alamine_Aminotransferase": st.number_input("Alamine_Aminotransferase", 0, 2500, step=5),
        "Aspartate_Aminotransferase": st.number_input("Aspartate_Aminotransferase", 0, 5000, step=5),
        "Albumin": st.number_input("Albumin", 0.0, 10.0, step=0.1),
        "Albumin_and_Globulin_Ratio": st.number_input("Albumin_and_Globulin_Ratio", 0.0, 5.0, step=0.1)
    }

    user_df = pd.DataFrame([features])

    if st.button("Predict", key="predict3"):
        result = make_prediction(user_df, model=l_model, scaler=l_scaler)
        if result is not None:
            st.success(f"Liver Disease Prediction: {'No' if result == 1 else 'Yes'}")


def main():
    st.title("Multiple Disease Prediction Dashboard")

    tab1, tab2, tab3 = st.tabs(["📊 Parkinson's", "📈 Kidney", "📉 Liver"])

    with tab1:
        Parkinson_tab()
    with tab2:
        Kidney_tab()
    with tab3:
        Liver_tab()

if __name__ == "__main__":
    main()