import pandas as pd
import numpy as np
import streamlit as st
import pickle

# -----------------------------
# Parkinson's Disease Tab
# -----------------------------
def Parkinson_tab():
    st.subheader("Parkinsons Disease Prediction")

    with open(r"D:\guvi\Projects\.venv\project_4_multiple_disease_prediction\p_final_model.pkl", "rb") as f:
        p_model = pickle.load(f)

    with open(r"D:\guvi\Projects\.venv\project_4_multiple_disease_prediction\p_scaler.pkl", "rb") as f:
        p_scaler = pickle.load(f)


    # Input features
    MDVP_Fo_Hz = st.number_input("MDVP:Fo(Hz)", min_value=70.000, max_value=300.000, step=1.000) 
    MDVP_Flo_Hz = st.number_input("MDVP:Flo(Hz)", min_value=50.000, max_value=270.000, step=1.000) 
    MDVP_Jitter_Abs = st.number_input("MDVP:Jitter(Abs)", min_value=0.00001, max_value=0.00030, step=0.00001) 
    MDVP_Shimmer = st.number_input("MDVP:Shimmer", min_value=0.00500, max_value=0.20000, step=0.00100) 
    MDVP_Shimmer_dB = st.number_input("MDVP:Shimmer(dB)", min_value=0.050, max_value=2.000, step=0.010) 
    Shimmer_APQ3 = st.number_input("Shimmer:APQ3", min_value=0.050, max_value=2.000, step=0.010) 
    Shimmer_APQ5 = st.number_input("Shimmer:APQ5", min_value=0.00200, max_value=1.00000, step=0.00100) 
    MDVP_APQ = st.number_input("MDVP:APQ", min_value=0.00500, max_value=0.20000, step=0.00100) 
    Shimmer_DDA = st.number_input("Shimmer:DDA", min_value=0.01000, max_value=0.20000, step=0.01000) 
    HNR = st.number_input("HNR", min_value=5.000, max_value=40.000, step=0.500) 
    RPDE = st.number_input("RPDE", min_value=0.200000, max_value=1.000000, step=0.050000) 
    spread1 = st.number_input("spread1", min_value=-9.000000, max_value=-1.000000, step=0.500000) 
    spread2 = st.number_input("spread2", min_value=0.000000, max_value=0.600000, step=0.010000) 
    D2 = st.number_input("D2", min_value=1.000000, max_value=4.000000, step=0.500000) 
    PPE = st.number_input("PPE", min_value=0.000000, max_value=0.700000, step=0.010000)

    user_df = pd.DataFrame({
        'MDVP:Fo(Hz)':[MDVP_Fo_Hz],
        'MDVP:Flo(Hz)':[MDVP_Flo_Hz],
        'MDVP:Jitter(Abs)':[MDVP_Jitter_Abs],
        'MDVP:Shimmer':[MDVP_Shimmer],
        'MDVP:Shimmer(dB)':[MDVP_Shimmer_dB],
        'Shimmer:APQ3':[Shimmer_APQ3],
        'Shimmer:APQ5':[Shimmer_APQ5],
        'MDVP:APQ':[MDVP_APQ],
        'Shimmer:DDA':[Shimmer_DDA],
        'HNR':[HNR],
        'RPDE':[RPDE],
        'spread1':[spread1],
        'spread2':[spread2],
        'D2':[D2],
        'PPE':[PPE]
    })

    if st.button("Predict", key="predict1"):
        result = make_prediction(user_df, model=p_model, scaler=p_scaler)
        output = "Yes" if result == 1 else "No"
        st.success(f"Parkinsons Disease Prediction: {output}")

# -----------------------------
# Kidney Disease Tab
# -----------------------------
def Kidney_tab():
    st.subheader("Kidney Disease Prediction")

    with open(r"D:\guvi\Projects\.venv\project_4_multiple_disease_prediction\k_final_model.pkl", "rb") as f:
        k_model = pickle.load(f)

    with open(r"D:\guvi\Projects\.venv\project_4_multiple_disease_prediction\k_scaler.pkl", "rb") as f:
        k_scaler = pickle.load(f)

    with open(r"D:\guvi\Projects\.venv\project_4_multiple_disease_prediction\k_encoder.pkl", "rb") as f:
        k_encoder = pickle.load(f)

    blood_pressure = st.number_input("blood_pressure", min_value=50, max_value=150, step=1) 
    specific_gravity = st.number_input("specific_gravity", min_value=0.800, max_value=1.200, step=0.010) 
    albumin = st.number_input("albumin", min_value=0, max_value=5, step=1) 
    sugar = st.number_input("sugar", min_value=0, max_value=5, step=1) 
    red_blood_cells = st.selectbox("red_blood_cells", ["normal", "abnormal"]) 
    pus_cell = st.selectbox("pus_cell", ["normal", "abnormal"]) 
    pus_cell_clumps = st.selectbox("pus_cell_clumps", ["present", "notpresent"]) 
    blood_glucose_random = st.number_input("blood_glucose_random", min_value=0, max_value=600, step=5) 
    blood_urea = st.number_input("blood_urea", min_value=0, max_value=500, step=5) 
    serum_creatinine = st.number_input("serum_creatinine", min_value=0.0, max_value=80.0, step=0.1) 
    sodium = st.number_input("sodium", min_value=0, max_value=200, step=5) 
    haemoglobin = st.number_input("haemoglobin", min_value=0.0, max_value=20.0, step=0.1) 
    packed_cell_volume = st.number_input("packed_cell_volume", min_value=0, max_value=60, step=1) 
    red_blood_cell_count = st.number_input("red_blood_cell_count", min_value=0.0, max_value=10.0, step=0.1) 
    hypertension = st.selectbox("hypertension", ["yes", "no"]) 
    diabetes_mellitus = st.selectbox("diabetes_mellitus", ["yes", "no"]) 
    appetite = st.selectbox("appetite", ["good", "poor"]) 
    peda_edema = st.selectbox("peda_edema", ["yes", "no"]) 
    anemia = st.selectbox("anemia", ["yes", "no"])

    user_df = pd.DataFrame({
        'blood_pressure':[blood_pressure],
        'specific_gravity':[specific_gravity],
        'albumin':[albumin],
        'sugar':[sugar],
        'red_blood_cells':[red_blood_cells],
        'pus_cell':[pus_cell],
        'pus_cell_clumps':[pus_cell_clumps],
        'blood_glucose_random':[blood_glucose_random],
        'blood_urea':[blood_urea],
        'serum_creatinine':[serum_creatinine],
        'sodium':[sodium],
        'haemoglobin':[haemoglobin],
        'packed_cell_volume':[packed_cell_volume],
        'red_blood_cell_count':[red_blood_cell_count],
        'hypertension':[hypertension],
        'diabetes_mellitus':[diabetes_mellitus],
        'appetite':[appetite],
        'peda_edema':[peda_edema],
        'anemia':[anemia]  
    })

    if st.button("Predict", key="predict2"):
        result = make_prediction(user_df, model=k_model, scaler=k_scaler, encoder=k_encoder)
        output = "Yes" if result == 1 else "No"
        st.success(f"Kidney Disease Prediction: {output}")

# -----------------------------
# Liver Disease Tab
# -----------------------------
def Liver_tab():
    st.subheader("Liver Disease Prediction")

    with open(r"D:\guvi\Projects\.venv\project_4_multiple_disease_prediction\l_final_model.pkl", "rb") as f:
        l_model = pickle.load(f)

    with open(r"D:\guvi\Projects\.venv\project_4_multiple_disease_prediction\l_scaler.pkl", "rb") as f:
        l_scaler = pickle.load(f)

    Age = st.number_input("Age", min_value=1, max_value=100, step=1) 
    Total_Bilirubin = st.number_input("Total_Bilirubin", min_value=0.0, max_value=80.0, step=0.1) 
    Direct_Bilirubin = st.number_input("Direct_Bilirubin", min_value=0.0, max_value=25.0, step=0.1) 
    Alkaline_Phosphotase = st.number_input("Alkaline_Phosphotase", min_value=50, max_value=2500, step=5) 
    Alamine_Aminotransferase = st.number_input("Alamine_Aminotransferase", min_value=0, max_value=2500, step=5) 
    Aspartate_Aminotransferase = st.number_input("Aspartate_Aminotransferase", min_value=0, max_value=5000, step=5) 
    Albumin = st.number_input("Albumin", min_value=0.0, max_value=10.0, step=0.1) 
    Albumin_and_Globulin_Ratio = st.number_input("Albumin_and_Globulin_Ratio", min_value=0.00, max_value=5.00, step=0.10)

    user_df = pd.DataFrame({
        'Age':[Age],
        'Total_Bilirubin':[Total_Bilirubin],
        'Direct_Bilirubin':[Direct_Bilirubin],
        'Alkaline_Phosphotase':[Alkaline_Phosphotase],
        'Alamine_Aminotransferase':[Alamine_Aminotransferase],
        'Aspartate_Aminotransferase':[Aspartate_Aminotransferase],
        'Albumin':[Albumin],
        'Albumin_and_Globulin_Ratio':[Albumin_and_Globulin_Ratio]
    })

    if st.button("Predict", key="predict3"):
        result = make_prediction(user_df, model=l_model, scaler=l_scaler)
        output = "Yes" if result == 1 else "No"
        st.success(f"Liver Disease Prediction: {output}")

# -----------------------------
# Main App
# -----------------------------
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
