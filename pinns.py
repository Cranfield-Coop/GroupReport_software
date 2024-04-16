import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tempfile
from run_inference import run_inference

# Define the simulation function
def simulation(df):
    col = ["U_pred", "u'u'_pred", "v'v'_pred", "w'w'_pred", "u'v'_pred"]
    for i in range(5):
        plt.figure(figsize=(12, 6))
        subset = df[col[i]]
        if col[i] == "u'v'_pred":
            subset = -1 * df[col[i]]
        plt.plot(df["y^+"].values, subset.values, label=f"{col[i]} (PINNs)")
        plt.xscale('log')
        plt.xlabel('y^+')
        plt.ylabel(col[i])
        plt.title(f"{col[i]} as a function of y^+")
        plt.legend()
        st.pyplot(plt)

# Ensure Streamlit reruns when the selection changes
def on_model_change():
    # Reset y_max in session state when model changes
    st.session_state.y_max = float(selected_model)

# Function to prepare and update the CSV file based on user inputs
def prepare_prediction_csv(Re_tau, y_plus_min, y_plus_max, y_plus_delta, filename):
    data_dict = {
        5200: {"Re_tau": 5185.897, "u_tau": 4.14872e-02, "nu": 8.00000e-06},
        2000: {"Re_tau": 1994.756, "u_tau": 4.58794e-02, "nu": 2.30000e-05},
        1000: {"Re_tau": 1000.512, "u_tau": 5.00256e-02, "nu": 5.00000e-05},
        550: {"Re_tau": 543.496, "u_tau": 5.43496e-02, "nu": 1.00000e-04},
        180: {"Re_tau": 182.088, "u_tau": 6.37309e-02, "nu": 3.50000e-04},
    }

    if Re_tau not in data_dict:
        st.error("Invalid Re_tau value provided!")
        return None
    
    y_plus_values = np.arange(y_plus_min, y_plus_max + y_plus_delta, y_plus_delta)
    results = [{
        "y/delta": (y_plus * data_dict[Re_tau]["nu"]) / data_dict[Re_tau]["u_tau"],
        "y^+": y_plus,
        "u_tau": data_dict[Re_tau]["u_tau"],
        "nu": data_dict[Re_tau]["nu"],
        "Re_tau": data_dict[Re_tau]["Re_tau"]
    } for y_plus in y_plus_values]

    df = pd.DataFrame(results)
    df.to_csv(filename, index=False)
    return filename

# Main code
st.image("C:/Users/Louis/Desktop/Streamlit/cranfield_logo.png", width=100)
st.write("# Turbulence Modelling Predictor")
st.markdown('<p style="font-size: 15px; font-style: italic;"> ~Developed by Group 2 Cranfield CO-OP</p>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("**Please upload the CSV below.**", type=['csv'], key="file-uploader")
if uploaded_file is not None:
    st.success("File successfully uploaded!")
    model_options = ['5200', '2000', '1000', '550', '180']
    selected_model = st.selectbox("Select a Model (Re_tau)", model_options)
    y_min = st.number_input("Enter y_minimum (>0)", min_value=0.000000, value=0.1, format="%.6f")
    y_max = st.number_input("Enter y_maximum (0 to selected Re_tau)", min_value=0.000000, max_value=float(selected_model), value=min(0.1, float(selected_model)), format="%.6f")
    y_delta = st.number_input("Enter y_delta (>0)", min_value=0.00000, value=0.1, format="%.6f")

    if st.button('Run Model'):
        csv_filename = tempfile.mktemp(suffix='.csv')
        updated_csv_path = prepare_prediction_csv(int(selected_model), y_min, y_max, y_delta, csv_filename)
        if updated_csv_path:
            output_csv_path = tempfile.mktemp(suffix='.csv')
            model_checkpoint_path = "C:/Users/Louis/Desktop/Streamlit/epoch=10461-step=188316.ckpt"
            run_inference(model_checkpoint_path, updated_csv_path, output_csv_path)
            prediction_df = pd.read_csv(output_csv_path)
            st.write(prediction_df)
            simulation(prediction_df)  # Call simulation to plot results
            st.success("Model completed!")
            with open(output_csv_path, "rb") as f:
                st.download_button("Download Prediction Results", f, "prediction_output.csv")
