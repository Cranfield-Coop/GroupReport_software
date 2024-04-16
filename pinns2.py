import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tempfile
import os
from run_inference import run_inference
from prepare_prediction_csv import prepare_prediction_csv

# Define the simulation function
def simulation(df, dns_df=None):
    col = ["U_pred", "u'u'_pred", "v'v'_pred", "w'w'_pred", "u'v'_pred"]
    for i in range(5):
        plt.figure(figsize=(12, 6))
        subset = df[col[i]]
        if col[i] == "u'v'_pred":
            subset = -1 * df[col[i]]
        plt.plot(df["y^+"].values, subset.values, label=f"{col[i]} (PINNs)", marker='o')
        if dns_df is not None and col[i] in dns_df.columns:
            plt.plot(dns_df["y^+"].values, dns_df[col[i]].values, label=f"{col[i]} (DNS)", linestyle='--')
        plt.xscale('log')
        plt.xlabel('y^+')
        plt.ylabel(col[i])
        plt.title(f"{col[i]} as a function of y^+")
        plt.legend()
        st.pyplot(plt)
        plt.close()  # Close the figure after rendering to free up memory

# Main code
st.image("C:/Users/Louis/Desktop/Streamlit/cranfield_logo.png", width=100)
st.write("# Turbulence Modelling Predictor")
st.markdown('<p style="font-size: 15px; font-style: italic;"> ~Developed by Group 2 Cranfield CO-OP</p>', unsafe_allow_html=True)

data_source_option = st.radio("Select your data source:", ("Use existing library", "Upload new CSV file"))

model_options = ['5200', '2000', '1000', '550', '180']
selected_model = st.selectbox("Select a Model (Re_tau)", model_options)

y_min = st.number_input("Enter y_minimum (>0)", min_value=0.000000, value=0.1, format="%.6f")
y_max = st.number_input("Enter y_maximum (y_min to selected Re_tau)", min_value=0.000000, max_value=float(selected_model), value=min(0.2, float(selected_model)), format="%.6f")
y_delta = st.number_input("Enter y_delta (>0)", min_value=0.00000, value=0.1, format="%.6f")

uploaded_file = None
if data_source_option == "Upload new CSV file":
    uploaded_file = st.file_uploader("**Please upload the CSV below.**", type=['csv'], key="file-uploader")

if y_max <= y_min:
    st.error("Error: y_max value must be greater than y_min value.")
else:
    if st.button('Run Model'):
    
        model_checkpoint_path = "C:/Users/Louis/Desktop/Streamlit/epoch=10461-step=188316.ckpt"  # Define path only once
        dns_data_path = "C:/Users/Louis/Desktop/Streamlit/DNS_data.csv"  # Path to DNS data if needed for plotting
        output_csv_path = tempfile.mktemp(suffix='.csv')  # Define the output path for predictions here
        
        # Initialize the progress bar
        st.write ("Running in progress, please wait.")
        progress_bar = st.progress(0)
        
        if uploaded_file:
            uploaded_csv_path = tempfile.mktemp(suffix='.csv')
            with open(uploaded_csv_path, 'wb') as f:
                f.write(uploaded_file.getvalue())
            
            output_csv_path = tempfile.mktemp(suffix='.csv')
            
            # Simulate updating progress during processing
            for i in range(10):
                # Update progress bar by incrementing by 10% each iteration
                progress_bar.progress((i+1) * 10)
                mse, rmse = run_inference(model_checkpoint_path, uploaded_csv_path, output_csv_path)
            
            prediction_df = pd.read_csv(output_csv_path)
            dns_df = pd.read_csv(dns_data_path)
            simulation(prediction_df, dns_df)
            st.success("Model completed with uploaded file!")
            with open(output_csv_path, "rb") as f:
                st.download_button("Download Prediction Results", f, "prediction_output.csv")
        else:
            csv_filename = tempfile.mktemp(suffix='.csv')
            updated_csv_path = prepare_prediction_csv(int(selected_model), y_min, y_max, y_delta, csv_filename)
            
            for i in range(10):
                # Assume each step in your process can be quantified such that the loop reflects progress
                progress_bar.progress((i+1) * 10)
                mse, rmse = run_inference(model_checkpoint_path, updated_csv_path, output_csv_path)
            
            prediction_df = pd.read_csv(output_csv_path)
            dns_df = pd.read_csv(dns_data_path)
            simulation(prediction_df, dns_df)
            st.success("Model completed with generated data!")
            with open(output_csv_path, "rb") as f:
                st.download_button("Download Prediction Results", f, "prediction_output.csv")
        
        # Complete the progress bar when done
        progress_bar.progress(100)
