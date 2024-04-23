import streamlit as st
import pandas as pd
import streamlit as st
import pandas as pd
import tempfile
import time
from utils.PINNs.pinns import *

columns = ["y/delta", "y^+", "Re_tau","U","u'u'", "v'v'", "w'w'", "u'v'", "P", "dU/dy","nu","u_tau","k"]

def app():
    #st.image(logo_path, width=100)
    st.write("# Turbulence Modelling Predictor")
    st.markdown('<p style="font-size: 15px; font-style: italic;"> ~Developed by Group 2 Cranfield CO-OP</p>',unsafe_allow_html=True) 
    mode_option = st.radio("Select mode:", ("Inference Mode", "Test Mode"))

    if mode_option == "Inference Mode":
        uploaded_file = st.file_uploader(
            "Upload a CSV file with features or let the app generate it:", type=["csv"]
        )

        if uploaded_file is not None:
            # Step 1: User uploads a CSV file
            step1 = True
        else:
            # Allow user to generate CSV
            step1 = False
            selected_model = st.selectbox(
                "Select a Model (Re_tau):", ["5200", "2000", "1000", "550", "180"]
            )
            y_min = st.number_input("Enter y_minimum (>0):", value=0.1, format="%.6f")
            y_max = st.number_input(
                "Enter y_maximum (y_min to selected Re_tau):",
                min_value=y_min,
                max_value=float(selected_model),
                value=float(selected_model),
                format="%.6f",
            )
            y_delta = st.number_input("Enter y_delta (>0):", value=0.1, format="%.6f")

            if st.button("Run Inference"):
                csv_path = prepare_prediction_csv(
                    int(selected_model), y_min, y_max, y_delta
                )
                step1 = True

        if step1:
            # Step 2: Run Inference
            temp_csv_path = tempfile.mktemp(suffix=".csv")
            if isinstance(uploaded_file, str):
                csv_path = uploaded_file  # Generated CSV file path
            elif uploaded_file:
                csv_path = tempfile.mktemp(suffix=".csv")
                with open(csv_path, 'wb') as f:
                    f.write(uploaded_file.getvalue())

            run_inference(
                model_checkpoint_path="epoch=20332-step=731988.ckpt",
                prediction_dataset_path=csv_path,
                prediction_output_path=temp_csv_path,
            )

            # Step 3: Display Results
            pred_df = pd.read_csv(temp_csv_path)
            st.subheader("Prediction Values Table")
            st.dataframe(pred_df)
            display_plots(pred_df)
            st.download_button(
                "Download Prediction CSV",
                data=pd.read_csv(temp_csv_path).to_csv(index=False),
                file_name="prediction_output_pinns.csv",
            )

    elif mode_option == "Test Mode":
        uploaded_file_pinns = st.file_uploader(
            "Upload CSV file with features + target values:", type=["csv"]
        )

        if uploaded_file_pinns is not None:
            df = pd.read_csv(uploaded_file_pinns)
            if all(col in df.columns for col in columns):
                df = df[columns]
                Reynolds_Numbers = sorted(df["Re_tau"].unique(), reverse=True)
                if len(Reynolds_Numbers) > 1:
                    st.error("Please upload a csv file with a single Reynolds number.")
            else:
                st.error("Please upload a csv file with the columns y/delta, y^+, Re_tau, U, u'u', v'v', w'w', u'v', P, dU/dy, nu, u_tau and k to proceed.")
                return None
            if 'run_test' not in st.session_state:
                st.session_state.run_test = False
            if st.button("Run Test Inference"):
                st.session_state.run_test = True
            if st.session_state.run_test:
                timer_start = time.time()
                temp_csv_test_path = tempfile.mktemp(suffix=".csv")
                with open(temp_csv_test_path, "wb") as f:
                    f.write(uploaded_file_pinns.getvalue())
            

                # Step 2: Run Test Inference
                metrics = run_test_inference(
                    model_checkpoint_path="epoch=20332-step=731988.ckpt",
                    test_dataset_path=temp_csv_test_path,
                    prediction_output_path="output_test.csv",
                )
                timer_stop = time.time()
                # Step 3: Display Metrics and Prediction Results
                display_metrics(format_dataframe(pd.DataFrame(metrics)),timer_start,timer_stop)
                pred_test_df = pd.read_csv("output_test.csv")
                st.subheader("Prediction & Target Values Table")
                st.dataframe(pred_test_df)
                display_test_plots(pred_test_df)
                csv = pred_test_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Prediction Results",
                    data=csv,
                    file_name='prediction_output_pinns.csv',
                    mime='text/csv',
                )     
        else:
            st.error("Please upload a csv file to proceed.")


