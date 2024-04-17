import time
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pysindy as ps
import seaborn as sns
import sys
from io import StringIO
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from utils.psindy import *
# import tempfile
# from run_inference import *
# from pinns import *


logo_path = "img/cranfield_logo.png"
columns = ["y^+", "U", "Re_tau", "u'u'", "v'v'", "w'w'", "u'v'", "P", "dU/dy"]


def main():
    st.image(logo_path, width=100)
    st.write("# Turbulence Modelling Predictor")
    st.markdown('<p style="font-size: 15px; font-style: italic;"> ~Developed by Group 2 Cranfield CO-OP</p>',
                unsafe_allow_html=True)
    noise_level = st.slider(
        'Before uploading, select the noise level (%)', 0, 50, 0)
    uploaded_file = st.file_uploader(
        "\n\n**Please upload the CSV below.**", type=['csv'], key="file-uploader")
    if uploaded_file is not None:
        st.success("File successfully uploaded!")
        df = pd.read_csv(uploaded_file)
        # Add noise to the dataframe
        print(
            f"Adding noise to the dataframe with noise level: {noise_level}%")
        df_noise = df.drop(columns='Re_tau').applymap(
            lambda x: x + np.random.normal(0, noise_level/100))
        df_noise['Re_tau'] = df['Re_tau']
        df = df_noise.copy()

        st.write("**Please choose a model:**")
        model_choice = st.radio("Please choose a model:", ('PySINDy', 'PINNs'))
        if model_choice == 'PySINDy':
            run_pysindy(df)
        elif model_choice == 'PINNs':
            run_pinns()
        else:
            st.write("Please choose a model to get started.")
    else:
        st.error("Please upload a csv file to proceed.")


def run_pysindy(df):
    st.header("PySINDy Model")
    if all(col in df.columns for col in columns):
        df = df[columns]
    else:
        st.error(
            "Please upload a csv file with the columns y^+, U, Re_tau, u'u', v'v', w'w', u'v', P and dU/dy to proceed.")
        return None
    Reynolds_Numbers = sorted(df["Re_tau"].unique(), reverse=True)
    st.subheader("Data processing")
    mode_choice = st.selectbox("Choose processing mode:", [
                               'Data with interpolation', 'Data without interpolation'])
    if mode_choice == 'Data with interpolation':
        nb_interpolation = st.number_input("Please enter the number of data points to be generated with interpolation:",
                                           min_value=5000, max_value=10000, step=1000, key='nb_interp')
        data = data_processing(df, 1, nb_interpolation, Reynolds_Numbers)
    elif mode_choice == "Data without interpolation":
        data = data_processing(df, 0, 0, Reynolds_Numbers)
    if data is not None:
        if 'show_graph' not in st.session_state:
            st.session_state.show_graph = False
        if st.button('Show raw data visualizations'):
            st.session_state.show_graph = True
        if st.button('Hide raw data visualizations'):
            st.session_state.show_graph = False
        if st.session_state.show_graph:
            df = pd.concat(data.values(), ignore_index=True)
            plot(df, Reynolds_Numbers)
        X, y = data_PySINDy(data, Reynolds_Numbers)
        model = {}
        feature_names = ["U", "u'u'", "v'v'", "w'w'", "u'v'", "P",
                         "dU/dy", "du'u'/dy", "dv'v'/dy", "dw'w'/dy", "du'v'/dy", "dP/dy"]
        feature_library = ps.PolynomialLibrary(1)
        st.subheader("Selection of PySINDy hyperparameters")
        threshold_value = st.number_input(
            "Please enter the threshold value for the PySINDy optimiser:", value=0.001, min_value=0.0, format="%.15f")
        alpha_coeff = st.number_input(
            "Please enter the alpha coefficient for the PySINDy optimiser:", value=0.01, min_value=0.0, format="%.15f")
        if 'run_model' not in st.session_state:
            st.session_state.run_model = False

        if st.button('Run Model'):
            st.session_state.run_model = True

        if st.session_state.run_model:
            timer_start = time.time()
            for i in range(0, len(Reynolds_Numbers)):
                optimizer = ps.STLSQ(
                    threshold=threshold_value, alpha=alpha_coeff)
                model[Reynolds_Numbers[i]] = ps.SINDy(
                    feature_library=feature_library, optimizer=optimizer, feature_names=feature_names)
                model[Reynolds_Numbers[i]].fit(X[i], x_dot=y[i])
                model[Reynolds_Numbers[i]].print()

            nb_model = 0
            if len(Reynolds_Numbers) > 1:
                X_train = X[0]
                y_train = y[0]
                X_test = X[1]
                y_test = y[1]
            else:
                ratio = st.number_input(
                    "Please enter the ratio to split the data into training and testing datasets:", value=0.8, format="%.2f")
                if ratio < 0 or ratio > 1:
                    st.error(
                        "Please enter a positive value between 0 and 1 for the ratio!")
                else:
                    X = X[0]
                    y = y[0]
                    ratio_split = int(len(X) * ratio)
                    X_train = X[:ratio_split]
                    X_test = X[ratio_split:]
                    y_train = y[:ratio_split]
                    y_test = y[ratio_split:]
            timer_stop = time.time()
            st.subheader("PySINDy Output")
            old_stdout = sys.stdout
            sys.stdout = buffer = StringIO()
            model[Reynolds_Numbers[0]].print()
            sys.stdout = old_stdout
            model_output = buffer.getvalue()
            st.write("PySINDy Model - Equations")
            st.text(model_output)
            st.metric("Excecution time",
                      value=f"{timer_stop - timer_start:.2f} seconds")
            print(f"Execution time: {timer_stop - timer_start:.2f} seconds")

            sparsity = 0
            for i in range(len(model[Reynolds_Numbers[nb_model]].optimizer.coef_)):
                if np.all(model[Reynolds_Numbers[nb_model]].optimizer.coef_[i] == 0):
                    sparsity = -1
            if sparsity == -1:
                st.error(
                    f"Sparsity parameter is too big ({threshold_value}) and eliminated all coefficients!")
            else:
                y_pred = model[Reynolds_Numbers[nb_model]].predict(X_test)
                y_true = y_test
                r2 = r2_score(y_true, y_pred)
                print(f"R-square: {r2}")
                mse = mean_squared_error(y_true, y_pred)
                print(f"MSE: {mse}")

                st.metric(label="R-Squared", value=f"{r2:.10f}")
                st.metric(label="Mean Squared Error",
                          value=f"{mse:.10f}")

                test_overfitting(X_train, y_train,
                                 X_test, y_test, model, Reynolds_Numbers, nb_model)

                st.subheader("PySINDy Simulation")
                if len(Reynolds_Numbers) > 1:
                    solution = simulation(
                        X_test, data[f"df_{Reynolds_Numbers[1]}"], model[Reynolds_Numbers[nb_model]])
                else:
                    solution = simulation(
                        X_test, data[f"df_{Reynolds_Numbers[0]}"], model[Reynolds_Numbers[nb_model]])
                if 'simulation' not in st.session_state:
                    st.session_state.simulation = False

                if st.button('Show PySINDy Model Simulation'):
                    st.session_state.simulation = True

                if st.button('Hide PySINDy Model Simulation'):
                    st.session_state.simulation = False

                if st.session_state.simulation:
                    if len(Reynolds_Numbers) > 1:
                        simulation_plot(
                            data[f"df_{Reynolds_Numbers[1]}"], solution)
                        df_sim = pd.DataFrame(
                            {"y^+": solution.t, 'U': solution.y[0], "u'u'": solution.y[1], "v'v'": solution.y[2], "w'w'": solution.y[3], "u'v'": solution.y[4]})
                        df_pred = data[f"df_{Reynolds_Numbers[1]}"][data[f"df_{Reynolds_Numbers[1]}"]['y^+'].isin(
                            solution.t)]
                        mae_sim = mean_absolute_error(df_pred[["U", "u'u'", "v'v'", "w'w'", "u'v'"]].values, df_sim[[
                                                      "U", "u'u'", "v'v'", "w'w'", "u'v'"]].values)
                    else:
                        simulation_plot(
                            data[f"df_{Reynolds_Numbers[0]}"], solution)
                        df_sim = pd.DataFrame(
                            {"y^+": solution.t, 'U': solution.y[0], "u'u'": solution.y[1], "v'v'": solution.y[2], "w'w'": solution.y[3], "u'v'": solution.y[4]})
                        df_pred = data[f"df_{Reynolds_Numbers[1]}"][data[f"df_{Reynolds_Numbers[1]}"]['y^+'].isin(
                            solution.t)]
                        mae_sim = mean_absolute_error(df_pred[["U", "u'u'", "v'v'", "w'w'", "u'v'"]].values, df_sim[[
                                                      "U", "u'u'", "v'v'", "w'w'", "u'v'"]].values)
                    if mae_sim < 1:
                        st.write(
                            "The PySINDy model successfully generates equations with physical meaning, and the model simulation is highly accurate.")
                    elif mae_sim < 5:
                        st.write(
                            "The PySINDy model successfully generates equations with physical meaning, and the model simulation is quite accurate.")
                    elif mae_sim > 5:
                        st.write(
                            "The PySINDy model is unable to generate equations with physical meaning, and the simulation of the model is not very accurate.")
                    csv = df_sim.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download Prediction Results",
                        data=csv,
                        file_name='prediction_output.csv',
                        mime='text/csv',
                    )

                st.subheader("PySINDy Prediction")
                cols = ["dU/dy", "du'u'/dy", "dv'v'/dy", "dw'w'/dy", "du'v'/dy", "dP/dy",
                        "d(dU/dy)/dy", "d(du'u'/dy)/dy", "d(dv'v'/dy)/dy", "d(dw'w'/dy)/dy", "d(du'v'/dy)/dy", "d(dP/dy)/dy"]
                if len(Reynolds_Numbers) > 1:
                    y_plus = data[f"df_{Reynolds_Numbers[1]}"]["y^+"].to_numpy()
                else:
                    y_plus = data[f"df_{Reynolds_Numbers[0]}"]["y^+"].to_numpy()

                if 'prediction' not in st.session_state:
                    st.session_state.prediction = False

                if st.button('Show PySINDy Model Prediction'):
                    st.session_state.prediction = True

                if st.button('Hide PySINDy Model Prediction'):
                    st.session_state.prediction = False

                if st.session_state.prediction:
                    for i in range(12):
                        plt.figure(figsize=(12, 6))
                        plt.plot(
                            y_plus, y_true[:, i], "k", label="Real value")
                        plt.plot(
                            y_plus, y_pred[:, i], "r--", label="model prediction")
                        plt.xscale("log")
                        plt.xlabel('y^+')
                        plt.ylabel(cols[i])
                        plt.legend()
                        st.pyplot(plt)


def run_pinns():
    model_options = ['5200', '2000', '1000', '550', '180']
    selected_model = st.selectbox("Select a Model (Re_tau)", model_options)
    y_min = st.number_input("Enter y_minimum (>0)",
                            min_value=0.000000, value=0.1, format="%.6f")
    y_max = st.number_input("Enter y_maximum (0 to selected Re_tau)", min_value=0.000000, max_value=float(
        selected_model), value=min(0.1, float(selected_model)), format="%.6f")
    y_delta = st.number_input("Enter y_delta (>0)",
                              min_value=0.00000, value=0.1, format="%.6f")
    if st.button('Run Model'):
        csv_filename = tempfile.mktemp(suffix='.csv')
        updated_csv_path = prepare_prediction_csv(
            int(selected_model), y_min, y_max, y_delta, csv_filename)
        if updated_csv_path:
            output_csv_path = tempfile.mktemp(suffix='.csv')
            model_checkpoint_path = "epoch=10461-step=188316.ckpt"
            run_inference(model_checkpoint_path,
                          updated_csv_path, output_csv_path)
            prediction_df = pd.read_csv(output_csv_path)
            st.write(prediction_df)
            simulation_pinns(prediction_df)  # Call simulation to plot results
            st.success("Model completed!")
            with open(output_csv_path, "rb") as f:
                st.download_button(
                    "Download Prediction Results", f, "prediction_output.csv")


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

    y_plus_values = np.arange(
        y_plus_min, y_plus_max + y_plus_delta, y_plus_delta)
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


if __name__ == "__main__":
    main()
