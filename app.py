
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


logo_path = "img/cranfield_logo.png"
logo_path = "img/cranfield_logo.png"

st.image(logo_path, width=100)

st.write("# Turbulence Modelling Predictor")
st.markdown('<p style="font-size: 15px; font-style: italic;"> ~Developed by Group 2 Cranfield CO-OP</p>',
            unsafe_allow_html=True)

# Ask the user to upload a document, allow to upload with a button.
uploaded_file = st.file_uploader(
    "\n\n**Please upload the CSV below.**", type=['csv'], key="file-uploader")

# Check if a file has been uploaded
if uploaded_file is not None:
    st.success("File successfully uploaded!")

# Ask the user to choose between PySINDy and PINNs
st.write("**Please choose a model:**")

model_choice = st.radio(
    "Please choose a model:",
    ('PySINDy', 'PINNs')
)

# Check which model was chosen and display the corresponding page
if model_choice == 'PySINDy':
    # Show PySINDy specific items
    st.header("PySINDy Model")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        columns = ["y^+", "U", "Re_tau", "u'u'",
                   "v'v'", "w'w'", "u'v'", "P", "dU/dy"]
        if all(col in df.columns for col in columns):
            df = df[["y^+", "U", "Re_tau", "u'u'",
                     "v'v'", "w'w'", "u'v'", "P", "dU/dy"]]
            Reynolds_Numbers = sorted(df["Re_tau"].unique(), reverse=True)

            st.subheader("Data processing")
            mode_choice = st.radio(
                "Please choose a mode:",
                ('Data with interpolation', 'Data without interpolation')
            )

            # Create buttons for choosing the mode
            if mode_choice == 'Data with interpolation':
                nb_interpolation = st.number_input(
                    "Please enter the number of data points to be generated with interpolation:", value=0)
                if nb_interpolation < 5000:
                    st.error("Please enter a number greater than 5000 !")
                elif nb_interpolation > 10000:
                    st.error("Please enter a number lower than 10000 !")
                else:
                    data = data_processing(
                        df, 1, nb_interpolation, Reynolds_Numbers)
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
                        "Please enter the threshold value for the PySINDy optimiser:", value=0.001, format="%.15f")
                    alpha_coeff = st.number_input(
                        "Please enter the alpha coefficient for the PySINDy optimiser:", value=0.01, format="%.15f")
                    if threshold_value < 0 or alpha_coeff < 0:
                        st.error(
                            "Please enter a positive value for the threshold value and the alpha coefficient!")
                    else:
                        for i in range(0, len(Reynolds_Numbers)):
                            optimizer = ps.STLSQ(
                                threshold=threshold_value, alpha=alpha_coeff)
                            model[Reynolds_Numbers[i]] = ps.SINDy(
                                feature_library=feature_library, optimizer=optimizer, feature_names=feature_names)
                            model[Reynolds_Numbers[i]].fit(X[i], x_dot=y[i])
                            model[Reynolds_Numbers[i]].print()

                        nb_model = 0
                        X_train = X[0]
                        y_train = y[0]
                        X_test = X[1]
                        y_test = y[1]

                        st.subheader("PySINDy Output")
                        old_stdout = sys.stdout
                        sys.stdout = buffer = StringIO()
                        model[Reynolds_Numbers[0]].print()
                        sys.stdout = old_stdout
                        model_output = buffer.getvalue()
                        st.write("PySINDy Model - Equations")
                        st.text(model_output)

                        y_pred = model[Reynolds_Numbers[nb_model]].predict(
                            X_test)
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
                        solution = simulation(
                            X_test, data[f"df_{Reynolds_Numbers[1]}"], model[Reynolds_Numbers[nb_model]])
                        if 'simulation' not in st.session_state:
                            st.session_state.simulation = False

                        if st.button('Show PySINDy Model Simulation'):
                            st.session_state.simulation = True

                        if st.button('Hide PySINDy Model Simulation'):
                            st.session_state.simulation = False

                        if st.session_state.simulation:
                            simulation_plot(
                                data[f"df_{Reynolds_Numbers[1]}"], solution)
                            df_sim = pd.DataFrame(
                                {'U': solution.y[0], "u'u'": solution.y[1], "v'v'": solution.y[2], "w'w'": solution.y[3], "u'v'": solution.y[4]})
                            mae_sim = mean_absolute_error(
                                data["df_2000"][["U", "u'u'", "v'v'", "w'w'", "u'v'"]].values, df_sim.values)
                            if mae_sim < 2:
                                st.write(
                                    "The PySINDy model successfully generates equations with physical meaning, and the model simulation is highly accurate.")
                            elif mae_sim < 5:
                                st.write(
                                    "The PySINDy model successfully generates equations with physical meaning, and the model simulation is quite accurate.")
                            elif mae_sim > 5:
                                st.write(
                                    "The PySINDy model is unable to generate equations with physical meaning, and the simulation of the model is not very accurate.")

                        st.subheader("PySINDy Prediction")
                        cols = ["dU/dy", "du'u'/dy", "dv'v'/dy", "dw'w'/dy", "du'v'/dy", "dP/dy",
                                "d(dU/dy)/dy", "d(du'u'/dy)/dy", "d(dv'v'/dy)/dy", "d(dw'w'/dy)/dy", "d(du'v'/dy)/dy", "d(dP/dy)/dy"]
                        y_plus = data[f"df_{Reynolds_Numbers[1]}"]["y^+"].to_numpy()

                        solution = simulation(
                            X_test, data[f"df_{Reynolds_Numbers[1]}"], model[Reynolds_Numbers[nb_model]])
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

            if mode_choice == 'Data without interpolation':
                data = data_processing(df, 0, 0, Reynolds_Numbers)
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
                    "Please enter the threshold value for the PySINDy optimiser:", value=0.001, format="%.15f")
                alpha_coeff = st.number_input(
                    "Please enter the alpha coefficient for the PySINDy optimiser:", value=0.01, format="%.15f")
                if threshold_value < 0 or alpha_coeff < 0:
                    st.error(
                        "Please enter a positive value for the threshold value and the alpha coefficient!")
                else:
                    for i in range(0, len(Reynolds_Numbers)):
                        optimizer = ps.STLSQ(
                            threshold=threshold_value, alpha=alpha_coeff)
                        model[Reynolds_Numbers[i]] = ps.SINDy(
                            feature_library=feature_library, optimizer=optimizer, feature_names=feature_names)
                        model[Reynolds_Numbers[i]].fit(X[i], x_dot=y[i])
                        model[Reynolds_Numbers[i]].print()

                    nb_model = 0
                    X_train = X[0]
                    y_train = y[0]
                    X_test = X[1]
                    y_test = y[1]

                    st.subheader("PySINDy Output")
                    old_stdout = sys.stdout
                    sys.stdout = buffer = StringIO()
                    model[Reynolds_Numbers[0]].print()
                    sys.stdout = old_stdout
                    model_output = buffer.getvalue()
                    st.write("PySINDy Model - Equations")
                    st.text(model_output)

                    y_pred = model[Reynolds_Numbers[nb_model]].predict(X_test)
                    y_true = y_test
                    r2 = r2_score(y_true, y_pred)
                    print(f"R-square: {r2}")
                    mse = mean_squared_error(y_true, y_pred)
                    print(f"MSE: {mse}")

                    st.metric(label="R-Squared", value=f"{r2:.10f}")
                    st.metric(label="Mean Squared Error", value=f"{mse:.10f}")

                    test_overfitting(X_train, y_train, X_test,
                                     y_test, model, Reynolds_Numbers, nb_model)

                    st.subheader("PySINDy Simulation")
                    solution = simulation(
                        X_test, data[f"df_{Reynolds_Numbers[1]}"], model[Reynolds_Numbers[nb_model]])
                    if 'simulation' not in st.session_state:
                        st.session_state.simulation = False

                    if st.button('Show PySINDy Model Simulation'):
                        st.session_state.simulation = True

                    if st.button('Hide PySINDy Model Simulation'):
                        st.session_state.simulation = False

                    if st.session_state.simulation:
                        simulation_plot(
                            data[f"df_{Reynolds_Numbers[1]}"], solution)
                        df_sim = pd.DataFrame(
                            {'U': solution.y[0], "u'u'": solution.y[1], "v'v'": solution.y[2], "w'w'": solution.y[3], "u'v'": solution.y[4]})
                        mae_sim = mean_absolute_error(
                            data["df_2000"][["U", "u'u'", "v'v'", "w'w'", "u'v'"]].values, df_sim.values)
                        if mae_sim < 1:
                            st.write(
                                "The PySINDy model successfully generates equations with physical meaning, and the model simulation is highly accurate.")
                        elif mae_sim >= 1 and mae_sim < 5:
                            st.write(
                                "The PySINDy model successfully generates equations with physical meaning, and the model simulation is quite accurate.")
                        elif mae_sim >= 5:
                            st.write(
                                "The PySINDy model is unable to generate equations with physical meaning, and the simulation of the model is not very accurate.")

                    st.subheader("PySINDy Prediction")
                    cols = ["dU/dy", "du'u'/dy", "dv'v'/dy", "dw'w'/dy", "du'v'/dy", "dP/dy",
                            "d(dU/dy)/dy", "d(du'u'/dy)/dy", "d(dv'v'/dy)/dy", "d(dw'w'/dy)/dy", "d(du'v'/dy)/dy", "d(dP/dy)/dy"]
                    y_plus = data[f"df_{Reynolds_Numbers[1]}"]["y^+"].to_numpy()

                    solution = simulation(
                        X_test, data[f"df_{Reynolds_Numbers[1]}"], model[Reynolds_Numbers[nb_model]])
                    if 'prediction' not in st.session_state:
                        st.session_state.prediction = False

                    if st.button('Show PySINDy Model Prediction'):
                        st.session_state.prediction = True

                    if st.button('Hide PySINDy Model Prediction'):
                        st.session_state.prediction = False

                    if st.session_state.prediction:
                        for i in range(12):
                            plt.figure(figsize=(12, 6))
                            plt.plot(y_plus, y_true[:, i],
                                     "k", label="Real value")
                            plt.plot(
                                y_plus, y_pred[:, i], "r--", label="model prediction")
                            plt.xscale("log")
                            plt.xlabel('y^+')
                            plt.ylabel(cols[i])
                            plt.legend()
                            st.pyplot(plt)
        else:
            st.error(
                "Please upload a csv file with the columns y^+, U, Re_tau, u'u', v'v', w'w', u'v', P and dU/dy to proceed.")
    else:
        st.error("Please upload a csv file to proceed.")

elif model_choice == 'PINNs':
    # Show items specific to PINNs
    st.write("## PINNs Model")
    st.write("Content for PINNs model goes here...")

    # Add your logic for the PINNs model here

else:
    st.write("Please choose a model to get started.")
