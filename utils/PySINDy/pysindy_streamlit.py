import streamlit as st
import pandas as pd
import numpy as np
import pysindy as ps
import sys
import time
from io import StringIO
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from utils.PySINDy.psindy import *



columns = ["y^+", "U", "Re_tau", "u'u'", "v'v'", "w'w'", "u'v'", "P", "dU/dy"]
logo_path = "img/cranfield_logo.png"

def app():
    #st.image(logo_path, width=100)
    st.write("# Turbulence Modelling Predictor")
    st.markdown('<p style="font-size: 15px; font-style: italic;"> ~Developed by Group 2 Cranfield CO-OP</p>',unsafe_allow_html=True) 
    uploaded_file_pysindy = st.file_uploader("\n\n**Please upload the CSV below.**", type=['csv'], key="PySINDy")
    if uploaded_file_pysindy is not None:
        st.success("File successfully uploaded!")
        df = pd.read_csv(uploaded_file_pysindy)
        if all(col in df.columns for col in columns):
            df = df[columns]
        else:
            st.error("Please upload a csv file with the columns y^+, U, Re_tau, u'u', v'v', w'w', u'v', P and dU/dy to proceed.")
            return None
        Reynolds_Numbers = sorted(df["Re_tau"].unique(), reverse=True)
        st.header("PySINDy Model")
        st.subheader("Data processing")
        mode_choice = st.selectbox("Choose processing mode:", ['Data with interpolation', 'Data without interpolation'])
        if mode_choice == 'Data with interpolation':
            nb_interpolation = st.number_input("Please enter the number of data points to be generated with interpolation:", min_value=5000, max_value=100000, value=10000,step=1000, key='nb_interp')
            if nb_interpolation < len(df):
                st.error(f"The number of data points to be generated with interpolation must be greater than the number of data points in the csv file ({len(df)})")
                return None
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
                "Please enter the threshold value for the PySINDy optimiser:", value=0.001, min_value=0.0, format="%.6f")
            alpha_coeff = st.number_input(
                "Please enter the alpha coefficient for the PySINDy optimiser:", value=0.01, min_value=0.0, format="%.6f")             
            nb_model = 0
            if len(Reynolds_Numbers) > 1:
                X_train = X[0]
                y_train = y[0]
                X_test = X[1]
                y_test = y[1]
            else:
                ratio = st.number_input("Please enter the ratio to split the data into training and testing datasets:", value=0.8,min_value=0.0,max_value=1.0,format="%.2f")
                X_1 = X[0]
                y_1 = y[0]
                ratio_split = int(len(X_1) * ratio)
                X_train = X_1[:ratio_split]
                X_test = X_1[ratio_split:]
                y_train = y_1[:ratio_split]
                y_test = y_1[ratio_split:]

            if 'run_model' not in st.session_state:
                st.session_state.run_model = False

            if st.button('Run Model'):
                st.session_state.run_model = True

            if st.session_state.run_model:
                timer_start = time.time()
                for i in range(0, 1):
                    optimizer = ps.STLSQ(
                        threshold=threshold_value, alpha=alpha_coeff)
                    model[Reynolds_Numbers[i]] = ps.SINDy(
                        feature_library=feature_library, optimizer=optimizer, feature_names=feature_names)
                    model[Reynolds_Numbers[i]].fit(X[i], x_dot=y[i])
                    model[Reynolds_Numbers[i]].print()
                timer_stop = time.time()

                st.subheader("PySINDy Output")
                old_stdout = sys.stdout
                sys.stdout = buffer = StringIO()
                model[Reynolds_Numbers[0]].print()
                sys.stdout = old_stdout
                model_output = buffer.getvalue()
                st.write("PySINDy Model - Equations")
                st.text(model_output)
                st.subheader("Metrics")
                st.metric("Excecution time",
                          value=f"{timer_stop - timer_start:.2f} seconds",
                          delta=None,
                          delta_color="normal",
                          help=None,
                          label_visibility="visible")

                sparsity = 0
                for i in range (len(model[Reynolds_Numbers[nb_model]].optimizer.coef_)):
                    if np.all(model[Reynolds_Numbers[nb_model]].optimizer.coef_[i] == 0):
                        sparsity = -1
                if sparsity == -1:
                    st.error(f"Sparsity parameter is too big ({threshold_value}) and eliminated all coefficients!") 
                else:
                    y_pred = model[Reynolds_Numbers[nb_model]].predict(X_test)
                    y_true = y_test
                    r2 = r2_score(y_true, y_pred)
                    print(f"R-square: {r2}")
                    mse = mean_squared_error(y_true, y_pred)
                    print(f"MSE: {mse}")

                    st.metric(label="R-Squared", 
                              value=f"{r2:.10f}",
                              delta=None,
                              delta_color="normal",
                              help=None,
                              label_visibility="visible")
                    st.metric(label="Mean Squared Error",
                              value=f"{mse:.10f}",
                              delta=None,
                              delta_color="normal",
                              help=None,
                              label_visibility="visible")

                    test_overfitting(X_train, y_train,
                                    X_test, y_test, model, Reynolds_Numbers, nb_model)

                    st.subheader("PySINDy Simulation")
                    if len(Reynolds_Numbers)> 1:
                        solution = simulation(X_test, data[f"df_{Reynolds_Numbers[1]}"], model[Reynolds_Numbers[nb_model]])
                    else:
                        solution = simulation(X_test, data[f"df_{Reynolds_Numbers[0]}"], model[Reynolds_Numbers[nb_model]])
                    
                    if len(Reynolds_Numbers) > 1:
                        simulation_plot(data[f"df_{Reynolds_Numbers[1]}"], solution)
                        df_sim = pd.DataFrame({"y^+":solution.t,'U': solution.y[0], "u'u'": solution.y[1], "v'v'": solution.y[2], "w'w'": solution.y[3], "u'v'": solution.y[4]})
                        df_pred = data[f"df_{Reynolds_Numbers[1]}"][data[f"df_{Reynolds_Numbers[1]}"]['y^+'].isin(solution.t)]
                        mae_sim = mean_absolute_error(df_pred[["U","u'u'","v'v'","w'w'","u'v'"]].values,df_sim[["U","u'u'","v'v'","w'w'","u'v'"]].values)
                    else:
                        simulation_plot(data[f"df_{Reynolds_Numbers[0]}"], solution)
                        df_sim = pd.DataFrame({"y^+":solution.t,'U': solution.y[0], "u'u'": solution.y[1], "v'v'": solution.y[2], "w'w'": solution.y[3], "u'v'": solution.y[4]})
                        df_pred = data[f"df_{Reynolds_Numbers[0]}"][data[f"df_{Reynolds_Numbers[0]}"]['y^+'].isin(solution.t)]
                        mae_sim = mean_absolute_error(df_pred[["U","u'u'","v'v'","w'w'","u'v'"]].values,df_sim[["U","u'u'","v'v'","w'w'","u'v'"]].values)
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
                        file_name='prediction_output_pysindy.csv',
                        mime='text/csv',
                    )
    else:
        st.error("Please upload a csv file to proceed.")
                
