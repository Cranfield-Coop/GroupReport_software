#Add header

#amsvdm

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
logo_path = "C:/Users/Louis/Desktop/Codes/cranfield_logo.png" ##remember to change the path to the logo
logo_path = "C:/Users/moi/OneDrive - etu.utc.fr/Bureau/UTC/Cranfield/Group_project/Software/cranfield_logo.png" ##remember to change the path to the logo

st.image(logo_path, width= 100)

st.write("# Turbulence Modelling Predictor")
st.markdown('<p style="font-size: 15px; font-style: italic;"> ~Developed by Group 2 Cranfield CO-OP</p>', unsafe_allow_html=True)

# Ask the user to upload a document, allow to upload with a button.
uploaded_file = st.file_uploader("\n\n**Please upload the CSV below.**", type=['csv'],key="file-uploader")

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
if model_choice ==  'PySINDy':
    # Show PySINDy specific items
    st.header("PySINDy Model")
    if uploaded_file is not None:
      df= pd.read_csv(uploaded_file)
      columns = ["y^+","U","Re_tau","u'u'","v'v'","w'w'","u'v'","P","dU/dy"]
      if all(col in df.columns for col in columns):
        df = df[["y^+","U","Re_tau","u'u'","v'v'","w'w'","u'v'","P","dU/dy"]]
        Reynolds_Numbers = sorted(df["Re_tau"].unique(),reverse=True)

        def plot(df,Reynolds_Numbers):
          columns = ["U","u'u'","v'v'","w'w'","u'v'"]
          for col in columns:
              plt.figure(figsize=(12, 6))
              for Re in Reynolds_Numbers:
                  subset = df[df["Re_tau"] == Re]
                  if col == "u'v'":
                      subset[col] = - subset[col]
                  plt.plot(subset["y^+"], subset[col], label=f"Re={Re}")
              plt.xscale('log')
              plt.xlim(left=0.9)
              plt.xlabel('y^+')
              if col == "u'v'":
                  plt.ylabel(f"- {col}")
                  plt.title(f"- {col} as a function of y^+")
              else:
                  plt.ylabel(col)
                  plt.title(f"{col} as a function of y^+")
              plt.legend()
              st.pyplot(plt)

        def data_interpolation(df,nb, Reynolds_Numbers):
            interval = np.linspace(df['y^+'].min(), df['y^+'].max(), num=nb)
            interpolated_columns = {}
            for column in df.columns:
                if column != 'y^+': 
                    f = interp1d(df['y^+'], df[column], kind='linear')  
                    interpolated_columns[column] = f(interval)
            df = pd.DataFrame(interpolated_columns)
            df['y^+'] = interval  
            df["Re_tau"] = Reynolds_Numbers
            return df

        def first_derivative(columns,df):
            for col in columns:  
                x = df[col].to_numpy()
                y = df["y^+"].to_numpy()
                dx = np.gradient(x,y)
                df["d" + col + "/dy"] = dx
            return df

        def second_derivative(columns,df):
            for col in columns:  
                x = df[col].to_numpy()
                y = df["y^+"].to_numpy()
                dx = np.gradient(x,y)
                df["d(" + col + ")/dy"] = dx
            return df

        def data_processing(df, interpolation,nb_interpolation,Reynolds_Numbers):
            if interpolation == 0:
                data = {f'df_{Re}': df[df["Re_tau"] == Re] for Re in Reynolds_Numbers}
                for Re in Reynolds_Numbers:
                    columns_to_derive = ["u'u'", "v'v'", "w'w'", "u'v'","P"]
                    data[f"df_{Re}"] = first_derivative(columns_to_derive,data[f"df_{Re}"])
                    columns_to_derive = ["dU/dy","du'u'/dy", "dv'v'/dy", "dw'w'/dy", "du'v'/dy","dP/dy"]
                    data[f"df_{Re}"] = second_derivative(columns_to_derive,data[f"df_{Re}"])
                return data
            else:
                data = {f'df_{Re}': df[df["Re_tau"] == Re].drop(columns=["Re_tau"]) for Re in Reynolds_Numbers}
                for Re in Reynolds_Numbers:
                    data[f"df_{Re}"] = data_interpolation(data[f"df_{Re}"],nb_interpolation,Re)
                for Re in Reynolds_Numbers:
                    columns_to_derive = ["u'u'", "v'v'", "w'w'", "u'v'","P"]
                    data[f"df_{Re}"] = first_derivative(columns_to_derive,data[f"df_{Re}"])
                    columns_to_derive = ["dU/dy","du'u'/dy", "dv'v'/dy", "dw'w'/dy", "du'v'/dy","dP/dy"]
                    data[f"df_{Re}"] = second_derivative(columns_to_derive,data[f"df_{Re}"])
                return data

        st.subheader("Data processing")
        mode_choice = st.radio(
          "Please choose a mode:",
          ('Data with interpolation', 'Data without interpolation')
        )

        # Create buttons for choosing the mode
        if mode_choice == 'Data with interpolation':
            nb_interpolation = st.number_input("Please enter the number of data points to be generated with interpolation:", value=0)
            if nb_interpolation < 5000:
                st.error("Please enter a number greater than 5000 !")
            elif nb_interpolation > 10000:
                st.error("Please enter a number lower than 10000 !")
            else:
              data = data_processing(df,1,nb_interpolation,Reynolds_Numbers)
              if 'show_graph' not in st.session_state:
                  st.session_state.show_graph = False  

              if st.button('Show raw data visualizations'):
                  st.session_state.show_graph = True

              if st.button('Hide raw data visualizations'):
                  st.session_state.show_graph = False

              if st.session_state.show_graph:
                  df = pd.concat(data.values(), ignore_index=True)
                  plot(df,Reynolds_Numbers)

              def data_PySINDy(data):
                  for Re in Reynolds_Numbers:
                      data[f"df_{Re}"] = data[f"df_{Re}"].drop(columns=["Re_tau"])
                  X = [[] for _ in range(len(Reynolds_Numbers))]
                  y = [[] for _ in range(len(Reynolds_Numbers))]
                  for i in range(0,len(Reynolds_Numbers)):
                      X[i] = data[f"df_{Reynolds_Numbers[i]}"].drop(columns=["y^+","d(dU/dy)/dy","d(du'u'/dy)/dy", "d(dv'v'/dy)/dy", "d(dw'w'/dy)/dy", "d(du'v'/dy)/dy","d(dP/dy)/dy"]).values
                      y[i] = data[f"df_{Reynolds_Numbers[i]}"][["dU/dy", "du'u'/dy", "dv'v'/dy", "dw'w'/dy", "du'v'/dy","dP/dy","d(dU/dy)/dy","d(du'u'/dy)/dy", "d(dv'v'/dy)/dy", "d(dw'w'/dy)/dy", "d(du'v'/dy)/dy","d(dP/dy)/dy"]].values
                  return X,y
                  

              X, y = data_PySINDy(data)

              model = {}
              feature_names = ["U","u'u'", "v'v'", "w'w'", "u'v'","P","dU/dy", "du'u'/dy", "dv'v'/dy", "dw'w'/dy", "du'v'/dy","dP/dy"]
              feature_library = ps.PolynomialLibrary(1)
              st.subheader("Selection of PySINDy hyperparameters")
              threshold_value = st.number_input("Please enter the threshold value for the PySINDy optimiser:", value=0.001,format="%.15f")
              alpha_coeff = st.number_input("Please enter the alpha coefficient for the PySINDy optimiser:", value=0.01,format="%.15f")
              if threshold_value < 0 or alpha_coeff < 0:
                  st.error("Please enter a positive value for the threshold value and the alpha coefficient!")
              else:
                for i in range (0,len(Reynolds_Numbers)):
                    optimizer = ps.STLSQ(threshold=threshold_value,alpha=alpha_coeff)
                    model[Reynolds_Numbers[i]] = ps.SINDy(feature_library=feature_library,optimizer=optimizer,feature_names=feature_names)
                    model[Reynolds_Numbers[i]].fit(X[i],x_dot = y[i])
                    model[Reynolds_Numbers[i]].print()

                nb_model = 0
                X_train = X[0]
                y_train = y[0]
                X_test = X[1]
                y_test = y[1]

                def integrate(y_plus, y, coeffs):
                    U, uu, vv, ww, uv, P, dUdy, duudy, dvvdy, dwwdy, duvdy, dPdy = y
                    
                    dU_dy = dUdy
                    duu_dy = duudy
                    dvv_dy = dvvdy
                    dww_dy = dwwdy
                    duv_dy = duvdy
                    dP_dy = dPdy
                    
                    derivatives = [dU_dy, duu_dy, dvv_dy, dww_dy, duv_dy, dP_dy]
                    for i in range(6, 12):
                        dd = coeffs[i][0] + np.dot(coeffs[i][1:], y)
                        derivatives.append(dd)
                    
                    return derivatives

                def simulation(X, df, model):
                    initial_conditions = X[0]
                    coeffs = model.optimizer.coef_
                    y_plus_points = df["y^+"].values
                    solution = solve_ivp(
                        fun=integrate, 
                        t_span=(y_plus_points.min(), y_plus_points.max()), 
                        y0=initial_conditions, 
                        args=(coeffs,), 
                        t_eval=y_plus_points,  
                        method='RK45',
                        dense_output = True
                    )
                    return solution

                def simulation_plot(df, solution):
                    col = ["U","u'u'","v'v'","w'w'","u'v'"]
                    for i in range(5):
                        plt.figure(figsize=(12, 6))
                        subset = df[col[i]]
                        if col[i] == "u'v'":
                            solution.y[i] = - solution.y[i] 
                            subset = -1 * df[col[i]]
                        plt.plot(solution.t, solution.y[i],label = f"{col[i]} (PySINDy)")
                        plt.plot(solution.t, subset.values, label=f"{col[i]} (DNS)")
                        plt.xscale('log')
                        plt.xlabel('y^+')
                        if col[i] == "u'v'":
                            plt.ylabel(f"- {col[i]}")
                            plt.title(f"- {col[i]} as a function of y^+")
                        else:
                            plt.ylabel(col[i])
                            plt.title(f"{col[i]} as a function of y^+")
                        plt.legend()
                        st.pyplot(plt)
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

                def test_overfitting(X_train,y_train,X_test,y_test):
                  y_train_pred =  model[Reynolds_Numbers[nb_model]].predict(X_train)
                  mse_train = mean_squared_error(y_train, y_train_pred)
                  y_test_pred =  model[Reynolds_Numbers[nb_model]].predict(X_test)
                  mse_test = mean_squared_error(y_test, y_test_pred)
                  if abs(mse_train - mse_test) > 0.001:
                      #st.write("PySINDy Model - Overfitting!")
                      st.metric(label="PySINDy Model", value="Overfitting !")
                  else:
                      #st.write("PySINDy Model - No Overfitting!")
                      st.metric(label="PySINDy Model", value="No Overfitting !")

                test_overfitting(X_train,y_train,X_test,y_test)

                st.subheader("PySINDy Simulation")
                solution = simulation(X_test, data[f"df_{Reynolds_Numbers[1]}"], model[Reynolds_Numbers[nb_model]])
                if 'simulation' not in st.session_state:
                    st.session_state.simulation = False  

                if st.button('Show PySINDy Model Simulation'):
                    st.session_state.simulation = True

                if st.button('Hide PySINDy Model Simulation'):
                    st.session_state.simulation = False

                if st.session_state.simulation:
                    simulation_plot(data[f"df_{Reynolds_Numbers[1]}"], solution)
                    df_sim = pd.DataFrame({'U': solution.y[0], "u'u'": solution.y[1], "v'v'": solution.y[2], "w'w'": solution.y[3], "u'v'": solution.y[4]})
                    mae_sim = mean_absolute_error(data["df_2000"][["U","u'u'","v'v'","w'w'","u'v'"]].values,df_sim.values)
                    if mae_sim < 2:
                        st.write("The PySINDy model successfully generates equations with physical meaning, and the model simulation is highly accurate.")
                    elif mae_sim < 5:
                        st.write("The PySINDy model successfully generates equations with physical meaning, and the model simulation is quite accurate.")
                    elif mae_sim > 5:
                        st.write("The PySINDy model is unable to generate equations with physical meaning, and the simulation of the model is not very accurate.")

                st.subheader("PySINDy Prediction")
                cols = ["dU/dy", "du'u'/dy", "dv'v'/dy", "dw'w'/dy", "du'v'/dy","dP/dy","d(dU/dy)/dy","d(du'u'/dy)/dy", "d(dv'v'/dy)/dy", "d(dw'w'/dy)/dy", "d(du'v'/dy)/dy","d(dP/dy)/dy"]
                y_plus = data[f"df_{Reynolds_Numbers[1]}"]["y^+"].to_numpy()
                
                solution = simulation(X_test, data[f"df_{Reynolds_Numbers[1]}"], model[Reynolds_Numbers[nb_model]])
                if 'prediction' not in st.session_state:
                    st.session_state.prediction = False  

                if st.button('Show PySINDy Model Prediction'):
                    st.session_state.prediction = True

                if st.button('Hide PySINDy Model Prediction'):
                    st.session_state.prediction = False

                if st.session_state.prediction:   
                    for i in range(12):
                        plt.figure(figsize=(12, 6))
                        plt.plot(y_plus, y_true[:,i], "k", label="Real value")
                        plt.plot(y_plus, y_pred[:,i], "r--", label="model prediction")
                        plt.xscale("log")
                        plt.xlabel('y^+')
                        plt.ylabel(cols[i])
                        plt.legend()
                        st.pyplot(plt)

        if mode_choice == 'Data without interpolation':
            data = data_processing(df,0,0,Reynolds_Numbers)
            if 'show_graph' not in st.session_state:
                st.session_state.show_graph = False  

            if st.button('Show raw data visualizations'):
                st.session_state.show_graph = True

            if st.button('Hide raw data visualizations'):
                st.session_state.show_graph = False

            if st.session_state.show_graph:
                df = pd.concat(data.values(), ignore_index=True)
                plot(df,Reynolds_Numbers)

            def data_PySINDy(data):
                for Re in Reynolds_Numbers:
                    data[f"df_{Re}"] = data[f"df_{Re}"].drop(columns=["Re_tau"])
                X = [[] for _ in range(len(Reynolds_Numbers))]
                y = [[] for _ in range(len(Reynolds_Numbers))]
                for i in range(0,len(Reynolds_Numbers)):
                    X[i] = data[f"df_{Reynolds_Numbers[i]}"].drop(columns=["y^+","d(dU/dy)/dy","d(du'u'/dy)/dy", "d(dv'v'/dy)/dy", "d(dw'w'/dy)/dy", "d(du'v'/dy)/dy","d(dP/dy)/dy"]).values
                    y[i] = data[f"df_{Reynolds_Numbers[i]}"][["dU/dy", "du'u'/dy", "dv'v'/dy", "dw'w'/dy", "du'v'/dy","dP/dy","d(dU/dy)/dy","d(du'u'/dy)/dy", "d(dv'v'/dy)/dy", "d(dw'w'/dy)/dy", "d(du'v'/dy)/dy","d(dP/dy)/dy"]].values
                return X,y
                

            X, y = data_PySINDy(data)

            model = {}
            feature_names = ["U","u'u'", "v'v'", "w'w'", "u'v'","P","dU/dy", "du'u'/dy", "dv'v'/dy", "dw'w'/dy", "du'v'/dy","dP/dy"]
            feature_library = ps.PolynomialLibrary(1)
            st.subheader("Selection of PySINDy hyperparameters")
            threshold_value = st.number_input("Please enter the threshold value for the PySINDy optimiser:", value=0.001,format="%.15f")
            alpha_coeff = st.number_input("Please enter the alpha coefficient for the PySINDy optimiser:", value=0.01,format="%.15f")
            if threshold_value < 0 or alpha_coeff < 0:
                st.error("Please enter a positive value for the threshold value and the alpha coefficient!")
            else:
              for i in range (0,len(Reynolds_Numbers)):
                  optimizer = ps.STLSQ(threshold=threshold_value,alpha=alpha_coeff)
                  model[Reynolds_Numbers[i]] = ps.SINDy(feature_library=feature_library,optimizer=optimizer,feature_names=feature_names)
                  model[Reynolds_Numbers[i]].fit(X[i],x_dot = y[i])
                  model[Reynolds_Numbers[i]].print()

              nb_model = 0
              X_train = X[0]
              y_train = y[0]
              X_test = X[1]
              y_test = y[1]

              def integrate(y_plus, y, coeffs):
                  U, uu, vv, ww, uv, P, dUdy, duudy, dvvdy, dwwdy, duvdy, dPdy = y
                  
                  dU_dy = dUdy
                  duu_dy = duudy
                  dvv_dy = dvvdy
                  dww_dy = dwwdy
                  duv_dy = duvdy
                  dP_dy = dPdy
                  
                  derivatives = [dU_dy, duu_dy, dvv_dy, dww_dy, duv_dy, dP_dy]
                  for i in range(6, 12):
                      dd = coeffs[i][0] + np.dot(coeffs[i][1:], y)
                      derivatives.append(dd)
                  
                  return derivatives

              def simulation(X, df, model):
                  initial_conditions = X[0]
                  coeffs = model.optimizer.coef_
                  y_plus_points = df["y^+"].values
                  solution = solve_ivp(
                      fun=integrate, 
                      t_span=(y_plus_points.min(), y_plus_points.max()), 
                      y0=initial_conditions, 
                      args=(coeffs,), 
                      t_eval=y_plus_points,  
                      method='RK45',
                      dense_output = True
                  )
                  return solution

              def simulation_plot(df, solution):
                  col = ["U","u'u'","v'v'","w'w'","u'v'"]
                  for i in range(5):
                      plt.figure(figsize=(12, 6))
                      subset = df[col[i]]
                      if col[i] == "u'v'":
                          solution.y[i] = - solution.y[i] 
                          subset = -1 * df[col[i]]
                      plt.plot(solution.t, solution.y[i],label = f"{col[i]} (PySINDy)")
                      plt.plot(solution.t, subset.values, label=f"{col[i]} (DNS)")
                      plt.xscale('log')
                      plt.xlabel('y^+')
                      if col[i] == "u'v'":
                          plt.ylabel(f"- {col[i]}")
                          plt.title(f"- {col[i]} as a function of y^+")
                      else:
                          plt.ylabel(col[i])
                          plt.title(f"{col[i]} as a function of y^+")
                      plt.legend()
                      st.pyplot(plt)
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

              def test_overfitting(X_train,y_train,X_test,y_test):
                y_train_pred =  model[Reynolds_Numbers[nb_model]].predict(X_train)
                mse_train = mean_squared_error(y_train, y_train_pred)
                y_test_pred =  model[Reynolds_Numbers[nb_model]].predict(X_test)
                mse_test = mean_squared_error(y_test, y_test_pred)
                if abs(mse_train - mse_test) > 0.001:
                    #st.write("PySINDy Model - Overfitting!")
                    st.metric(label="PySINDy Model", value="Overfitting !")
                else:
                    #st.write("PySINDy Model - No Overfitting!")
                    st.metric(label="PySINDy Model", value="No Overfitting !")

              test_overfitting(X_train,y_train,X_test,y_test)

              st.subheader("PySINDy Simulation")
              solution = simulation(X_test, data[f"df_{Reynolds_Numbers[1]}"], model[Reynolds_Numbers[nb_model]])
              if 'simulation' not in st.session_state:
                  st.session_state.simulation = False  

              if st.button('Show PySINDy Model Simulation'):
                  st.session_state.simulation = True

              if st.button('Hide PySINDy Model Simulation'):
                  st.session_state.simulation = False

              if st.session_state.simulation:
                  simulation_plot(data[f"df_{Reynolds_Numbers[1]}"], solution)
                  df_sim = pd.DataFrame({'U': solution.y[0], "u'u'": solution.y[1], "v'v'": solution.y[2], "w'w'": solution.y[3], "u'v'": solution.y[4]})
                  mae_sim = mean_absolute_error(data["df_2000"][["U","u'u'","v'v'","w'w'","u'v'"]].values,df_sim.values)
                  if mae_sim < 1:
                      st.write("The PySINDy model successfully generates equations with physical meaning, and the model simulation is highly accurate.")
                  elif mae_sim >= 1 and mae_sim < 5:
                      st.write("The PySINDy model successfully generates equations with physical meaning, and the model simulation is quite accurate.")
                  elif mae_sim >= 5:
                      st.write("The PySINDy model is unable to generate equations with physical meaning, and the simulation of the model is not very accurate.")


              st.subheader("PySINDy Prediction")
              cols = ["dU/dy", "du'u'/dy", "dv'v'/dy", "dw'w'/dy", "du'v'/dy","dP/dy","d(dU/dy)/dy","d(du'u'/dy)/dy", "d(dv'v'/dy)/dy", "d(dw'w'/dy)/dy", "d(du'v'/dy)/dy","d(dP/dy)/dy"]
              y_plus = data[f"df_{Reynolds_Numbers[1]}"]["y^+"].to_numpy()
              
              solution = simulation(X_test, data[f"df_{Reynolds_Numbers[1]}"], model[Reynolds_Numbers[nb_model]])
              if 'prediction' not in st.session_state:
                  st.session_state.prediction = False  

              if st.button('Show PySINDy Model Prediction'):
                  st.session_state.prediction = True

              if st.button('Hide PySINDy Model Prediction'):
                  st.session_state.prediction = False

              if st.session_state.prediction:   
                  for i in range(12):
                      plt.figure(figsize=(12, 6))
                      plt.plot(y_plus, y_true[:,i], "k", label="Real value")
                      plt.plot(y_plus, y_pred[:,i], "r--", label="model prediction")
                      plt.xscale("log")
                      plt.xlabel('y^+')
                      plt.ylabel(cols[i])
                      plt.legend()
                      st.pyplot(plt)
      else:
          st.error("Please upload a csv file with the columns y^+, U, Re_tau, u'u', v'v', w'w', u'v', P and dU/dy to proceed.")
    else:
      st.error("Please upload a csv file to proceed.")

elif model_choice == 'PINNs':
    # Show items specific to PINNs
    st.write("## PINNs Model")
    st.write("Content for PINNs model goes here...")
    
    # Add your logic for the PINNs model here 

else:
    st.write("Please choose a model to get started.")

