import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp
from sklearn.metrics import mean_squared_error


def plot(df, Reynolds_Numbers):
    """Plot the data for each Reynolds number
    Args:
        df (pd.DataFrame): The dataframe containing the data
        Reynolds_Numbers (list): The list of Reynolds numbers

    Returns:
        None
    """
    columns = ["U", "u'u'", "v'v'", "w'w'", "u'v'"]
    for col in columns:
        plt.figure(figsize=(12, 6))
        for Re in Reynolds_Numbers:
            subset = df[df["Re_tau"] == Re]
            if col == "u'v'":
                subset[col] = - subset[col]
            plt.plot(subset["y^+"], subset[col], label=f"Re={Re}")
        plt.xscale('log')
        plt.xlim(left=0.1, right = 10000)
        plt.xlabel('y^+')
        if col == "u'v'":
            plt.ylabel(f"- {col}")
            plt.title(f"- {col} as a function of y^+")
        else:
            plt.ylabel(col)
            plt.title(f"{col} as a function of y^+")
        plt.legend()
        st.pyplot(plt)


def data_interpolation(df, nb, Reynolds_Numbers):
    """
    Interpolate the data to have the same number of points for each Reynolds number
    Args:
        df (pd.DataFrame): The dataframe containing the data
        nb (int): The number of points to interpolate
        Reynolds_Numbers (list): The list of Reynolds numbers

    Returns:
        pd.DataFrame: The interpolated dataframe    """
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


def first_derivative(columns, df):
    """Compute the first derivative of the columns"""
    for col in columns:
        x = df[col].to_numpy()
        y = df["y^+"].to_numpy()
        dx = np.gradient(x, y)
        df["d" + col + "/dy"] = dx
    return df


def second_derivative(columns, df):
    """Compute the second derivative of the columns"""
    for col in columns:
        x = df[col].to_numpy()
        y = df["y^+"].to_numpy()
        dx = np.gradient(x, y)
        df["d(" + col + ")/dy"] = dx
    return df


def data_processing(df, interpolation, nb_interpolation, Reynolds_Numbers):
    """Process the data to have the first and second derivative
    Args:
        df (pd.DataFrame): The dataframe containing the data
        interpolation (int): The interpolation method
        nb_interpolation (int): The number of points to interpolate
        Reynolds_Numbers (list): The list of Reynolds numbers
    Returns:
        dict: The processed data
        """
    #print(len(Reynolds_Numbers))
    if interpolation == 0:
        data = {f'df_{Re}': df[df["Re_tau"] == Re] for Re in Reynolds_Numbers}
        for Re in Reynolds_Numbers:
            columns_to_derive = ["u'u'", "v'v'", "w'w'", "u'v'", "P"]
            data[f"df_{Re}"] = first_derivative(columns_to_derive, data[f"df_{Re}"])
            columns_to_derive = ["dU/dy", "du'u'/dy",
                                 "dv'v'/dy", "dw'w'/dy", "du'v'/dy", "dP/dy"]
            data[f"df_{Re}"] = second_derivative(
                columns_to_derive, data[f"df_{Re}"])
        return data
    else:
        data = {f'df_{Re}': df[df["Re_tau"] == Re].drop(
            columns=["Re_tau"]) for Re in Reynolds_Numbers}
        for Re in Reynolds_Numbers:
            data[f"df_{Re}"] = data_interpolation(
                data[f"df_{Re}"], nb_interpolation, Re)
        for Re in Reynolds_Numbers:
            columns_to_derive = ["u'u'", "v'v'", "w'w'", "u'v'", "P"]
            data[f"df_{Re}"] = first_derivative(
                columns_to_derive, data[f"df_{Re}"])
            columns_to_derive = ["dU/dy", "du'u'/dy",
                                 "dv'v'/dy", "dw'w'/dy", "du'v'/dy", "dP/dy"]
            data[f"df_{Re}"] = second_derivative(
                columns_to_derive, data[f"df_{Re}"])
        return data


def data_PySINDy(data, Reynolds_Numbers):
    """Prepare the data for PySINDy
    Args:
        data (dict): The processed data
        Reynolds_Numbers (list): The list of Reynolds numbers
    Returns:
        list: The input data
        list: The output data
        """
    for Re in Reynolds_Numbers:
        data[f"df_{Re}"] = data[f"df_{Re}"].drop(columns=["Re_tau"])
    X = [[] for _ in range(len(Reynolds_Numbers))]
    y = [[] for _ in range(len(Reynolds_Numbers))]
    for i in range(0, len(Reynolds_Numbers)):
        X[i] = data[f"df_{Reynolds_Numbers[i]}"].drop(
            columns=["y^+", "d(dU/dy)/dy", "d(du'u'/dy)/dy", "d(dv'v'/dy)/dy", "d(dw'w'/dy)/dy", "d(du'v'/dy)/dy", "d(dP/dy)/dy"]).values
        y[i] = data[f"df_{Reynolds_Numbers[i]}"][["dU/dy", "du'u'/dy", "dv'v'/dy", "dw'w'/dy", "du'v'/dy", "dP/dy",
                                                  "d(dU/dy)/dy", "d(du'u'/dy)/dy", "d(dv'v'/dy)/dy", "d(dw'w'/dy)/dy", "d(du'v'/dy)/dy", "d(dP/dy)/dy"]].values
    return X, y


def integrate(y_plus, y, coeffs):
    """Integrate the ODEs
    Args:
        y_plus (float): The y_plus value
        y (list): The list of variables
        coeffs (list): The coefficients
    Returns:
        list: The derivatives"""
    U, uu, vv, ww, uv, P, dUdy, duudy, dvvdy, dwwdy, duvdy, dPdy = y
    dU_dy = dUdy
    duu_dy = duudy
    dvv_dy = dvvdy
    dww_dy = dwwdy
    duv_dy = duvdy
    dP_dy = dPdy

    derivatives = [dU_dy, duu_dy,
                   dvv_dy, dww_dy, duv_dy, dP_dy]
    for i in range(6, 12):
        dd = coeffs[i][0] + np.dot(coeffs[i][1:], y)
        derivatives.append(dd)
    return derivatives


def test_overfitting(X_train, y_train, X_test, y_test, model, Reynolds_Numbers, nb_model):
    """Test if the model is overfitting
    Args:
        X_train (list): The input training data
        y_train (list): The output training data
        X_test (list): The input testing data
        y_test (list): The output testing data
        model (dict): The models
        Reynolds_Numbers (list): The list of Reynolds numbers
        nb_model (int): The number of the model

    Returns:
        None"""
    y_train_pred = model[Reynolds_Numbers[nb_model]].predict(X_train)
    mse_train = mean_squared_error(y_train, y_train_pred)
    y_test_pred = model[Reynolds_Numbers[nb_model]].predict(X_test)
    mse_test = mean_squared_error(y_test, y_test_pred)
    if abs(mse_train - mse_test) > 0.001:
        # st.write("PySINDy Model - Overfitting!")
        st.metric(label="PySINDy Model", value="Overfitting !")
    else:
        # st.write("PySINDy Model - No Overfitting!")
        st.metric(label="PySINDy Model", value="No Overfitting !")


def simulation(X, df, model):
    """Simulate the model
    Args:
        X (list): The input data
        df (pd.DataFrame): The dataframe containing the data
        model (dict): The models
    """
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
        dense_output=True
    )
    return solution


def simulation_plot(df, solution):
    """Plot the simulation
    Args:
        df (pd.DataFrame): The dataframe containing the data
        solution (scipy.integrate._ivp.ivp.OdeSolution): The solution

    Returns:
        None"""
    col = ["U","u'u'","v'v'","w'w'","u'v'"]
    for i in range(5):
        plt.figure(figsize=(12, 6))
        df1 = df[df['y^+'].isin(solution.t)]
        subset = df1[col[i]]
        if col[i] == "u'v'":
            solution.y[i] = - solution.y[i] 
            subset = -1 * df1[col[i]]
        plt.plot(solution.t, solution.y[i],label = f"{col[i]} (PySINDy)")
        plt.plot(solution.t, subset.values, label=f"{col[i]} (DNS)")
        plt.xscale('log')
        plt.xlim(left=0.1, right = 10000)
        plt.xlabel('y^+')
        if col[i] == "u'v'":
            plt.ylabel(f"- {col[i]}")
            plt.title(f"- {col[i]} as a function of y^+")
        else:
            plt.ylabel(col[i])
            plt.title(f"{col[i]} as a function of y^+")
        plt.legend()
        st.pyplot(plt)
