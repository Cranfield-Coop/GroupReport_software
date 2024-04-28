import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp
from sklearn.metrics import mean_squared_error


def plot(df, Reynolds_Numbers):
    """
    Plot the data for each Reynolds number.
    
    Args:
        df (pd.DataFrame): The dataframe containing the DNS data
        Reynolds_Numbers (list): The list of Reynolds numbers

    Returns:
        None
    """
    
    columns = ["U","u'u'","v'v'","w'w'","u'v'"]
    
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
        plt.close()


def data_interpolation(df, nb, Reynolds_Numbers):
    """
    Interpolates data for each column in the DataFrame over a specified number of points.

    Args:
        df (pd.DataFrame): The dataframe containing the DNS data
        nb (int): The number of points to interpolate
        Reynolds_Number (float): The Reynolds number to be assigned to the interpolated dataframe

    Returns:
        pd.DataFrame: The interpolated dataframe  
    """
    # Generate nb points for the 'y/delta' column to space it evenly from its minimum to its maximum value
    interval = np.linspace(df['y^+'].min(), df['y^+'].max(), num=nb)

    # Initialise a dictionary to contain the interpolated data
    interpolated_columns = {}
    
    # Interpolate the data in each column of the dataframe with the for loop
    for column in df.columns:
        if column != 'y^+':
            # Create a interpolation function for each column
            f = interp1d(df['y^+'], df[column], kind='linear')
            # Apply the interpolation function
            interpolated_columns[column] = f(interval)
    # Create a dataframe with interpolated data
    df = pd.DataFrame(interpolated_columns)
    
    # Add the interpolated points y^+ and add the constant Reynolds number as columns in the dataframe
    df['y^+'] = interval
    df["Re_tau"] = Reynolds_Numbers
    
    return df

def first_derivative(columns,df):
    """
    Compute the first derivative of the specified columns.
    
    Args:
        columns (list): List of column names in the dataframe for which the first derivative is to be calculated
        df (pd.DataFrame): The dataframe containing the DNS data
       
    Returns:
        pd.DataFrame: The input dataframe with the additional columns representing the first derivative 
        of the columns present in the list (columns)
    """
    for col in columns:  
        # Extract the data from a column and y^+ into numpy arrays
        x = df[col].to_numpy()
        y = df["y^+"].to_numpy()
        
        # Calculate the first derivative of the column with respect to y^+
        dx = np.gradient(x,y)
        
        # Store the result of the first derivative of the column in the dataframe
        df["d" + col + "/dy"] = dx
        
    return df

def second_derivative(columns,df):
    """
    Compute the second derivative of the specified columns.
    
    Args:
        columns (list): List of column names in the dataframe for which the second derivative is to be calculated
        df (pd.DataFrame): The dataframe containing the DNS data
       
    Returns:
        pd.DataFrame: The input dataframe with the additional columns representing the second derivative 
        of the columns present in the list (columns)
    """
    for col in columns: 
        # Extract the data from a column and y^+ into numpy arrays
        x = df[col].to_numpy()
        y = df["y^+"].to_numpy()
        
        # Calculate the first derivative of the column with respect to y^+
        dx = np.gradient(x,y)
        
        # Store the result of the first derivative of the column in the dataframe
        df["d(" + col + ")/dy"] = dx
        
    return df


def data_processing(df, interpolation, nb_interpolation, Reynolds_Numbers):
    """
    Process, interpolate the DNS data and compute the first and the second derivatives of the specified columns 
    for different Reynolds numbers
    
    Args:
        df (pd.DataFrame): The dataframe containing the DNS data
        interpolation (int): The interpolation method
        nb_interpolation (int): The number of points to interpolate
        Reynolds_Numbers (list): The list of Reynolds numbers
    
    Returns:
        dict: The processed data
    """

    # Checks whether interpolation mode has been selected
    if interpolation == 0:
        # Dictionary containing dataframes associated with a specific number of reynolds
        data = {f'df_{Re}': df[df["Re_tau"] == Re] for Re in Reynolds_Numbers}
        
        # Calculate the first and second derivatives of the specified columns for each dataframe in the dictionary
        for Re in Reynolds_Numbers:
            columns_to_derive = ["u'u'", "v'v'", "w'w'", "u'v'", "P"]
            data[f"df_{Re}"] = first_derivative(columns_to_derive, data[f"df_{Re}"])
            columns_to_derive = ["dU/dy", "du'u'/dy","dv'v'/dy", "dw'w'/dy", "du'v'/dy", "dP/dy"]
            data[f"df_{Re}"] = second_derivative(columns_to_derive, data[f"df_{Re}"])
        return data
    else:
        # Dictionary containing dataframes associated with a specific number of reynolds
        data = {f'df_{Re}': df[df["Re_tau"] == Re].drop(columns=["Re_tau"]) for Re in Reynolds_Numbers}
        
        # Interpolate the data from each dataframe in the dictionary 
        for Re in Reynolds_Numbers:
            data[f"df_{Re}"] = data_interpolation(data[f"df_{Re}"], nb_interpolation, Re)
        
        # Calculate the first and second derivatives of the specified columns for each dataframe in the dictionary
        for Re in Reynolds_Numbers:
            columns_to_derive = ["u'u'", "v'v'", "w'w'", "u'v'", "P"]
            data[f"df_{Re}"] = first_derivative(columns_to_derive, data[f"df_{Re}"])
            columns_to_derive = ["dU/dy", "du'u'/dy","dv'v'/dy", "dw'w'/dy", "du'v'/dy", "dP/dy"]
            data[f"df_{Re}"] = second_derivative(columns_to_derive, data[f"df_{Re}"])
        return data


def data_PySINDy(data,Reynolds_Number):
    """
    Prepare the data for PySINDy
   
    Args:
        data (dict): The processed data
        Reynolds_Numbers (list): The list of Reynolds numbers
    
    Returns:
        list: The input data (containing the feature variables)
        list: The output data (containing the target variables)
    """
    # Initialise the lists containing the feature variables (X) and target variables (y) for each Reynolds number
    X = [[] for _ in range(len(Reynolds_Number))]
    y = [[] for _ in range(len(Reynolds_Number))]
    
    for i in range(0,len(Reynolds_Number)):
        # Extract the feature variables 
        X[i] = data[f"df_{Reynolds_Number[i]}"].drop(columns=["y^+","Re_tau","d(dU/dy)/dy","d(du'u'/dy)/dy", "d(dv'v'/dy)/dy", "d(dw'w'/dy)/dy", "d(du'v'/dy)/dy","d(dP/dy)/dy"]).values
        
        # Extract the target variables 
        y[i] = data[f"df_{Reynolds_Number[i]}"][["dU/dy", "du'u'/dy", "dv'v'/dy", "dw'w'/dy", "du'v'/dy","dP/dy","d(dU/dy)/dy","d(du'u'/dy)/dy", "d(dv'v'/dy)/dy", "d(dw'w'/dy)/dy", "d(du'v'/dy)/dy","d(dP/dy)/dy"]].values
    
    return X,y

def test_overfitting(X_train, y_train, X_test, y_test, model):
    """
    Test if the model is overfitting.
    
    Args:
        X_train (list): The input training data
        y_train (list): The output training data
        X_test (list): The input testing data
        y_test (list): The output testing data
        model (dict): The PySINDy model

    Returns:
        None
    """

    # Use the model to predict values based on the train features
    y_train_pred = model.predict(X_train)
    # Compute the train MSE value
    mse_train = mean_squared_error(y_train, y_train_pred)

    # Use the model to predict values based on the train features
    y_test_pred = model.predict(X_test)
    # Compute the test MSE value
    mse_test = mean_squared_error(y_test, y_test_pred)
    
    # Check that there is no overfitting by comparing the training and test MSEs
    if abs(mse_train - mse_test) > 0.001:
        st.metric(label="PySINDy Model", 
                  value="Overfitting !",
                  delta=None,
                  delta_color="normal",
                  help=None,
                  label_visibility="visible")
    else:
        st.metric(label="PySINDy Model", 
                  value="No Overfitting !",
                  delta=None,
                  delta_color="normal",
                  help=None,
                  label_visibility="visible")


def integrate(y_plus, y, coeffs):    
    """
    Computes the derivatives at a point for integration.

    Args:
        y_plus (float): The spatial variable for the integration
        y (array): Current state variables
        coeffs (array): Coefficients of the trained PySINDy Model

    Returns:
        list: Derivatives of the state variables and their second derivatives
    """
    
    # Initialise a variable to store the derivatives
    derivatives = []
    
    # Calculate the derivative for each state variable using the coefficients of the trained PySINDy Model
    for i in range(len(y)):
        dd = coeffs[i][0] + np.dot(coeffs[i][1:], y) 
        derivatives.append(dd)
    
    return derivatives


def simulation(X, df, model):
    """
    Simulate the PySINDy model using numerical integration.
    
    Args:
        X (list): The input data
        df (pd.DataFrame): The dataframe containing the DNS data
        model (object): The trained PySINDy Model
        
    Returns:
        OdeResult: The result of the simulation
    """
    
    # Extract the initial conditions 
    initial_conditions = X[0]
    
    # Extract the coefficients of the PySINDy Model
    coeffs = model.optimizer.coef_
    
    # Extract the y^+ values of the dataframe
    y_plus_points = df["y^+"].values
    
    # Run the integration over the range of y^+ points
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
    """
    Plot the simulation
    
    Args:
        df (pd.DataFrame): The dataframe containing the DNS data
        solution (scipy.integrate._ivp.ivp.OdeSolution): The solutions of the PySINDy Simulation

    Returns:
        None
    """

    col = ["U","u'u'","v'v'","w'w'","u'v'"]
    
    for i in range(5):
        
        plt.figure(figsize=(12, 6))
        
        # Extract the y^+ values of the dataframe
        y = df['y^+'].values
        
        subset_pred = solution.y[i]
        subset_target = df[col[i]]
        
        if col[i] == "u'v'":
            subset_pred = -1 * solution.y[i] 
            subset_target = -1 * df[col[i]]
        
        # Plot predictions and DNS data
        plt.plot(solution.t, subset_pred,color="blue",linestyle="-",label = f"{col[i]} (PySINDy)")
        plt.plot(y, subset_target.values, color="red",linestyle="--",label=f"{col[i]} (DNS)")
        
        plt.xscale('log')
        plt.xlim(left=0.1, right = 10000)
        plt.xlabel('y^+')
        
        if col[i] == "u'v'":
            plt.ylabel(f"- {col[i]}")
            plt.title(f"- {col[i]} Prediction vs. DNS")
        else:
            plt.ylabel(col[i])
            plt.title(f"{col[i]} Prediction vs. DNS")
        
        plt.legend()
        st.pyplot(plt)
        plt.close()


