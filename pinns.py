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
import tempfile


# Define the simulation function
def simulation_pinns(df):
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


