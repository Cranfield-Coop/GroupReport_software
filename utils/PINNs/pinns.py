import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import tempfile
from utils.PINNs.run_inference import *
from utils.PINNs.run_test_inference import *
from utils.PINNs.prepare_prediction_csv import *


def display_plots(df):
    st.subheader("Prediction Values Plots")
    col = ["U_pred", "u'u'_pred", "v'v'_pred", "w'w'_pred", "u'v'_pred"]
    for i in range(5):
        plt.figure(figsize=(12, 6))
        subset = df[col[i]]
        if col[i] == "u'v'_pred":
            subset = -1 * df[col[i]]
        plt.plot(df["y^+"].values, subset.values, label=f"{col[i]} (PINNs)")
        plt.xscale("log")
        plt.xlabel("y^+")
        plt.ylabel(col[i])
        plt.title(f"{col[i]} as a function of y^+")
        plt.legend()
        st.pyplot(plt)
        plt.close()


def format_dataframe(df):
    return df.applymap(lambda x: f"{x:.2e}" if isinstance(x, float) else x)


def display_test_plots(df):
    st.subheader("Prediction vs. Target Values Plots")
    pred_cols = ["U_pred", "u'u'_pred", "v'v'_pred", "w'w'_pred", "u'v'_pred"]
    target_cols = [
        "U_target",
        "u'u'_target",
        "v'v'_target",
        "w'w'_target",
        "u'v'_target",
    ]

    for i in range(5):
        plt.figure(figsize=(12, 6))

        # Plotting predicted values
        subset_pred = df[pred_cols[i]]
        plt.plot(
            df["y^+"].values,
            subset_pred.values,
            label=f"{pred_cols[i]} (PINNs)",
            linestyle="-",
            color="blue",
        )

        # Plotting target values
        subset_target = df[target_cols[i]]
        plt.plot(
            df["y^+"].values,
            subset_target.values,
            label=f"{target_cols[i]} (Target)",
            linestyle="--",
            color="red",
        )

        plt.xscale("log")
        plt.xlabel("y^+")
        plt.ylabel(pred_cols[i].replace("_pred", ""))
        plt.title(f"{pred_cols[i].replace('_pred', '')} Prediction vs. Target")
        plt.legend()
        st.pyplot(plt)
        plt.close()


def display_metrics(metrics):
    st.subheader("Metrics")
    st.metric(
        "Total MSE",
        metrics["mse_total"][0],
        delta=None,
        delta_color="normal",
        help=None,
        label_visibility="visible",
    )
    st.metric(
        "Total RMSE",
        metrics["rmse_total"][0],
        delta=None,
        delta_color="normal",
        help=None,
        label_visibility="visible",
    )
    st.metric(
        "Total R2",
        metrics["r2_total"][0],
        delta=None,
        delta_color="normal",
        help=None,
        label_visibility="visible",
    )
    st.dataframe(metrics)


