import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import requests
from scipy.interpolate import interp1d
import io
import re


def skip_header(content):
    """Find dynamiccly the number of lines to skip in the header of the file."""
    lines = content.split("\n")
    lines_to_skip = 0
    for line in lines:
        if line.strip().startswith("%"):
            lines_to_skip += 1
        else:
            break
    return lines_to_skip


def get_header(content):
    """Get dynamiccly the header of the dataset."""
    lines = content.split("\n")
    line = lines[skip_header(content) - 2]
    header = re.split(r"\s+", line.strip())[1:]
    return header


def download_and_combine_data(Re_tau):
    parameters = ["mean","vel_fluc"]
    for i, parameter in enumerate(parameters):
        url = f"https://turbulence.oden.utexas.edu/channel2015/data/LM_Channel_{Re_tau}_{parameter}_prof.dat"
        response = requests.get(url)
        if response.status_code == 200:
            file_content = response.text
            file_like_object = io.StringIO(file_content)

            df = pd.read_csv(
                file_like_object,
                sep=r"\s+",
                skiprows=skip_header(file_content),
                names=get_header(file_content),
            )
            df["Re_tau"] = Re_tau
            if i < 1:
                df1 = df
            else:
                df2 = df
                df = pd.merge(df1, df2,  how='left', left_on=['y/delta','y^+','Re_tau'], right_on = ['y/delta','y^+','Re_tau'])           
        else:
            st.write(f"Failed to get the DNS Data (Oden) for Re_tau={Re_tau}. Status code: {response.status_code}")
    return df


def data_interpolation(df, nb):
    """
    Interpolate the data to have the same number of points for each Reynolds number
    Args:
        df (pd.DataFrame): The dataframe containing the data
        nb (int): The number of points to interpolate

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
    return df

def display_plots(df_PySINDy=None, df_PINNs=None, df_DNS=None, df_DNS_Oden=None, df_RSM=None, df_kw=None,log_scale=None,grid=None):
    col = ["U","u'u'","v'v'","w'w'","u'v'"]
    for i in range(5):
        plt.figure(figsize=(12, 6))
        if col[i] == "U":
            if df_kw is not None:
                kw = df_kw[col[i]]
                y = df_kw["y^+"]
        if df_DNS_Oden is not None:
            dns_oden = df_DNS_Oden[col[i]]
            y = df_DNS_Oden["y^+"]
        if col[i] != "w'w'":
            if df_RSM is not None:
                rsm = df_RSM[col[i]]
                y = df_RSM["y^+"]
        if df_PySINDy is not None:
            pysindy = df_PySINDy[col[i]]
            y = df_PySINDy["y^+"]
        if df_PINNs is not None:
            pinns = df_PINNs[f"{col[i]}_pred"]
            y = df_PINNs["y^+"]
        if col[i] == "u'v'":
            if df_DNS_Oden is not None:
                dns_oden = -1 * df_DNS_Oden[col[i]] 
            if df_PySINDy is not None:
                pysindy = -1 * df_PySINDy[col[i]]
            if df_RSM is not None:
                rsm = -1 * df_RSM[col[i]]
            if df_PINNs is not None:
                pinns = -1 * df_PINNs[f"{col[i]}_pred"]
        if df_PySINDy is not None:
            plt.plot(y.values, pysindy.values,linestyle="-",color="green",label = f"{col[i]} (PySINDy)")
        if df_PINNs is not None:
            plt.plot(y.values, pinns.values,linestyle="-",color="blue",label = f"{col[i]} (PINNs)")
        if df_DNS_Oden is not None:
            plt.plot(y.values, dns_oden.values,linestyle="--",color="red" ,label=f"{col[i]} (DNS Oden)")
        if df_RSM is not None:
            plt.plot(y.values, rsm.values,linestyle="-",color="black" ,label=f"{col[i]} (RSM)")
        if col[i] == "U":
            if df_kw is not None:
                plt.plot(y.values, kw.values,linestyle="-",color="orange" ,label=f"{col[i]} (k–ω)")
        if log_scale:
            plt.xscale('log')
        plt.xlabel('y^+')
        if col[i] == "u'v'":
            plt.ylabel(f"- {col[i]}")
            plt.title(f"- {col[i]} as a function of y^+")
        else:
            plt.ylabel(col[i])
            plt.title(f"{col[i]} as a function of y^+")
        if grid:    
            plt.grid(True, which="both", linestyle='--', linewidth=0.5)
        plt.legend()
        st.pyplot(plt)
        plt.close()




