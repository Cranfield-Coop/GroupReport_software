import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import requests
from scipy.interpolate import interp1d
import io
import re


def skip_header(content):
    """
    Find dynamiccly the number of lines to skip in the header of the file.
    """
    lines = content.split("\n")
    lines_to_skip = 0
    for line in lines:
        if line.strip().startswith("%"):
            lines_to_skip += 1
        else:
            break
    return lines_to_skip


def get_header(content):
    """
    Get dynamiccly the header of the dataset.
    """
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


def display_plots(df_PySINDy=None, df_PINNs=None, df_DNS=None, df_DNS_Oden=None, df_RSM=None, df_kw=None,log_scale=None,grid=None):
    """
    Plot the predicted values from multiple turbulence models.
    
    Args:
        df_PySINDy (pd.DataFrame): The dataframe containing PySINDy results
        df_PINNs (pd.DataFrame): The dataframe containing PINNs results
        df_DNS (pd.DataFrame): The dataframe containing the DNS data
        df_DNS_Oden (pd.DataFrame): The dataframe containing the DNS data (from Oden)
        df_RSM (pd.DataFrame): The dataframe containing RSM results
        df_kw (pd.DataFrame): The dataframe containing k-w results
        log_scale (bool): If true, use logarithmic scale 
        grid (bool): If true, display the grid on the plots

    Returns:
        None
    """
    # Define the names of the columns to be plotted
    col = ["U","u'u'","v'v'","w'w'","u'v'"]

    for i in range(5):
        plt.figure(figsize=(12, 6))

        # For U, include data from the k-w model  
        if col[i] == "U":
            if df_kw is not None:
                kw = df_kw[col[i]]
                y_kw = df_kw["y^+"]
        
        # Retrieve data from the DNS model (Oden) if available
        if df_DNS_Oden is not None:
            dns_oden = df_DNS_Oden[col[i]]
            y_dns_oden = df_DNS_Oden["y^+"]
        
        # Retrieve data from the DNS model if available
        if df_DNS is not None:
            dns = df_DNS[col[i]]
            y_dns = df_DNS["y^+"]
        
        # Except for w'w', include data from the RSM model
        if col[i] != "w'w'":
            if df_RSM is not None:
                rsm = df_RSM[col[i]]
                y_rsm = df_RSM["y^+"]
        
        # Retrieve data from the PySINDy model (Oden) if available
        if df_PySINDy is not None:
            pysindy = df_PySINDy[col[i]]
            y_pysindy = df_PySINDy["y^+"]
        
        # Retrieve data from the PINNs model (Oden) if available
        if df_PINNs is not None:
            pinns = df_PINNs[f"{col[i]}_pred"]
            y_pinns = df_PINNs["y^+"]
        

        if col[i] == "u'v'":
            if df_DNS_Oden is not None:
                dns_oden = -1 * df_DNS_Oden[col[i]] 
            if df_DNS is not None:
                dns = -1 * df_DNS[col[i]] 
            if df_PySINDy is not None:
                pysindy = -1 * df_PySINDy[col[i]]
            if df_RSM is not None:
                rsm = -1 * df_RSM[col[i]]
            if df_PINNs is not None:
                pinns = -1 * df_PINNs[f"{col[i]}_pred"]
        
        # Plot data for each model if available
        if df_PySINDy is not None:
            plt.plot(y_pysindy.values, pysindy.values,linestyle="-",color="green",label = f"{col[i]} (PySINDy)")
        if df_PINNs is not None:
            plt.plot(y_pinns.values, pinns.values,linestyle="-",color="blue",label = f"{col[i]} (PINNs)")
        if df_DNS_Oden is not None:
            plt.plot(y_dns_oden.values, dns_oden.values,linestyle="--",color="red" ,label=f"{col[i]} (DNS Oden)")
        if df_DNS is not None:
            plt.plot(y_dns.values, dns.values,linestyle="--",color="yellow" ,label=f"{col[i]} (DNS)")
        if col[i] != "w'w'":
            if df_RSM is not None:
                plt.plot(y_rsm.values, rsm.values,linestyle="-",color="black" ,label=f"{col[i]} (RSM)")
        if col[i] == "U":
            if df_kw is not None:
                plt.plot(y_kw.values, kw.values,linestyle="-",color="orange" ,label=f"{col[i]} (k–ω)")
        
        # Set logarithmic scale if enabled
        if log_scale:
            plt.xscale('log')
        # Display the grid on the plots if enabled
        if grid:    
            plt.grid(True, which="both", linestyle='--', linewidth=0.5)

        plt.xlabel('y^+')

        if col[i] == "u'v'":
            plt.ylabel(f"- {col[i]}")
            plt.title(f"- {col[i]} as a function of y^+")
        else:
            plt.ylabel(col[i])
            plt.title(f"{col[i]} as a function of y^+")


        plt.legend()
        st.pyplot(plt)
        plt.close()




