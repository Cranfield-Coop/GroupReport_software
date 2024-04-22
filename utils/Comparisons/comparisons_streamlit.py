import streamlit as st
import pandas as pd
import streamlit as st
import pandas as pd
from utils.Comparisons.comparisons import *

columns = ["y^+", "U", "u'u'", "v'v'", "w'w'", "u'v'"]
columns1 = ["y^+", "U_pred", "u'u'_pred", "v'v'_pred", "w'w'_pred", "u'v'_pred"]
columns2 = ["y^+","U","uu (y^+) RST","vv (y^+)","uv RST (y^+)"]
columns3 = ["y^+","U k-w"]


def app():
    #st.image(logo_path, width=100)
    if "comparison" not in st.session_state:
        st.session_state.comparison = False
    st.write("# Turbulence Modelling Predictor")
    st.markdown('<p style="font-size: 15px; font-style: italic;"> ~Developed by Group 2 Cranfield CO-OP</p>',unsafe_allow_html=True) 
    options = ['DNS (Oden)', 'DNS', 'PySINDy', 'PINNs','RSM','k–ω']
    selected_models = st.multiselect("Please select the models to be compared:",options)
    if len(selected_models) == 0:
        st.session_state.comparison = False
        return None 
        
    if "PySINDy" in selected_models:
        st.header("PySINDy")
        uploaded_file_pysindy = st.file_uploader("\n\n**Please upload the CSV below.**", type=['csv'], key="PySINDy")
        if uploaded_file_pysindy is not None:
            df_PySINDy = pd.read_csv(uploaded_file_pysindy)
            if all(col in df_PySINDy.columns for col in columns):
                df_PySINDy = df_PySINDy[columns]
                st.success("File successfully uploaded!")
            else:
                st.error("Please upload a csv file with the columns y^+, U, u'u', v'v', w'w'and u'v' to proceed.")
                return None
        else:
            st.error("Please upload a csv file to proceed.")
            return None
    else:
        df_PySINDy = None
    
    if "PINNs" in selected_models:
        st.header("PINNs")
        uploaded_file_pinns = st.file_uploader("\n\n**Please upload the CSV below.**", type=['csv'], key="PINNs")
        if uploaded_file_pinns is not None:
            df_PINNs = pd.read_csv(uploaded_file_pinns)
            if all(col in df_PINNs.columns for col in columns1):
                df_PINNs = df_PINNs[columns1]
                st.success("File successfully uploaded!")
            else:
                st.error("Please upload a csv file with the columns y^+, U_pred, u'u'_pred, v'v'_pred, w'w'_pred and u'v'_pred to proceed.")
                return None
        else:
            st.error("Please upload a csv file to proceed.")
            return None
    else:
        df_PINNs = None

    if "DNS (Oden)" in selected_models:
        st.header("DNS (Oden)")
        selected_re = st.selectbox("Select a Reynolds Number:", ["5200", "2000", "1000", "550", "180"])
        df_DNS_oden = download_and_combine_data(selected_re)
    else:
        df_DNS_oden = None

    if "DNS" in selected_models:
        st.header("DNS")
        uploaded_file_dns = st.file_uploader("\n\n**Please upload the CSV below.**", type=['csv'], key="DNS")
        if uploaded_file_dns is not None:
            df_DNS = pd.read_csv(uploaded_file_dns)
            if all(col in df_DNS.columns for col in columns):
                df_DNS = df_DNS[columns]
                st.success("File successfully uploaded!")
            else:
                st.error("Please upload a csv file with the columns y^+, U, u'u', v'v', w'w'and u'v' to proceed.")
                return None
        else:
            st.error("Please upload a csv file to proceed.")
            return None
    else:
        df_DNS = None

    if "RSM" in selected_models:
        st.header("RSM")
        uploaded_file_rsm = st.file_uploader("\n\n**Please upload the CSV below.**", type=['csv'], key="RSM")
        if uploaded_file_rsm is not None:
            df_RSM = pd.read_csv(uploaded_file_rsm)
            if all(col in df_RSM.columns for col in columns2):
                st.success("File successfully uploaded!")
                df_RSM["u'u'"] = df_RSM["uu (y^+) RST"]
                df_RSM["v'v'"] = df_RSM["vv (y^+)"]
                df_RSM["u'v'"] = df_RSM["uv RST (y^+)"]
                df_RSM = df_RSM[["y^+","U","u'u'","v'v'","u'v'"]]
            else:
                st.error("Please upload a csv file with the columns y^+, uu (y^+) RST, vv (y^+), uv RST (y^+) to proceed.")
                return None
        else:
            st.error("Please upload a csv file to proceed.")
            return None
    else:
        df_RSM = None

    if "k–ω" in selected_models:
        st.header("k–ω")
        uploaded_file_kw = st.file_uploader("\n\n**Please upload the CSV below.**", type=['csv'], key="kw")
        if uploaded_file_kw is not None:
            df_kw = pd.read_csv(uploaded_file_kw)
            if all(col in df_kw.columns for col in columns3):
                st.success("File successfully uploaded!")
                df_kw["U"] = df_kw["U k-w"]
                df_kw = df_kw[["y^+","U"]]
            else:
                st.error("Please upload a csv file with the columns y^+ and U k-w to proceed.")
                return None
        else:
            st.error("Please upload a csv file to proceed.")
            return None
    else:
        df_kw = None
    
    if st.button("Comparison of selected models"):
        st.session_state.comparison = True

    if st.session_state.comparison:
        dfs = [df_PySINDy, df_PINNs, df_DNS, df_DNS_oden, df_RSM, df_kw]
        df = [None] * len(dfs)
        for i in range (len(dfs)):
            if dfs[i] is not None:
                df[i] = dfs[i]
        if len(dfs) > 1: 
            nb =  max(len(dataset) for dataset in df if dataset is not None)
            for i in range(len(dfs)):
                if df[i] is not None and len(df[i]) < nb:
                    df[i] = data_interpolation(df[i],nb)
            log_scale = st.checkbox("Use logarithmic scale")
            show_grid = st.checkbox("Show grid on the plots")
            display_plots(df[0], df[1], df[2], df[3], df[4], df[5],log_scale,show_grid)
        else:
            st.error("Please upload more than one valid dataset for comparison.")
       
