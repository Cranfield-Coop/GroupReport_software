import streamlit as st
import pandas as pd
import streamlit as st
import pandas as pd
from utils.Comparisons.comparisons import *

# Define the relevant columns for the PySINDy model 
columns_pysindy = ["y^+", "U", "u'u'", "v'v'", "w'w'", "u'v'"]
# Define the relevant columns for the PINNs model 
columns_pinns = ["y^+", "U_pred", "u'u'_pred", "v'v'_pred", "w'w'_pred", "u'v'_pred"]
# Define the relevant columns for the RSM  
columns_rsm = ["y^+","U","uu (y^+) RST","vv (y^+)","uv RST (y^+)"]
# Define the relevant columns for the kw model 
columns_kw = ["y^+","U k-w"]


def app():
    if "comparison" not in st.session_state:
        st.session_state.comparison = False
    
    st.write("# Turbulence Modelling Predictor")
    st.markdown('<p style="font-size: 15px; font-style: italic;"> ~Developed by Group 2 Cranfield CO-OP</p>',unsafe_allow_html=True) 
    
    # Selection of models to compare
    options = ['DNS (Oden)', 'DNS', 'PySINDy', 'PINNs','RSM','k–ω']
    selected_models = st.multiselect("Please select the models to be compared:",options)
    
    # Check whether any models have been selected for the comparison
    if len(selected_models) == 0:
        st.session_state.comparison = False
        return None 
        
    if "PySINDy" in selected_models:
        st.header("PySINDy")

        # File uploader for users to upload a CSV file for PySINDy
        uploaded_file_pysindy = st.file_uploader("\n\n**Please upload the CSV below.**", type=['csv'], key="PySINDy")

        # Check if a csv file has been uploaded
        if uploaded_file_pysindy is not None:
            
            df_PySINDy = pd.read_csv(uploaded_file_pysindy)

            # Check if the uploaded CSV file contains all the necessary columns
            if all(col in df_PySINDy.columns for col in columns_pysindy):
                df_PySINDy = df_PySINDy[columns_pysindy]
                st.success("File successfully uploaded!")
            else:
                st.error("Please upload a csv file with the columns y^+, U, u'u', v'v', w'w'and u'v' to proceed.")
                return None
        else:
            # Error message if no file is uploaded
            st.error("Please upload a csv file to proceed.")
            return None
    else:
        df_PySINDy = None
    
    if "PINNs" in selected_models:
        st.header("PINNs")
        
        # File uploader for users to upload a CSV file for PINNs
        uploaded_file_pinns = st.file_uploader("\n\n**Please upload the CSV below.**", type=['csv'], key="PINNs")
        
        # Check if a csv file has been uploaded
        if uploaded_file_pinns is not None:
            
            df_PINNs = pd.read_csv(uploaded_file_pinns)

            # Check if the uploaded CSV file contains all the necessary columns
            if all(col in df_PINNs.columns for col in columns_pinns):
                df_PINNs = df_PINNs[columns_pinns]
                st.success("File successfully uploaded!")
            else:
                st.error("Please upload a csv file with the columns y^+, U_pred, u'u'_pred, v'v'_pred, w'w'_pred and u'v'_pred to proceed.")
                return None
        else:
            # Error message if no file is uploaded
            st.error("Please upload a csv file to proceed.")
            return None
    else:
        df_PINNs = None

    if "DNS (Oden)" in selected_models:
        st.header("DNS (Oden)")
        # Select a Reynolds number for the extraction of the data present on the Oden site associated with this selected Reynolds number
        selected_re = st.selectbox("Select a Reynolds Number:", ["5200", "2000", "1000", "550", "180"])
        df_DNS_oden = download_and_combine_data(selected_re)
    else:
        df_DNS_oden = None

    if "DNS" in selected_models:
        st.header("DNS")

        # File uploader for users to upload a CSV file for DNS
        uploaded_file_dns = st.file_uploader("\n\n**Please upload the CSV below.**", type=['csv'], key="DNS")
        
        # Check if a csv file has been uploaded
        if uploaded_file_dns is not None:
            
            df_DNS = pd.read_csv(uploaded_file_dns)

            # Check if the uploaded CSV file contains all the necessary columns
            if all(col in df_DNS.columns for col in columns_pysindy):
                df_DNS = df_DNS[columns_pysindy]
                st.success("File successfully uploaded!")
            else:
                st.error("Please upload a csv file with the columns y^+, U, u'u', v'v', w'w'and u'v' to proceed.")
                return None
        else:
            # Error message if no file is uploaded
            st.error("Please upload a csv file to proceed.")
            return None
    else:
        df_DNS = None

    if "RSM" in selected_models:
        st.header("RSM")

        # File uploader for users to upload a CSV file for RSM
        uploaded_file_rsm = st.file_uploader("\n\n**Please upload the CSV below.**", type=['csv'], key="RSM")
        
        # Check if a csv file has been uploaded
        if uploaded_file_rsm is not None:
            
            df_RSM = pd.read_csv(uploaded_file_rsm)

            # Check if the uploaded CSV file contains all the necessary columns
            if all(col in df_RSM.columns for col in columns_rsm):
                st.success("File successfully uploaded!")
                df_RSM["u'u'"] = df_RSM["uu (y^+) RST"]
                df_RSM["v'v'"] = df_RSM["vv (y^+)"]
                df_RSM["u'v'"] = df_RSM["uv RST (y^+)"]
                df_RSM = df_RSM[["y^+","U","u'u'","v'v'","u'v'"]]
            else:
                st.error("Please upload a csv file with the columns y^+, uu (y^+) RST, vv (y^+), uv RST (y^+) to proceed.")
                return None
        else:
            # Error message if no file is uploaded
            st.error("Please upload a csv file to proceed.")
            return None
    else:
        df_RSM = None

    if "k–ω" in selected_models:
        st.header("k–ω")
        
        # File uploader for users to upload a CSV file for k-w 
        uploaded_file_kw = st.file_uploader("\n\n**Please upload the CSV below.**", type=['csv'], key="kw")
        
        # Check if a csv file has been uploaded
        if uploaded_file_kw is not None:
            
            df_kw = pd.read_csv(uploaded_file_kw)

            # Check if the uploaded CSV file contains all the necessary columns
            if all(col in df_kw.columns for col in columns_kw):
                st.success("File successfully uploaded!")
                df_kw["U"] = df_kw["U k-w"]
                df_kw = df_kw[["y^+","U"]]
            else:
                st.error("Please upload a csv file with the columns y^+ and U k-w to proceed.")
                return None
        else:
            # Error message if no file is uploaded
            st.error("Please upload a csv file to proceed.")
            return None
    else:
        df_kw = None
    
    # Run the comparison of the selected models
    if st.button("Comparison of selected models"):
        st.session_state.comparison = True

    if st.session_state.comparison:
        # List containing the dataframes of different models
        dfs = [df_PySINDy, df_PINNs, df_DNS, df_DNS_oden, df_RSM, df_kw]

        # Initialise a list to store possibly interpolated dataframes
        df = [None] * len(dfs)

        # Iterate on the dataframes associated with the models
        for i in range (len(dfs)):
            # Checks whether the dataframe associated with this model has been uploaded
            if dfs[i] is not None:
                df[i] = dfs[i]

        log_scale = st.checkbox("Use logarithmic scale")

        # Allow the user to choose whether to use a grid for the plots
        show_grid = st.checkbox("Show grid on the plots")

        # Display the plots for the uploaded models 
        display_plots(df[0], df[1], df[2], df[3], df[4], df[5],log_scale,show_grid)

