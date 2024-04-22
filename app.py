import streamlit as st
from streamlit_option_menu import option_menu
import utils.PySINDy.pysindy_streamlit as pysindy_streamlit
import utils.PINNs.pinns_streamlit as pinns_streamlit
import utils.Comparisons.comparisons_streamlit as comparisons_streamlit

st.set_page_config(
        page_title="Turbulence Modelling Predictor",
)


logo_path = "img/cranfield_logo.png"

class MultiApp:

    def __init__(self):
        self.apps = []

    def add_app(self, title, func):

        self.apps.append({
            "title": title,
            "function": func
        })

    def run():
        # app = st.sidebar(
        with st.sidebar:
            st.sidebar.image(logo_path, use_column_width=True)      
            app = option_menu(
                menu_title='Menu',
                options=['PySINDy','PINNs','Comparisons'],
                #icons=['house-fill','person-circle','trophy-fill','chat-fill','info-circle-fill'],
                menu_icon='chat-text-fill',
                default_index=0,
                styles={
                    "container": {"padding": "5!important","background-color":'white'},
        "icon": {"black": "#9ca1a1", "font-size": "23px"}, 
        "nav-link": {"color":"black","font-size": "20px", "text-align": "left", "margin":"0px", "--hover-color": "#9ca1a1"},
        "nav-link-selected": {"background-color": "#c0c4c4"},}
                
                )

        if app == "PySINDy":
            pysindy_streamlit.app()
        if app == "PINNs":
            pinns_streamlit.app()
        if app == "Comparisons":
            comparisons_streamlit.app()
             
       
  
    run()
