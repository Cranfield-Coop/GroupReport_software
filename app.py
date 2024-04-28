import streamlit as st
from streamlit_option_menu import option_menu
import utils.PySINDy.pysindy_streamlit as pysindy_streamlit
import utils.PINNs.pinns_streamlit as pinns_streamlit
import utils.Comparisons.comparisons_streamlit as comparisons_streamlit


# Configure the Streamlit application page
st.set_page_config(page_title="Turbulence Modelling Predictor",)


logo_path = "img/cranfield_logo.png"

class MultiApp:

    def run():
        """
        Displays functional modules using the Streamlit sidebar and manages navigation between them.
        """
        with st.sidebar:
            # Display the logo in the application sidebar
            st.sidebar.image(logo_path, use_column_width=True)      
            # Create a menu in the application
            app = option_menu(
                menu_title='Menu',
                options=['PySINDy','PINNs','Comparisons'],
                menu_icon='chat-text-fill',
                default_index=0,
                styles={
                    "container": {"padding": "5!important","background-color":'white'},
        "icon": {"black": "#9ca1a1", "font-size": "23px"}, 
        "nav-link": {"color":"black","font-size": "20px", "text-align": "left", "margin":"0px", "--hover-color": "#9ca1a1"},
        "nav-link-selected": {"background-color": "#c0c4c4"},}
                
                )

        # Select the module to be used from the menu
        if app == "PySINDy":
            pysindy_streamlit.app() # Call the PySINDy app function
        if app == "PINNs":
            pinns_streamlit.app() # Call the PINNs app function
        if app == "Comparisons":
            comparisons_streamlit.app() # Call the comparisons app function
             
       
    # Run the app
    run()
