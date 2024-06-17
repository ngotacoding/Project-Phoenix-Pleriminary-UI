import streamlit as st
from streamlit_option_menu import option_menu
from home_tab import display_home
from analysis_tab import display_analysis
from prediction_tab import display_prediction
from team_tab import display_team
from PIL import Image

st.set_page_config(page_title="Omdena Personal Injury Claims Prediction",
                   page_icon='',
                   layout='wide')


def main():
    col1, col2 = st.columns((.5, 2))

    with col1:
        logo = Image.open('figures/phoenix.webp')
        st.image(logo)
        
    with col2:
        st.title(':orange[Phoenix - Utilizing Machine Learning for Enhanced Valuation of Personal Injury Claims]')
        st.subheader("_Omdena Innovation Project_")
        
    # Define Tabs
    selected = option_menu(
        menu_title=None,
        options=['Home', 'Prediction', 'Analysis', 'Team'],
        icons=['house-fill', 'x-diamond-fill', 'bar-chart-fill', 'person-fill'],
        orientation='horizontal',
    )
    
    # Define Tab behaviour
    if selected == 'Home':
        display_home()
    elif selected == 'Prediction':
        display_prediction()
    elif selected == 'Analysis':
        display_analysis()
    elif selected == 'Team':
        display_team()

if __name__ == "__main__":
    main()
