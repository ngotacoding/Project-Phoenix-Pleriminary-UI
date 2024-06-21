import streamlit as st

def display_home():

    cols = st.columns([.2,5,.4])
    with cols[1]:
        with st.container():
            
            st.header(':blue[Project Background]')
            st.write("""
                The goal of this project is to design a streamlined machine learning model that accurately predicts the net settlement values of car accident-related personal injury claims. The model will utilize a concise set of easily accessible data points, ensuring both efficiency and reliability in its predictions. It specifically addresses settlements from car accidents and focuses on achieving credible results with minimal data.
            """)

            st.header(':blue[Problem Statement]')
            st.write("""
                In the U.S., victims of car accidents depend on personal injury lawyers to secure equitable settlements from insurance companies. This process often disadvantages plaintiffs, as insurers employ advanced algorithms like Colossus to propose minimal settlement amounts based on extensive data analysis. In contrast, personal injury lawyers typically rely on their own experiences and limited data, lacking comparable analytical tools.
            """)

            st.header(':blue[How Personal Injury Cases Work]')
            st.write("""
                Liability in personal injury cases determines who is legally at fault for causing harm. Plaintiffs must prove that the defendant's negligence or intentional actions caused their injuries. Compensation covers two types of damages: economic (quantifiable financial losses such as medical bills and lost wages) and non-economic (subjective impacts like pain and suffering). Successfully claiming these requires demonstrating both liability and the extent of damages, often with legal assistance.
              This often involves assessing future pain. 
              
              Assessing future pain involves evaluating medical opinions on long-term injury impacts, including chronic pain and lifestyle restrictions. Lawyers use this information to predict how pain will affect a client’s daily life and future well-being. Estimating monetary compensation for future pain involves expert medical testimony and personal evidence, like pain journals, which help in negotiating with insurance companies.
            """)

            st.header(':blue[Important Questions in Evaluating a Personal Injury Case]')
            st.write("""
                Critical inquiries in personal injury evaluations include:
                - When and where did the accident occur?
                - How did the injury happen, and what were the circumstances?
                - Were there any witnesses?
                - Have you received medical treatment, and are there records?
                - What is the nature of your injuries, and have they caused work absences or income loss?
                - What are your medical expenses?
                - Do you have a history of similar injuries or claims?
                - How has the injury affected your daily activities and quality of life?
            """)

            st.subheader(':blue[Project Objectives and Datasets]')
            st.write("""
                The goals for this project were to create an ML model with at least 60% accuracy, and building a front-end to interact with the model. """)
