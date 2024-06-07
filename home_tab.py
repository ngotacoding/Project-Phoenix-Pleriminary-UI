import streamlit as st
from PIL import Image

def display_home():

    st.write(" ")
    VIDEO_URL = "https://www.youtube.com/watch?v=sHD_YaE2J_U"
    st.video(VIDEO_URL)
    st.header(':blue[Project background]')
    st.write("""
        The objective of this project is to develop a parsimonious machine learning model to predict the net settlement value of personal injury cases arising from car accidents. This model will utilize a minimal set of readily obtainable data points while maintaining acceptable accuracy in its predictions. The key elements of this objective include focusing on the net value, emphasizing the use of minimal data points for efficiency, clearly specifying that the model applies to car accident personal injury cases, and acknowledging the need for reliable predictions.
    """)

    st.header(':blue[Problem statement]')
    st.write("""
        Car accident victims in the US rely on personal injury lawyers to negotiate fair settlements with insurance companies. However, this process is imbalanced. Insurance companies utilize sophisticated algorithms like Colossus, which analyze vast datasets to calculate a low-ball settlement range. In contrast, personal injury lawyers primarily rely on their individual experience and limited firm data, lacking access to the same level of data-driven insights.
    """)

    st.header(':blue[How personal injury cases work]')
    st.write("""
        Liability refers to who is legally responsible for causing your injury. In a personal injury case, you (the plaintiff) will need to prove that the other party (the defendant) acted negligently or intentionally in a way that resulted in your harm. Damages represent the losses you suffered due to the injury. These can be categorized into two main types: economic damages and non-economic damages. Economic damages are the concrete financial losses you can document with bills and receipts, such as medical expenses, lost wages, property damage, and rehabilitation costs. Non-economic damages are more subjective losses that are harder to quantify, representing the pain and suffering, emotional distress, loss of enjoyment of life, or loss of companionship you've experienced due to the injury. To win a personal injury case, you'll need to successfully establish both liability and damages. An attorney can help you navigate this process and determine the best course of action for your specific situation.
    """)

    st.header(':blue[Factors considered in future pain]')
    st.write("""
        Medical prognosis is crucial as your doctor's opinion on the long-term effects of your injury includes the likelihood of chronic pain, limitations on activities, and the need for future medical care. The lawyer will consider how your pain will affect your daily living, hobbies, work capacity, and overall well-being in the future. Lawyers may analyze past settlements or verdicts in cases with comparable injuries to estimate the value of your future pain. Assigning a dollar value to future pain is inherently subjective, and it can be a point of contention between lawyers and insurance companies. Lawyers approach this by calling upon expert testimony from doctors with experience in pain management and using "day in the life" evidence, which could include journals detailing your pain experience or limitations imposed by your injury.
    """)

    st.header(':blue[Important questions in evaluating a personal injury case]')
    st.write("""
        When did the accident or injury occur? How did the injury occur, and what were the circumstances around it? Where did the accident or event take place? Was anyone else present who witnessed the injury occur? Have you seen a doctor or been treated for your injuries, and if so, do you have records of any treatments you received? What is the nature of your injuries? Have you had to miss work or have you lost income as a result of your injury? What kind of medical treatment have you received, and what are the associated costs? Have you had any previous injuries or claims that might affect your current case? What has been the impact of the injury on your daily life and activities?
    """)

    st.header(':blue[Project objectives and datasets]')
    st.write("""
        The goals for this project include creating an ML model with at least 60% accuracy, developing an API to access the model, and building a front-end to interact with the model. The datasets available for this project are as follows: Dataset #1 contains 56,290 rows and 46 columns, Dataset #2 includes 50 records in an unstructured format, Dataset #3 consists of detailed Canadian-government provided data which is largely unstructured, and Dataset #4 contains ICD-10-CM/PCS medical codes.
    """)

