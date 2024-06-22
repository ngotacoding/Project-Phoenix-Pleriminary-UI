import streamlit as st
import numpy as np
from model_utils import predict_claim

# Function to define what happens when you click submit
def submit():
    """Define what happens when you click Submit.
    """
    
    # Predict the Total Claim Amount
    predicted_amount = predict_claim()
    
    # Root Mean Squared Error from modelling team
    rmse = 14423
    
    # Calculate the claim range
    lowest_amount = predicted_amount - rmse
    highest_amount = predicted_amount + rmse
    st.session_state["predicted_amount"] = predicted_amount
    st.session_state["highest_amount"] = highest_amount
    st.session_state["lowest_amount"] = lowest_amount if lowest_amount > 0 else 0
    st.session_state["form_submitted"] = True

# Function to define back button behavior
def back():
    """Define what happens when you click Back.
    """
    st.session_state["form_submitted"] = False

# Function to define the main form    
def show_form():
    """Define the outlook of the main form.
    """
    
    st.title(":green[Predict your Claim amount]")
    
    with st.container(border=1):
        
        # Demographic and Geographic Information
        st.header("Demographic and Geographic Information")
        st.session_state['age'] = st.number_input("Age", min_value=0, max_value=120, value=18)
        st.session_state['insured_sex'] = st.selectbox('Sex', ['MALE', 'FEMALE'])
        st.session_state['state'] = st.selectbox('State', ['NY', 'VA', 'WV', 'NC', 'SC', 'PA', 'OH'])
        
        # Accident and Injury Details Section
        st.header("Accident and Injury Details")
        st.session_state['accident_type'] = st.selectbox('Accident Type', ['Multi-vehicle Collision', 'Vehicle Theft', 'Single Vehicle Collision', 'Parked Car']) 
        st.session_state['auto_make'] = st.selectbox('Auto Make', ['Dodge', 'Accura', 'Nissan', 'Audi', 'Toyota', 'Saab', 'Ford', 'Suburu', 'BMW', 'Jeep', 'Mercedes', 'Honda', 'Volkswagen', 'Chevrolet'])
        st.session_state['collision_type'] = st.selectbox('Collision Type', ['Rear Collision', 'Front Collision', 'Side Collision'])
        st.session_state['incident_severity'] = st.selectbox('Incident Severity', ['Trivial Damage', 'Minor Damage', 'Major Damage', 'Total Loss'])
        st.session_state['property_damage'] = st.selectbox('Property Damage', ['YES', 'NO'])
        st.session_state['bodily_injuries'] = st.number_input('Bodily Injuries', min_value=0, max_value=2, step=1)
        st.session_state['authorities_contacted'] = st.selectbox('Authorities Contacted', ['Police', 'Other', 'Fire', 'Ambulance'])
        st.session_state['police_report_available'] = st.selectbox('Police Report Available', ['YES', 'NO'])
        
        # Insurance Information
        st.header("Insurance Details")
        st.session_state['policy_state'] = st.selectbox('Policy State', ['OH', 'IL', 'IN'])  
        st.session_state['policy_csl'] = st.selectbox('Policy CSL', ['100/300', '500/1000', '250/500'])  
        st.session_state['umbrella_limit'] = st.selectbox('Umbrella Limit', [5000000, 6000000, 0, 4000000, 3000000, 8000000, 7000000, 9000000, -1000000, 2000000, 10000000])  
        
    st.button("Predict", key="predict", type='primary', on_click=submit)

# Function to display the prediction
def show_prediction():
    """
    Display the prediction result after form submission.
    """
    
    # Title
    st.title(":green[Prediction Result]")
    
    # Get predicted claim amount from session state
    predicted_amount = st.session_state.get("predicted_amount", 0)
    highest_amount = st.session_state.get("highest_amount", 0)
    lowest_amount = st.session_state.get("lowest_amount", 0)

    # Show prediction result
    col1, col2 = st.columns(2)
    with col1:
        st.subheader(":blue[Estimated Claim Amount]")
        st.metric(label="Claim Amount", value=f"${predicted_amount:,.2f}")
    
    with col2:
        st.subheader(":blue[Acceptable Range]")
        st.metric(label="Claim Range", value=f"${lowest_amount:,.2f} - ${highest_amount:,.2f}")

    st.divider()
    back_button = st.button("Back", on_click=back)

# Main section for the prediction tab
def display_prediction():
    """Defines the display behaviour of the prediction tab.
    """
    if "form_submitted" not in st.session_state:
        st.session_state["form_submitted"] = False

    if st.session_state["form_submitted"]:
        show_prediction()
    else:
        show_form()

if __name__ == '__main__':
    display_prediction()
