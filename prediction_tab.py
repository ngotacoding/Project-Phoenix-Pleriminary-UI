import streamlit as st

def submit():
    """Define what happens when you click Submit.
    """
    # Placeholder for the model prediction
    predicted_amount = 10000  # Dummy prediction for illustration
    mean_absolute_error = 3000 # Placeholder mean absolute error
    lowest_amount = predicted_amount - mean_absolute_error
    highest_amount = predicted_amount + mean_absolute_error
    st.session_state["predicted_amount"] = predicted_amount
    st.session_state["highest_amount"] = highest_amount
    st.session_state["lowest_amount"] = lowest_amount
    st.session_state["form_submitted"] = True
    
def back():
    """Define what happens when you click Back.
    """
    # Define back button behavior to return to the form
    st.session_state["form_submitted"] = False

# Function to display the main form
def show_form():
    """
    Display the main form for inputting details to estimate the settlement value of a personal injury case.
    """

    # Title and Description
    st.title("Predict your Claim amount")

    # Form Section
    with st.form("claim_form"):
        
        # Injury Details Section
        st.header("Injury Details")
        # Dropdown for selecting the type of injury
        injury_type = st.selectbox("Injury Type", ["Type 1", "Type 2", "Type 3"])
        # Slider for selecting the severity of the injury
        injury_severity = st.slider("Injury Severity", 0, 10)

        # Medical Treatment Information Section
        st.header("Medical Treatment Information")
        # Multi-select boxes for selecting medical procedures and medication prescribed
        medical_procedures = st.multiselect("Medical Procedures", ["Procedure 1", "Procedure 2", "Procedure 3"])
        medications_prescribed = st.multiselect("Medications Prescribed", ["Medication 1", "Medication 2", "Medication 3"])
        # Number input for the duration of hospitalization in days
        hospitalization_duration = st.number_input("Hospitalization Duration (days)", min_value=0)
        # Multi-select box for selecting rehabilitation services
        rehabilitation_services = st.multiselect("Rehabilitation Services", ["Service 1", "Service 2", "Service 3"])

        # Insurance Information Section
        st.header("Insurance Information")
        # Number inputs for policy coverage limit, policy deductible, and policy co-pay
        policy_coverage_limit = st.number_input("Policy Coverage Limit", min_value=0)
        policy_deductible = st.number_input("Policy Deductible", min_value=0)
        policy_co_pay = st.number_input("Policy Co-pay", min_value=0)

        # Financial Losses Section
        st.header("Financial Losses")
        # Number input for the total financial losses
        financial_losses = st.number_input("Total Financial Losses", min_value=0)

        # Demographic and Geographic Information
        st.header("Demographic and Geographic Information")
        # Number input for claimant's age
        claimant_age = st.number_input("Claimant Age", min_value=0)
        # Dropdowns for selecting claimant's gender and claimant state
        claimant_gender = st.selectbox("Claimant Gender", ["Male", "Female", "Other"])
        claimant_state = st.selectbox("Claimant State", ["State 1", "State 2", "State 3", "State 4", "State 5"])

        # Submit button for the form
        submit_button = st.form_submit_button("Submit", on_click=submit)

# Function to display the prediction and chatbot UI
def show_prediction():
    """
    Display the prediction result after form submission.
    """
    
    # Title
    st.title("Prediction Result")
    
    # Get predicted claim amount from session state
    predicted_amount = st.session_state.get("predicted_amount", 0)
    highest_amount = st.session_state.get("highest_amount", 0)
    lowest_amount = st.session_state.get("lowest_amount", 0)

    # Show prediction result
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Estimated Claim Amount")
        st.metric(label="Claim Amount", value=f"${predicted_amount:,.2f}")
    
    with col2:
        st.subheader("Acceptable Range")
        st.metric(label="Claim Range", value=f"${lowest_amount:,.2f} - ${highest_amount:,.2f}")

    st.divider()
    # Sample chatbot UI
    st.subheader("Chat with the Result")
    user_input = st.text_input("Ask a question:")
    if user_input:
        response = f"Model response to: '{user_input}'"
        st.write(response)
    
    # Define back button
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

