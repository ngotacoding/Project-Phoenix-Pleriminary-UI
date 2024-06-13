import streamlit as st

def submit():
    """Define what happens when you click Predict."""
    # Placeholder for the model prediction
    predicted_amount = 10000  # Dummy prediction for illustration
    mean_absolute_error = 3000 # Placeholder mean absolute error
    lowest_amount = predicted_amount - mean_absolute_error
    highest_amount = predicted_amount + mean_absolute_error
    st.session_state["predicted_amount"] = predicted_amount
    st.session_state["highest_amount"] = highest_amount
    st.session_state["lowest_amount"] = lowest_amount
    st.session_state["form_submitted"] = True
    
def next_step(step):
    """Proceed to the next step of the form."""
    st.session_state["current_step"] = step

def back():
    """Define what happens when you click Back."""
    # Define back button behavior to return to the form
    st.session_state["form_submitted"] = False
    st.session_state["current_step"] = 1

# Function to display the main form in steps
def show_form():
    """Display the main form for inputting details to estimate the settlement value of a personal injury case."""
    if "current_step" not in st.session_state:
        st.session_state["current_step"] = 1

    step = st.session_state["current_step"]
    
    st.title("Predict your Claim amount")
    st.subheader(f"Step {step} of 4")

    if step == 1:
        with st.form("injury_details_form"):
            st.header("Injury Details")
            injury_type = st.selectbox("Injury Type", ["Type 1", "Type 2", "Type 3"])
            injury_severity = st.slider("Injury Severity", 0, 10)
            next_button = st.form_submit_button("Next", on_click=lambda: next_step(2))

    elif step == 2:
        with st.form("medical_treatment_form"):
            st.header("Medical Treatment Information")
            medical_procedures = st.multiselect("Medical Procedures", ["Procedure 1", "Procedure 2", "Procedure 3"])
            medications_prescribed = st.multiselect("Medications Prescribed", ["Medication 1", "Medication 2", "Medication 3"])
            hospitalization_duration = st.number_input("Hospitalization Duration (days)", min_value=0)
            next_button = st.form_submit_button("Next", on_click=lambda: next_step(3))

    elif step == 3:
        with st.form("legal_claim_form"):
            st.header("Legal and Claim Information")
            law_firm = st.text_input("Law Firm Name")
            lawyer_experience = st.number_input("Lawyer Experience (years)", min_value=0)
            settlement_offer = st.number_input("Initial Settlement Offer ($)", min_value=0)
            next_button = st.form_submit_button("Next", on_click=lambda: next_step(4))

    elif step == 4:
        with st.form("personal_geographic_form"):
            st.header("Personal and Geographic Information")
            claimant_age = st.number_input("Claimant Age", min_value=0)
            claimant_gender = st.selectbox("Claimant Gender", ["Male", "Female", "Other"])
            claimant_state = st.selectbox("Claimant State", ["State 1", "State 2", "State 3", "State 4", "State 5"])
            submit_button = st.form_submit_button("Predict", on_click=submit)

# Function to display the prediction and chatbot UI
def show_prediction():
    """Display the prediction result after form submission."""
    
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
    """Defines the display behavior of the prediction tab."""
    if "form_submitted" not in st.session_state:
        st.session_state["form_submitted"] = False

    if st.session_state["form_submitted"]:
        show_prediction()
    else:
        show_form()