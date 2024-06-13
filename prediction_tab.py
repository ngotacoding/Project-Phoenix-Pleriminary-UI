import streamlit as st
import numpy as np
from model_utils import predict_claim

specialty = ['Family Practice', 'OBGYN', 'Cardiology', 'Pediatrics',
             'Internal Medicine', 'Anesthesiology', 'Emergency Medicine',
             'Ophthamology', 'Urological Surgery', 'Orthopedic Surgery',
             'Neurology/Neurosurgery', 'Occupational Medicine', 'Resident',
             'Thoracic Surgery', 'General Surgery', 'Radiology', 'Pathology',
             'Physical Medicine', 'Plastic Surgeon', 'Dermatology']

def submit():
    """Define what happens when you click Predict."""
    # Placeholder for the model prediction
    predicted_amount = predict_claim()  # Dummy prediction for illustration
    mean_absolute_error = 3000  # Placeholder mean absolute error
    lowest_amount = predicted_amount - mean_absolute_error
    highest_amount = predicted_amount + mean_absolute_error
    st.session_state["predicted_amount"] = predicted_amount
    st.session_state["highest_amount"] = highest_amount
    st.session_state["lowest_amount"] = lowest_amount
    st.session_state["form_submitted"] = True

def reset():
    """Reset the form."""
    st.session_state["current_step"] = "Step 1"
    st.session_state["form_submitted"] = False
    for key in ['severity', 'specialty', 'insurance', 'age', 'gender', 'marital_status', 'private_attorney']:
        st.session_state[key] = None

def next_step():
    """Move to the next step."""
    current_step = st.session_state["current_step"]
    if current_step == "Step 1":
        st.session_state["current_step"] = "Step 2"
        st.rerun()
    elif current_step == "Step 2":
        st.session_state["current_step"] = "Step 3"
        st.rerun()

def previous_step():
    """Move to the previous step."""
    current_step = st.session_state["current_step"]
    if current_step == "Step 2":
        st.session_state["current_step"] = "Step 1"
        st.rerun()
    elif current_step == "Step 3":
        st.session_state["current_step"] = "Step 2"
        st.rerun()

def step_1():
    st.header(":green[Step 1/3 ]")
    with st.container(border=1):
        st.header(":blue[Injury & Medical Details]")
        st.session_state['severity'] = st.slider("Severity of Injury", 1, 9, help="Rating of damage from 1 (emotional trauma) to 9 (death)")
        st.session_state['specialty'] = st.selectbox("Specialty of Physician Seen", specialty)
        st.session_state['insurance'] = st.selectbox("Medical Insurance Type", ["Private", "Unknown", "Medicare/Medicaid", "No Insurance", "Workers Compensation"])
    if st.button("Next", key="next1", type='primary'):
        next_step()
        st.rerun()

def step_2():
    st.header(":green[Step 2/3 ]")
    with st.container(border=1):
        st.header(":blue[Personal Details]")
        st.session_state['age'] = st.number_input("Age", min_value=0, max_value=120)
        st.session_state['gender'] = st.selectbox("Gender", ["Male", "Female"])
        st.session_state['marital_status'] = st.selectbox("Marital Status", [0, 1, 2, 3, 4], format_func=lambda x: ["Divorced", "Single", "Married", "Widowed", "Unknown"][x])
        st.session_state['private_attorney'] = st.selectbox("Nature of Attorney", [0, 1], format_func=lambda x: "Private" if x == 1 else "Not Private")
        
    cols = st.columns(9)
    with cols[0]:
        if st.button("Previous", key="prev2"):
            previous_step()
    with cols[1]:
        if st.button("Next", key="next2", type='primary'):
            next_step()
            st.rerun()
    
def step_3():
    st.header(":green[Step 3/3 ]")
    with st.container(border=1):
        st.header(":blue[Confirm Details]")
        #st.subheader(":blue[Injury & Medical Details]")
        st.write(f"**Severity of Injury:** {st.session_state.get('severity')}")
        st.write(f"**Specialty of Physician Seen:** {st.session_state.get('specialty')}")
        st.write(f"**Medical Insurance Type:** {st.session_state.get('insurance')}")
        
        #st.subheader("Personal Details")
        st.write(f"**Age:** {st.session_state.get('age')}")
        st.write(f"**Gender:** {st.session_state.get('gender')}")
        st.write(f"**Marital Status:** {['Divorced', 'Single', 'Married', 'Widowed', 'Unknown'][st.session_state.get('marital_status')]}")
        st.write(f"**Nature of Attorney:** {'Private' if st.session_state.get('private_attorney') == 1 else 'Not Private'}")
        
    cols = st.columns(9)
    with cols[0]:
        if st.button("Previous", key="prev3"):
            previous_step()
            st.rerun()
    with cols[1]:
        if st.button("Predict", key="predict", type='primary'):
            submit()
            show_prediction()
            st.rerun()

def show_prediction():
    """Display the prediction result after form submission."""
    st.title(":green[Prediction Result]")
    predicted_amount = st.session_state.get("predicted_amount", 0)
    highest_amount = st.session_state.get("highest_amount", 0)
    lowest_amount = st.session_state.get("lowest_amount", 0)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader(":blue[Estimated Claim Amount]")
        st.metric(label="Claim Amount", value=f"${predicted_amount:,.2f}")
    
    with col2:
        st.subheader(":blue[Acceptable Range]")
        st.metric(label="Claim Range", value=f"${lowest_amount:,.2f} - ${highest_amount:,.2f}")

    st.divider()
    st.subheader(":green[Chat with the Result]")
    user_input = st.text_input("Ask a question:")
    if user_input:
        response = f"Model response to: '{user_input}'"
        st.write(response)
    
    st.button("New Prediction", type='primary', on_click=reset)

def display_prediction():
    if "current_step" not in st.session_state:
        st.session_state["current_step"] = "Step 1"
    if "form_submitted" not in st.session_state:
        st.session_state["form_submitted"] = False

    if st.session_state["form_submitted"]:
        show_prediction()
    else:
        current_step = st.session_state["current_step"]
        if current_step == "Step 1":
            step_1()
        elif current_step == "Step 2":
            step_2()
        elif current_step == "Step 3":
            step_3()

if __name__ == "__main__":
    display_prediction()
