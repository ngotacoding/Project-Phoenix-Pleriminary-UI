import streamlit as st

# Function to define the content of the main page
def display_home():
    """Define the outlook of the home page.
    """
    
    with st.container():
        
        st.header(':blue[Project Background]')
        st.write("""
            In the United States of America (USA), victims of personal injuries depend on personal injury lawyers to secure settlements from insurance companies. These victims are often disadvantaged as insurers use advanced algorithms like Colossus to propose minimal settlement amounts based on extensive data analysis. On the other hand, personal injury lawyers typically rely on their own experiences and limited data, lacking the sophisticated analytical tools available to insurers. This imbalance can lead to lower settlements for the victims (plaintiffs) as lawyers struggle to match the data-driven insights that insurance companies use to their advantage. The disparity highlights the need for a reliable and transparent method to predict settlement values, ensuring that the victims receive fair compensation for their injuries.
            """)

        st.header(':blue[Project Objectives]')
        st.write("""
            The primary objective of this project is to transform the estimation of personal injury case values in the United States by developing a predictive data model. This model aims to provide a reliable tool that supplements the expertise of legal professionals, offering a more objective basis for case valuation. Leveraging the model will empower legal professionals with better tools for negotiation and help plaintiffs secure fair and just compensation for their injuries.
            """)

        st.header(':blue[How Personal Injury Cases Work]')
        st.write("""
            Liability in personal injury cases determines who is at fault for causing harm. Plaintiffs must prove that the negligence or intentional actions of the defendant caused their injuries. Compensation covers two types of damages: economic (quantifiable financial losses such as medical bills and lost wages) and non-economic (subjective impacts like future pain and suffering). 
            
            Future pain refers to the ongoing physical and emotional suffering that an individual will endure in the future due to an injury or condition that has already occurred. Assessing future pain involves evaluating medical opinions on long-term injury impacts, including chronic pain and lifestyle restrictions. Lawyers use this information to predict how pain will affect a client’s daily life and future well-being. Estimating monetary compensation for future pain involves expert medical testimony and personal evidence, like pain journals, which help when negotiating with insurance companies. 
            
            Hence, successfully claiming compensation requires demonstrating both liability and the extent of damages, economic or non-economic, often with legal assistance.
            
            
            """)

        st.subheader(':blue[Important Questions in Evaluating a Personal Injury Case]')
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

        st.header(':blue[Facts claimants must know about making Insurance claims]')
        st.write("""
           When making an insurance claim, one of the most critical factors that claimants must be aware of is the statute of limitations. The statute of limitations sets the maximum period within which a claimant can initiate legal proceedings from the date of the incident or injury. Failing to file a claim within this period can result in losing the right to seek compensation, regardless of the validity.
        """)
        st.subheader(':blue[Understanding the Statute of Limitations]')
        st.write("""
            1. The statute of limitations can vary significantly depending on the jurisdiction and the type of claim. Therefore, claimants should familiarize themselves with the limits to their case in their legal area to ensure they file their claim on time.
            
            2. Once the statute of limitations has expired, the claimant typically loses the right to file a lawsuit. Hence, even if the evidence supporting the claim is strong, the court will not consider the case, barring the claimant from seeking compensation.
            
            3. There may be exceptions or extensions to the statute of limitations in certain situations, such as instances where the claimant was a minor at the time of the injury or if the injury was not immediately discoverable.

            Given the critical nature of the statute of limitations, claimants should take prompt action after an injury or incident, including gathering evidence, seeking legal advice, and filing the necessary paperwork as soon as possible to avoid any issues with timing.
            It is recommended to consult with legal professionals in personal injury and insurance claims for guidance and ensure claimants take the necessary steps within the required time frame.

        """)
    
