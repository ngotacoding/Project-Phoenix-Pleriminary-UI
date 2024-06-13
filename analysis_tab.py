import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import plotly.figure_factory as ff
from plotly.subplots import make_subplots

# Constants
DATA_PATH1 = "data/Kaggle_medical_practice_20.csv"
DATA_PATH2 = "data/sample_data_citystate_corrected.csv"
DATA_PATH3 = "data/Kaggle_medical_practice_20.csv - 20_preprocessed.csv"
DATA_PATH4 = "data/sample_data_formatted.csv"
DATA_PATH5 = "data/Insurance_claims_mendeleydata_6.csv"

AGE_GROUP_ORDER = ["0-10 years", "11-18 years", "19-50 years", "51+ years"]


def load_data(path):
    try:
        df = pd.read_csv(path)
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None


def age_group(age):
    if age >= 0 and age <= 10:
        return "0-10 years"
    elif age > 10 and age <= 18:
        return "11-18 years"
    elif age > 18 and age <= 50:
        return "19-50 years"
    elif age > 50:
        return "51+ years"
    else:
        return "Unknown"


def preprocess_data(df):
    df['total_bills'] = pd.to_numeric(df['total_bills'], errors='coerce')
    df['total_coverage'] = pd.to_numeric(df['total_coverage'], errors='coerce')
    df['total_claim_amount'] = df[[
        'total_bills', 'total_coverage']].min(axis=1)
    df['airbag_deployed'] = df['airbag_deployed'].fillna('No')
    df['called_911'] = df['called_911'].fillna('Unknown')
    df['called_911'] = df['called_911'].apply(
        lambda x: 'Yes' if x == 'Yes' else ('No' if x == 'No' else 'Unknown'))
    df = df.dropna(subset=['total_claim_amount'])
    Q1 = df['total_claim_amount'].quantile(0.25)
    Q3 = df['total_claim_amount'].quantile(0.75)
    IQR = Q3 - Q1
    df = df[~((df['total_claim_amount'] < (Q1 - 1.5 * IQR)) |
              (df['total_claim_amount'] > (Q3 + 1.5 * IQR)))]
    return df


def preprocess_df4(df):
    df['claim_amount'] = np.where(
        df['total_bills'] <= df['total_coverage'], df['total_bills'], df['total_coverage'])
    df['total_bills'] = np.where(
        df['total_bills'].isnull(), df['claim_amount'], df['total_bills'])
    df = df.dropna(subset=['claim_amount'])
    df = df[~(df['claim_amount'] == 0)]
    df = df[~(df['age'] < 0)]
    df["state"] = df["state"].str.upper()
    return df


def preprocess_sample_dataset(df):
    """
    These are the preprocessing steps for sample_data_formatted.csv to be
    transformed to the format that is used for Statewise Comparison
    of Car Accident Claims

    Args
    ---------------------
    df_samp: pd.DataFrame | sample dataset to be transformed

    Returns
    ---------------------
    df_samp: pd.DataFrame | dataframe after simple preprocessing

    Errors Raised
    --------------------
    KeyError | If a dataframe is used without the same column names, a KeyError will be raised

    """
    # Get the claim amount as it is final target variable
    df['claim_amount'] = np.where(
        df['total_bills'] <= df['total_coverage'], df['total_bills'], df['total_coverage'])
    # total_bills
    df['total_bills'] = np.where(
        df['total_bills'].isnull(), df['claim_amount'], df['total_bills'])
    # drop null value rows from claim amount
    df = df.dropna(subset=['claim_amount'])
    # drop rows with claim amount '0'
    df = df[~(df['claim_amount'] == 0)]
    # drop rows with -ve age
    df = df[~(df['age'] < 0)]

    # String Format for State Abbreviations
    df["state"] = df["state"].str.upper()

    # Selecting States with Fewer Than 45 rows of observations
    small_obs = df["state"].value_counts(
    )[df["state"].value_counts() < 45].index

    # Binning small-observation states
    df.loc[df["state"].isin(small_obs), "state"] = "Other"

    # Script for Binning Type of Injury Column

    df.rename(columns={"injury_type": "Type of Injury"}, inplace=True)

    df = df.dropna(subset="Type of Injury")
    # First consolidation - the backslash is not separated from 'Other Injury' with a space
    df.loc[df["Type of Injury"] == "Other Injury/ Pain",
           "Type of Injury"] = "Other Injury / Pain"

    # General Traumatic Brain Injury consolidation
    df.loc[df["Type of Injury"] == "Other Injury / Pain; Traumatic Brain Injury.", "Type of Injury"] \
        = "Other Injury / Pain; Traumatic Brain Injury"

    # Removing LOC ("Loss of Consciousness") from the category's granularity
    df.loc[df["Type of Injury"] == "Other Injury / Pain; Traumatic Brain Injury - w/LOC", "Type of Injury"] \
        = "Other Injury / Pain; Traumatic Brain Injury"

    # Handling Excessive spaces between words
    df.loc[df["Type of Injury"] == "Other Injury / Pain; Traumatic  Brain  Injury", "Type of Injury"] \
        = "Other Injury / Pain; Traumatic Brain Injury"

    # Another Different Entry with Redundant Information
    df.loc[df["Type of Injury"] == "Other Injury / Pain; Traumatic Brain Injury - w/LOC; Traumatic  Brain  Injury", "Type of Injury"]\
        = "Other Injury / Pain; Traumatic Brain Injury"

    # Same
    df.loc[df["Type of Injury"] == "Other Injury / Pain; Traumatic Brain Injury - w/LOC; Traumatic Brain Injury.", "Type of Injury"]\
        = "Other Injury / Pain; Traumatic Brain Injury"

    # Same
    df.loc[df["Type of Injury"] == "Other Injury / Pain; Traumatic  Brain  Injury; Traumatic  Brain  Injury", "Type of Injury"] \
        = "Other Injury / Pain; Traumatic Brain Injury"

    # ---- Handling Fatal Cases ----

    # 124 Distinct Injuries resulting in Death

    # Consolidate entries containing "Death" and "Traumatic Brain Injury"                                   #### CHECK
    df.loc[df["Type of Injury"].str.contains(
        "(?=.*Death)(?=.*Traumatic Brain Injury)"), "Type of Injury"] = "Death"

    # -------- Broken Bones -------
    df.loc[df["Type of Injury"].str.contains("Traumatic Brain Injury.*Broken Bones"), "Type of Injury"]\
        = "Other Injury / Pain; Traumatic Brain Injury; Broken Bones"

    df.loc[df["Type of Injury"].str.contains("Other Injury.*Broken Bones"), "Type of Injury"]\
        = "Other Injury / Pain; Traumatic Brain Injury; Broken Bones"

    # ------- Ruptured Discs -> regular expressions for the "Other Pain" and "Traumatic Brain Injury" as superseding categories
    df.loc[df["Type of Injury"].str.contains("(?=Other Injury / Pain)(?=.*Herniated/Bulging/Ruptured Disc)(?=.*Traumatic Brain Injury)"),
           "Type of Injury"] = "Other Injury / Pain; Traumatic Brain Injury; Herniated/Bulging/Ruptured Disc"

    # ------ Ruptured Discs -> regular expressions for the "Other Pain" and "Traumatic Brain Injury" as superseding categories
    df.loc[df["Type of Injury"].str.contains("(?=Other Injury / Pain)(?=.*Herniated/Bulging/Ruptured Disc)"),
           "Type of Injury"] = "Other Injury / Pain; Herniated/Bulging/Ruptured Disc"

    # Capturing the last remaining values
    df.loc[df["Type of Injury"].str.contains("Herniated/Bulging/Ruptured Disc"), "Type of Injury"]\
        = "Other Injury / Pain; Herniated/Bulging/Ruptured Disc"

    ##### AT THIS POINT: Remaining un-consolidated values all represent less than 1% of total values #########

    # -------- Tendon/Ligament -> Consolidating into the larger bin
    df.loc[df["Type of Injury"] == "Tendon/Ligament Tear/Rupture",
           "Type of Injury"] = "Other Injury / Pain; Tendon/Ligament Tear/Rupture"

    # ------------PTSD etc using the larger bin in existence
    df.loc[df["Type of Injury"] == "Other Injury / Pain; Traumatic Brain Injury - w/LOC; Anxiety,PTSD,Depression,Stress", "Type of Injury"]\
        = "Other Injury / Pain; Traumatic  Brain  Injury; Anxiety,PTSD,Depression,Stress"

    df.loc[df["Type of Injury"] == "Other Injury / Pain; Traumatic Brain Injury.; Anxiety,PTSD,Depression,Stress", "Type of Injury"]\
        = "Other Injury / Pain; Traumatic  Brain  Injury; Anxiety,PTSD,Depression,Stress"

    # After these bins have been created, the top 20 Values account for 96.2% of all rows in the data.

    # Binning all values not found in the top 20
    exclusion_list = df["Type of Injury"].value_counts(normalize=True)[
        :20].index

    # '~' accesses the complement - "Not In" the exclusion list
    df.loc[~df["Type of Injury"].isin(
        exclusion_list), "Type of Injury"] = "Other Injury"

    # TBI
    df.loc[df["Type of Injury"].str.contains(
        "Traumatic Brain Injury"), "Type of Injury"] = "Traumatic Brain Injury"

    # Broken Bones
    df.loc[df["Type of Injury"].str.contains(
        "Broken Bones"), "Type of Injury"] = "Broken Bones"

    # 21 Bins of Values left, and the final bin contains roughly 3.8 % of all entries
    df["Type of Injury"] = df["Type of Injury"].str.replace(
        "Other Injury / ", "")

    df["Type of Injury"] = df["Type of Injury"].str.replace("Pain; ", "")

    return df


def preprocess_insurance_data(data):
    """
    Preprocessing steps for Insurance_claims_mendeleydata_6.csv
    to be transformed in a manner that allows for state-wise visualization

    Args
    ---------------
    data: pd.DataFrame | pandas dataframe to be used for state-wise plots

    Returns
    ---------------
    data : pd.DataFrame | preprocessed with minimal steps

    Errors Raised
    ---------------
    KeyError | if data with different column names is used, then this function will raise an error

    """

    data = data.rename(columns={"total_claim_amount": "claim_amount",
                                "insured_sex": "gender"})
    data["gender"] = data["gender"].str.title()

    return data


def plotly_states(data):
    """
    Function to generate a plotly figure of barplots of mean and median state claim values for car accidents
    compatible with sample_data_formatted.csv and Insurance_claims_mendeleydata_6.csv

    Args
    -----------
    data: pd.DataFrame | data with columns: ["state", "total_claim_amount"]

    Returns
    -----------
    plotly figure | barplot with hover values of State, Mean/Median Value

    Errors
    -----------
    KeyError if data do not contain the correct columns
    """

    # Filtering out miscellaneous states
    data = data[data["state"] != "Other"]

    # Grouping data by state and calculating median and mean
    grouped = data.groupby("state")["claim_amount"].agg(
        ["median", "mean"]).sort_values(by="median", ascending=False)

    # Resetting index to make 'state' a column for Plotly
    grouped = grouped.reset_index()

    # Creating Plotly figure
    fig = px.bar(grouped, x='state', y=['median', 'mean'],
                 labels={'value': 'Claim Amount in USD', 'state': 'States'},
                 title='Mean and Median Claims by State Sorted by Median Claim',
                 barmode='group',
                 template="plotly")

    # Legend
    fig.update_layout(legend_title='')

    # Customizing hover info
    fig.update_traces(hovertemplate='State: %{x}<br>Value: %{y:.2f}')

    fig.for_each_trace(lambda t: t.update(name=t.name.capitalize()))

    # fig.update_layout(width=1000)

    # Returning the Plotly figure
    return fig


def plotly_box_states(data):
    """
    Function to generate a plotly figure of boxplots of car accidents claim distributions by state
    compatible with sample_data_formatted.csv and Insurance_claims_mendeleydata_6.csv

    Args
    -----------
    data: pd.DataFrame | data with columns: ["state", "total_claim_amount"]

    Returns
    -----------
    plotly figure | boxplot with hover values of State, [min, lower fence, 25 percentile, median, 75 percentile, upper fence, max]

    Errors
    -----------
    KeyError if data do not contain the correct columns
    """

    # Filter Data for States == Other
    data = data[data["state"] != "Other"]

    # Creating a list of states ordered by their median percentile value
    # to provide a left-to-right visual structure
    upper_q = list(data.groupby("state")[
                   "claim_amount"].median().sort_values(ascending=False).index)

    # Create traces for each state -> this was the only way I could get the whisker/plot scale correct
    traces = []
    for state in upper_q:
        state_data = data[data['state'] == state]
        trace = go.Box(
            y=state_data['claim_amount'],
            name=state,
            boxpoints='all',  # Show all points to maintain correct whisker length
            jitter=0.3,
            pointpos=-1.8,
            marker=dict(opacity=0),  # Make point markers invisible
            line=dict(width=2),
            boxmean=False  # Do not show mean
        )
        traces.append(trace)

    # Create the figure
    fig = go.Figure(data=traces)

    # Update layout
    fig.update_layout(
        title="Distribution of Car Accident Claims in Different States",
        yaxis=dict(
            title="Total Claim in USD"
        ),
        xaxis=dict(
            title="State"
        ),
        showlegend=False,
        template="plotly"
    )

    # Calculate IQR for each state to determine y-axis range
    iqr_ranges = data.groupby('state')['claim_amount'].apply(
        lambda x: (x.quantile(0.25), x.quantile(0.75)))
    iqr_min, iqr_max = iqr_ranges.apply(
        lambda x: x[0]).min(), iqr_ranges.apply(lambda x: x[1]).max()
    iqr = iqr_max - iqr_min

    # Update y-axis range to be slightly larger than the IQR range
    fig.update_yaxes(range=[-1000, iqr_max + 1.5 * iqr])
    # fig.update_layout(width=1000)

    return fig


def plotly_gender(data):
    """
    Function to generate a plotly figure of KDE distributions for Genders
    compatible with Kaggle_medical_practice_20.csv and Insurance_claims_mendeleydata_6.csv

    Args
    -----------
    data: pd.DataFrame | data with columns: ["gender", "total_claim_amount"]

    Returns
    -----------
    plotly figure | 3 kde plots with hover values of x coordinates (claim value)

    Errors
    -----------
    KeyError if data do not contain the correct columns
    """
    if "claim_amount" in data.columns:
        data = data.rename(columns={"claim_amount": "total_claim_amount"})

    male_data = data.query("gender == 'Male'")['total_claim_amount']
    female_data = data.query("gender == 'Female'")['total_claim_amount']

    male_median_x = male_data.median()
    female_median_x = female_data.median()

    # KDEs
    male_kde = ff.create_distplot([male_data], group_labels=[
                                  'Male'], show_hist=False, show_rug=False)
    female_kde = ff.create_distplot([female_data], group_labels=[
                                    'Female'], show_hist=False, show_rug=False)

    # Create subplots
    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=(f'Male Claim Amounts - Median Claim ${male_median_x:,.2f}',
                                        f'Female Claim Amounts - Median Claim ${female_median_x:,.2f}', 'Male vs Women Overlaid'),
                        specs=[[{"type": "scatter"}, {"type": "scatter"}], [
                            {"colspan": 2}, None]],
                        row_heights=[0.5, 0.5],
                        column_widths=[0.5, 0.5])

    # Male KDE Plot
    for trace in male_kde['data']:
        trace["hoverinfo"] = 'x'
        trace["showlegend"] = False
        fig.add_trace(trace, row=1, col=1)

    male_median_y = male_kde['data'][0]['y'].max()

    # Adding a vline
    fig.add_shape(type="line",
                  x0=male_median_x, y0=0,
                  x1=male_median_x, y1=male_median_y,
                  line={"color": "darkred", "dash": "dash"},
                  row=1, col=1)

    # Female KDE Plot
    for trace in female_kde['data']:
        trace["hoverinfo"] = 'x'
        trace["showlegend"] = False
        fig.add_trace(trace, row=1, col=2)

    female_median_y = female_kde['data'][0]['y'].max()

    # Adding a vline
    fig.add_shape(type="line",
                  x0=female_median_x, y0=0,
                  x1=female_median_x, y1=female_median_y,
                  line=dict(color="darkred", dash="dash"),
                  row=1, col=2)

    # Overlaid KDE Plot
    fig.add_trace(go.Scatter(x=male_kde['data'][0]['x'], y=male_kde['data'][0]['y'],
                             mode='lines', name='Male', fill='tozeroy', line=dict(color='blue'), opacity=0.1,
                             hoverinfo='x'),
                  row=2, col=1)

    fig.add_trace(go.Scatter(x=female_kde['data'][0]['x'], y=female_kde['data'][0]['y'],
                             mode='lines', name='Female', fill='tozeroy', line=dict(color='lightcoral'), opacity=0.1,
                             hoverinfo='x'),
                  row=2, col=1)

    # Update layout
    fig.update_layout(height=800, width=1200,
                      title_text="Distribution of All Claim Amounts for Men vs Women")
    fig.update_xaxes(title_text="Total Claim in USD", row=1, col=1)
    fig.update_xaxes(title_text="Total Claim in USD", row=1, col=2)
    fig.update_xaxes(title_text="Total Claim in USD", row=2, col=1)

    fig.update_layout(showlegend=True, legend=dict(x=0.875, y=0.275))
    fig.update_yaxes(showticklabels=False)
    # fig.update_layout(width=1000)

    # Show plot
    return fig


def plotly_box_gender(data):
    """
    Function to generate a plotly figure of Boxplot distributions without outliers for Genders Across Insurance Types
    compatible with Kaggle_medical_practice_20.csv

    Args
    -----------
    data: pd.DataFrame | data with columns: ["gender", "total_claim_amount", "insurance"]

    Returns
    -----------
    plotly figure | boxplot with hover values of State, then any of:
    [max, upper fence, 75th percentile, median, 25th percentile, lower fence, min]

    Errors
    -----------
    KeyError if data do not contain the correct columns
    """

    # Creating a list of states ordered by their median percentile value
    # to provide a left-to-right visual structure
    fig = px.box(data, x="insurance", y="total_claim_amount", color="gender")

    # Update layout
    fig.update_layout(
        title="Distribution of Claims by Insurance Type for Men and Women",
        yaxis=dict(
            title="Total Claim in USD"
        ),
        xaxis=dict(
            title="Insurance Type"
        ),
        showlegend=True,
        template="plotly"
    )

    fig.update_layout(legend_title='Gender')

    return fig


def plotly_injury_bar(data):
    grouped = data.groupby("Type of Injury")["claim_amount"].agg(["mean", "median"]).round(2).reset_index(
    ).sort_values(by="median", ascending=True).rename(columns={"mean": "Mean", "median": "Median"})
    fig = px.bar(grouped, y='Type of Injury', x=['Median', 'Mean'],
                 labels={'value': "Claim Value",
                         'Type of Injury': 'Injury', "variable": "Statistic"},
                 title='Mean and Median Claims by Injury', barmode='group', color_continuous_scale="Viridis")
    fig.update_layout(showlegend=False, width=1200, height=675)

    return fig


def plotly_injury_hist(data):
    fig_h = px.histogram(data, x="claim_amount", nbins=25, labels={
                         "claim_amount": "Claim", "value": "Count"})
    fig_h.update_traces(hovertemplate='Claim: %{x}<br>Count: %{y}')
    injury = data["Type of Injury"].unique()[0]
    fig_h.update_layout(yaxis={
                        "title": "Count"}, title=f"Histogram of Claim Distribution for {injury.title()}")
    return fig_h


def plotly_boxplot_injury(data):
    fig_b = px.box(data, x="claim_amount")
    injury = data["Type of Injury"].unique()[0]
    fig_b.update_layout(
        title=f"Boxplot of Claim Distribution for {injury.title()}")

    return fig_b


def display_analysis():
    st.header("Introduction")

    st.markdown(
        """
    </style>
    <div class="markdown-text-container">
        Welcome to the Medical Claims Analysis Dashboard.This comprehensive page allows you to explore and analyze the factors influencing medical claim values.
        By examining various demographic details and other related factors, you can gain valuable insights into how
        different elements impact claim amounts. Feel free to navigate, filter, and interact with the data to uncover
        trends and patterns relevant to your needs. Use this tool to make informed decisions and understand the intricacies
        of medical claims better.
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.header("Dataset Introduction")

    st.markdown(
        """
    </style>
    <div class="markdown-text-container">
        Datasets introduction , we will fill this later, i need inputs from dataset team
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Dataframes
    df1 = load_data(DATA_PATH1)
    if df1 is None:
        return
    df2 = load_data(DATA_PATH2)
    if df2 is None:
        return
    df2 = preprocess_data(df2)

    df3 = load_data(DATA_PATH3)
    if df3 is None:
        return
    # Rows where 'total_claim_amount' is NaN were dropped (since we can't analyze these)
    df3 = df3.dropna(subset=['total_claim_amount'])

    # Dealing with outliers using IQR
    Q1 = df3['total_claim_amount'].quantile(0.25)
    Q3 = df3['total_claim_amount'].quantile(0.75)
    IQR = Q3 - Q1
    df3 = df3[~((df3['total_claim_amount'] < (Q1 - 1.5 * IQR)) |
                (df3['total_claim_amount'] > (Q3 + 1.5 * IQR)))]

    # Age, gender analysis :

    col1, col2 = st.columns(2)

    with col1:

        # Age
        st.subheader("1. Age")
        age_bullet_points = """
            - Children (0-10 years): Recommended claim amount is $188,789.89, reflecting typical medical costs for this age group. <br>
            - Adolescents and Adults (11-50 years): Typical claim amounts are $163,336.22 (11-18 years) and $173,970.70 (19-50 years), considering common injury treatments. <br>
            - Elderly (51+ years): Reference claim amount is $129,096.38, reflecting the severity of injuries and associated medical costs for this demographic.
            """

        st.markdown(
            f"""
            <style>
            .text-area-wrapper {{
                background-color: rgba(128, 128, 128, 0.3); /* Grey with 30% opacity */
                padding: 10px;
                border-radius: 5px;
            }}
            </style>
            <div class="text-area-wrapper">{age_bullet_points}</div>
            """,
            unsafe_allow_html=True
        )

        color_palette = px.colors.qualitative.Plotly

        df1['AgeGroup'] = df1['age'].apply(age_group)
        average_claim_per_age_group = df1.groupby(
            'AgeGroup')['total_claim_amount'].mean().reindex(AGE_GROUP_ORDER).reset_index()

        fig1 = px.bar(average_claim_per_age_group, x='AgeGroup', y='total_claim_amount', text='total_claim_amount',
                      title='Average Claim Amount by Age Group', labels={'total_claim_amount': 'Average Claim Amount'}, color='AgeGroup', color_discrete_sequence=color_palette)

        fig1.update_traces(texttemplate='$%{text:.2f}', textposition='outside')
        st.plotly_chart(fig1)

    with col2:
        # Gender

        # Process the data
        df_sample = pd.read_csv(DATA_PATH4)
        df_sample = preprocess_sample_dataset(df_sample)
        df_ins = pd.read_csv(DATA_PATH5)
        df_ins = preprocess_insurance_data(df_ins)
        df_med = pd.read_csv(DATA_PATH1)
        st.subheader("2. Gender Data")
        data_source = st.radio(
            "Choose Data", ["Medical Practice", "Insurance Claims"])
        # data_source = st.selectbox(
        #     "Choose Data", ["Medical Practice", "Insurance Claims"])
        if data_source == "Medical Practice":
            data = df_med
            gender_bullet_points = """
                    - We will fill this later, i need inputs from dataset team <br>
                    """
            st.markdown(
                f"""
                <style>
                .text-area-wrapper {{
                    background-color: rgba(128, 128, 128, 0.2); /* Grey with 20% opacity */
                    padding: 10px;
                    border-radius: 5px;
                }}
                </style>
                <div class="text-area-wrapper">{gender_bullet_points}</div>
                """,
                unsafe_allow_html=True
            )
            data = df_med
            st.plotly_chart(plotly_gender(data))
            st.plotly_chart(plotly_box_gender(data))

        elif data_source == "Insurance Claims":
            gender_bullet_points = """
                    - We will fill this later, i need inputs from dataset team <br>
                    """
            st.markdown(
                f"""
                <style>
                .text-area-wrapper {{
                    background-color: rgba(128, 128, 128, 0.2); /* Grey with 20% opacity */
                    padding: 10px;
                    border-radius: 5px;
                }}
                </style>
                <div class="text-area-wrapper">{gender_bullet_points}</div>
                """,
                unsafe_allow_html=True
            )
            data = df_ins
            st.plotly_chart(plotly_gender(data))

    # Location /State wise analysis :

    col3, col4 = st.columns(2)

    with col3:

        # State wise

        st.subheader("3. State Wise Data")
        data_source = st.radio(
            "Choose Data", ["Sample Data", "Insurance Claims"])
        # data_source = st.selectbox(
        #     "Choose Data", ["Sample Data", "Insurance Claims"])
        if data_source == "Sample Data":
            data = df_sample
            state_bullet_points = """
                    - We will fill this later, i need inputs from dataset team <br>
                    """
            st.markdown(
                f"""
                <style>
                .text-area-wrapper {{
                    background-color: rgba(128, 128, 128, 0.2); /* Grey with 20% opacity */
                    padding: 10px;
                    border-radius: 5px;
                }}
                </style>
                <div class="text-area-wrapper">{state_bullet_points}</div>
                """,
                unsafe_allow_html=True
            )
            fig_states = plotly_states(data)
            # fig_states.update_layout(width=1200)
            st.plotly_chart(fig_states)

            # st.plotly_chart(plotly_states(data))
            fig_box_states = plotly_box_states(data)
            # fig_box_states.update_layout(width=1200)
            st.plotly_chart(fig_box_states)

        elif data_source == "Insurance Claims":
            data = df_ins
            st.plotly_chart(plotly_states(data))
            st.plotly_chart(plotly_box_states(data))

    with col4:

        # Post accident analysis
        st.subheader("4. Post Accident Analysis")
        plot_variable = st.radio("Select variable to plot", [
            'airbag_deployed', 'called_911', 'private_attorney'])

        if plot_variable == 'airbag_deployed':
            airbag_deployed_points = """
                - Airbag Deployment: Cases with airbag deployment had an average claim of $17,400, compared to $13,400 without deployment, emphasizing the impact of post-crash decisions. <br>
                """
            st.markdown(
                f"""
                <style>
                .text-area-wrapper {{
                    background-color: rgba(128, 128, 128, 0.2); /* Grey with 20% opacity */
                    padding: 10px;
                    border-radius: 5px;
                }}
                </style>
                <div class="text-area-wrapper">{airbag_deployed_points}</div>
                """,
                unsafe_allow_html=True
            )
            avg_claim_airbag = df2.groupby('airbag_deployed')[
                'total_claim_amount'].mean().reset_index()
            fig2 = px.bar(avg_claim_airbag, x='airbag_deployed', y='total_claim_amount', text='total_claim_amount',
                          title='Average Claim Amount by Airbag Deployment', labels={'total_claim_amount': 'Average Claim Amount'}, color='airbag_deployed', color_discrete_sequence=color_palette)

        elif plot_variable == 'called_911':
            post_accident_points = """
                - Emergency Calls: Contacting 911 post-accident increased average claims to $15,757.93, a 13% rise over cases without calls ($13,984.06). <br>
                """
            st.markdown(
                f"""
                <style>
                .text-area-wrapper {{
                    background-color: rgba(128, 128, 128, 0.2); /* Grey with 20% opacity */
                    padding: 10px;
                    border-radius: 5px;
                }}
                </style>
                <div class="text-area-wrapper">{post_accident_points}</div>
                """,
                unsafe_allow_html=True
            )

            avg_claim_911 = df2.groupby('called_911')[
                'total_claim_amount'].mean().reset_index()
            fig2 = px.bar(avg_claim_911, x='called_911', y='total_claim_amount', text='total_claim_amount',
                          title='Average Claim Amount by 911 Call', labels={'total_claim_amount': 'Average Claim Amount'}, color='called_911', color_discrete_sequence=color_palette)

        elif plot_variable == 'private_attorney':
            private_attorney_points = """
                - Private Attorney: Hiring a private attorney led to an average claim of $97,800, a 32% increase over cases without legal representation ($73,900), indicating their significant financial advantage in accident aftermaths.
                """
            st.markdown(
                f"""
                <style>
                .text-area-wrapper {{
                    background-color: rgba(128, 128, 128, 0.2); /* Grey with 20% opacity */
                    padding: 10px;
                    border-radius: 5px;
                }}
                </style>
                <div class="text-area-wrapper">{private_attorney_points}</div>
                """,
                unsafe_allow_html=True
            )
            avg_claim_by_attorney = df3.groupby('private_attorney')[
                'total_claim_amount'].mean().reset_index()
            fig2 = px.bar(avg_claim_by_attorney, x='private_attorney', y='total_claim_amount', text='total_claim_amount',
                          title='Average Claim Amount by Private Attorney', labels={'total_claim_amount': 'Average Claim Amount', 'private_attorney': 'Private Attorney'}, color=['No', 'Yes'], color_discrete_sequence=color_palette)

            fig2.update_layout(xaxis=dict(
                tickvals=[0, 1], ticktext=['No', 'Yes']))

        # elif plot_variable == 'age_group':
        #     age_bullet_points = """
        #         - Children (0-10 years): Recommended claim amount is $188,789.89, reflecting typical medical costs for this age group. <br>
        #         - Adolescents and Adults (11-50 years): Typical claim amounts are $163,336.22 (11-18 years) and $173,970.70 (19-50 years), considering common injury treatments. <br>
        #         - Elderly (51+ years): Reference claim amount is $129,096.38, reflecting the severity of injuries and associated medical costs for this demographic.
        #         """

        #     st.markdown(
        #         f"""
        #         <style>
        #         .text-area-wrapper {{
        #             background-color: rgba(128, 128, 128, 0.3); /* Grey with 30% opacity */
        #             padding: 10px;
        #             border-radius: 5px;
        #         }}
        #         </style>
        #         <div class="text-area-wrapper" style="width: 1200px">{age_bullet_points}</div>
        #         """,
        #         unsafe_allow_html=True
        #     )

        #     df1['AgeGroup'] = df1['age'].apply(age_group)
        #     average_claim_per_age_group = df1.groupby(
        #         'AgeGroup')['total_claim_amount'].mean().reindex(AGE_GROUP_ORDER).reset_index()

        #     fig2 = px.bar(average_claim_per_age_group, x='AgeGroup', y='total_claim_amount', text='total_claim_amount',
        #                   title='Average Claim Amount by Age Group', labels={'total_claim_amount': 'Average Claim Amount'}, color='AgeGroup', color_discrete_sequence=color_palette)

        fig2.update_layout(
            template='plotly_white',
            height=600,
        )
        fig2.update_traces(texttemplate='$%{text:.2f}', textposition='outside')
        # fig2.update_layout(width=1200)
        st.plotly_chart(fig2)

    col5, col6 = st.columns(2)

    with col5:

        # Severity Analysis
        # Mapping severity levels to categories
        st.subheader("5. Severity Analysis")
        severity_points = """
                    - We will fill this later, i need inputs from dataset team <br>
                    """
        st.markdown(
            f"""
                <style>
                .text-area-wrapper {{
                    background-color: rgba(128, 128, 128, 0.3); /* Grey with 30% opacity */
                    padding: 10px;
                    border-radius: 5px;
                }}
                </style>
                <div class="text-area-wrapper" ">{severity_points}</div>
                """,
            unsafe_allow_html=True
        )
        severity_mapping = {
            1: 'Very Low', 2: 'Low', 3: 'Moderate', 4: 'Moderately High',
            5: 'High', 6: 'Very High', 7: 'Severe', 8: 'Very Severe', 9: 'Critical'
        }
        # Create severity category column
        df1['severity_category'] = df1['severity'].map(severity_mapping)

        # Define the order of severity categories
        severity_order = ['Very Low', 'Low', 'Moderate', 'Moderately High',
                          'High', 'Very High', 'Severe', 'Very Severe', 'Critical']

        severity_avg_claim = df1.groupby('severity_category')[
            'total_claim_amount'].mean().reset_index()

        fig_severity_claim = px.bar(severity_avg_claim, x='severity_category', y='total_claim_amount',
                                    text='total_claim_amount', labels={'x': 'Severity Category', 'y': 'Average Claim Amount'},
                                    title='Average Claim Amount by Severity Category',
                                    color='total_claim_amount', color_continuous_scale='blues')

        fig_severity_claim.update_traces(
            texttemplate='$%{text:.2f}', textposition='outside')
        fig_severity_claim.update_layout(xaxis={'categoryorder': 'array', 'categoryarray': severity_order},
                                         xaxis_title="Severity Category", yaxis_title="Average Claim amount in USD")
        st.plotly_chart(fig_severity_claim)

    # data_source = st.selectbox(
    #     "Choose Data", ["Sample Data", "Medical Practice", "Insurance Claims"])

    # # Process the data
    # df_sample = pd.read_csv(DATA_PATH4)
    # df_sample = preprocess_sample_dataset(df_sample)
    # df_ins = pd.read_csv(DATA_PATH5)
    # df_ins = preprocess_insurance_data(df_ins)
    # df_med = pd.read_csv(DATA_PATH1)
    # if data_source == "Sample Data":
    #     data = df_sample
    #     st.subheader("3. State-wise Data")
    #     state_bullet_points = """
    #             - Most states have similar mean and median claim values for car accidents, except Florida. <br>
    #             - Florida stands out with an average car accident claim of $89,192.00. <br>
    #             - This amount exceeds the smallest average claim by $67,145.00 found in Pennsylvania.<br>
    #             - Large claims in Florida significantly impact both the state's average settlement values and the national average claim amount. <br>
    #             - Mean claims in states other than Florida range from approximately $22,000 to $28,000.
    #              """
    #     st.markdown(
    #         f"""
    #         <style>
    #         .text-area-wrapper {{
    #             background-color: rgba(128, 128, 128, 0.3); /* Grey with 30% opacity */
    #             padding: 10px;
    #             border-radius: 5px;
    #         }}
    #         </style>
    #         <div class="text-area-wrapper" style="width: 1200px">{state_bullet_points}</div>
    #         """,
    #         unsafe_allow_html=True
    #     )
    #     fig_states = plotly_states(data)
    #     fig_states.update_layout(width=1200)
    #     st.plotly_chart(fig_states)

    #     # st.plotly_chart(plotly_states(data))
    #     fig_box_states = plotly_box_states(data)
    #     fig_box_states.update_layout(width=1200)
    #     st.plotly_chart(fig_box_states)
    #     st.subheader("4. Injury Type")

    #     # Bar Chart of Injury Types
    #     fig_injury_bar = plotly_injury_bar(data)
    #     fig_injury_bar.update_layout(width=1200)
    #     st.plotly_chart(fig_injury_bar)

    #     # Injury Type Plots for Sample Data
    #     st.subheader("5. Distributions of Claims by Injury")

    #     # Option to pursue individual injuries
    #     injury = st.selectbox(
    #         "Choose an Injury to See the Distribution", data["Type of Injury"].unique())
    #     inj_data = data.loc[(data["Type of Injury"] == injury) & (
    #         data["claim_amount"] < data["claim_amount"].quantile(.95))]

    #     # Histogram
    #     fig_injury_hist = plotly_injury_hist(inj_data)
    #     fig_injury_hist.update_layout(width=1200)
    #     st.plotly_chart(fig_injury_hist)

    #     # Boxplot
    #     boxplot_fig = plotly_boxplot_injury(inj_data)
    #     boxplot_fig.update_layout(showlegend=False, width=1200)
    #     st.plotly_chart(boxplot_fig)

    # elif data_source == "Medical Practice":
    #     data = df_med
    #     st.subheader("3. Gender Data")
    #     st.plotly_chart(plotly_gender(data))
    #     st.subheader("4. Gender and Insurance Types")
    #     st.plotly_chart(plotly_box_gender(data))

    # elif data_source == "Insurance Claims":
    #     data = df_ins
    #     st.subheader("3. State-wise Data")
    #     st.plotly_chart(plotly_states(data))
    #     st.plotly_chart(plotly_box_states(data))
    #     st.subheader("4. Gender Data")
    #     st.plotly_chart(plotly_gender(data))


if __name__ == "__main__":

    display_analysis()
