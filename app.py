import streamlit as st
from openai import OpenAI
import pandas as pd
import matplotlib.pyplot as plt
import concurrent.futures
import seaborn as sns
from tenacity import retry, wait_random_exponential, stop_after_attempt
import json 
from enum import Enum
from pydantic import BaseModel
import io
import zipfile

from prompts import *

# Streamlit app title
st.title("DHS CS Review")

# User password for page protection
def check_password():
    """Function to check user password to protect the page"""
    password = st.text_input("Enter Password", type="password")
    if password == st.secrets["password"]:
        return True
    elif password:
        st.error("Incorrect password")
    return False

def initialize_openai_client(api_key: str) -> OpenAI:
    """Initialize OpenAI client"""
    return OpenAI(api_key=api_key)

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def openai_chatbot(client, system_content, user_content, response_format=None, max_tokens=4096):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content}
        ],
        max_tokens=max_tokens,
        response_format=response_format,
    )
    return response.choices[0].message.content.strip()

def openai_chatbot_o1(client, user_content, response_format=None, max_tokens=4096):
    response = client.chat.completions.create(
        model="o1-preview",
        messages=[
            {"role": "user", "content": user_content}
        ]
    )
    return response.choices[0].message.content.strip()

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def openai_chatbot_json(client, system_content, user_content, response_format=None, max_tokens=4096):
    response = client.beta.chat.completions.parse(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content}
        ],
        max_tokens=max_tokens,
        response_format=response_format,
    )
    return response.choices[0].message.parsed

def parallel_process_openai_chatbot(client, system_content, user_contents, chatbot_function):
    input_data = [(client, system_content, user_content) for user_content in user_contents]
    # Progress bar
    progress_bar = st.progress(0)
    progress_text = st.empty()
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        future_to_index = {
            executor.submit(chatbot_function, client, system_content, user_content): idx
            for idx, (client, system_content, user_content) in enumerate(input_data)
        }
        results = [None] * len(user_contents)
        total = len(user_contents)
        for count, future in enumerate(concurrent.futures.as_completed(future_to_index), start=1):
            index = future_to_index[future]
            try:
                result = future.result()
            except Exception as exc:
                st.write(f'Generated an exception: {exc}')
            else:
                results[index] = result
                progress = count / total
                progress_bar.progress(progress)
                progress_text.text(f"Processing unit {count} of {total}...")
    return results

class ContentCompliance(BaseModel):
    is_simple: bool

class RiskRatingEnum(str, Enum):
    A = "A"
    B = "B"
    C = "C"
    D = "D"
    # Unknown = "Unknown"

class ClueRatingEnum(str, Enum):
    A = "A"
    B = "B"
    C = "C"
    D = "D"
    # Unknown = "Unknown"

class RiskRating(BaseModel):
    rating: RiskRatingEnum

class ClueRating(BaseModel):
    rating: ClueRatingEnum

def main():
    if check_password():
        # Initialize OpenAI client
        api_key = st.secrets["OPENAI_API_KEY"]  # Ensure you have this key in your Streamlit secrets
        client = initialize_openai_client(api_key)

        st.sidebar.write("**This app assumes a 2 column CSV file**: Column A: 'Datetime', Column B: 'Content'")

        # File uploader to allow users to upload multiple file types
        uploaded_file = st.sidebar.file_uploader(
            "Upload a Document",
            type=["csv"]
        )
        
        # Num rows
        number_rows = st.sidebar.number_input("Number of Rows to Analyze", value=400)
        
        # # Which steps to run -- possible future feature if needed
        # st.sidebar.write("Steps to Run:")
        # include_time_analysis = st.sidebar.checkbox("Time Analysis", value=True)
        # include_filtering = st.sidebar.checkbox("Filtering", value=True)
        # include_categorization = st.sidebar.checkbox("Categorization", value=True)
        # include_assessments = st.sidebar.checkbox("Assessments", value=True)

        # Which analyses to run
        st.sidebar.write("Select Analyses to Include:")
        include_risk = st.sidebar.checkbox("Risk", value=False)
        include_clue = st.sidebar.checkbox("Clue", value=True)

        # Which ratings to include
        st.sidebar.write("Select Ratings to Include:")
        include_a = st.sidebar.checkbox("A", value=True)
        include_b = st.sidebar.checkbox("B", value=False)
        include_c = st.sidebar.checkbox("C", value=False)
        include_d = st.sidebar.checkbox("D", value=False)
        
        TabA, TabB, TabC, TabE, TabF, TabG = st.tabs(["Time Analysis", "Filtering", "Message Categorization", "**Risk Assessment**", "**Location Assessment**", "**Download Files**"])
        # Take Datetime column and infer what timezone the user likely lives in
        if uploaded_file:  
            if st.sidebar.button("Start"):
                with st.sidebar:
                    # Wait message
                    st.session_state["wait_message"] = "**Analysis has begun!** You'll know that the script is complete if the Running icon has stopped spinning in the top right hand corner! You'll also see a button in the last tab to download the reporting as a ZIP file. It should take <5 min to run."
                    st.write(st.session_state["wait_message"])

                ##################
                ### A: Time analysis
                ##################
                with TabA:
                    st.session_state["A_Header"] = '# A. Time analysis'
                    st.write(st.session_state["A_Header"])

                    # Read the CSV file
                    df = pd.read_csv(uploaded_file)

                    # Convert 'Datetime' column to datetime
                    df['Datetime'] = pd.to_datetime(df['Datetime'])

                    # Extract day of the week
                    df['DayOfWeek'] = df['Datetime'].dt.day_name()

                    # Extract hour of the day
                    df['HourOfDay'] = df['Datetime'].dt.hour

                    # Count messages by day of the week
                    messages_by_day = df['DayOfWeek'].value_counts().reindex([
                        'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'
                    ], fill_value=0)

                    # Count messages by hour of the day
                    messages_by_hour = df['HourOfDay'].value_counts().sort_index()

                    # Plot bar chart for day of the week
                    st.session_state["A_MessagesByDay"] = "### Number of Messages by Day of the Week"
                    st.write(st.session_state["A_MessagesByDay"])
                    st.bar_chart(messages_by_day)

                    # Plot bar chart for hour of the day
                    st.session_state["A_MessagesByHour"] = "### Number of Messages by Hour of the Day"
                    st.write(st.session_state["A_MessagesByHour"])
                    st.bar_chart(messages_by_hour)

                    # Create a heatmap for day of the week vs hour of the day
                    heatmap_data = df.groupby(['DayOfWeek', 'HourOfDay']).size().unstack(fill_value=0).reindex([
                        'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'
                    ])

                    # Plot heatmap using matplotlib
                    st.session_state["A_Heatmap"] = "### Heatmap of Messages by Day of the Week and Hour of the Day"
                    st.write(st.session_state["A_Heatmap"])
                    plt.figure(figsize=(10, 6))
                    sns.heatmap(heatmap_data, cmap="YlGnBu", annot=True, fmt=".0f")  # Changed fmt="d" to fmt=".0f"
                    plt.xlabel("Hour of the Day")
                    plt.ylabel("Day of the Week")
                    st.pyplot(plt)

                    heatmap_json_str = json.dumps(heatmap_data.to_dict(orient='index'))
                    system_prompt = prompt_sect1_quest1_system + heatmap_json_str
                    user_prompt = prompt_sect1_quest1_user + heatmap_json_str

                    st.session_state["A_BestCaseLocation"] = "### Best case on location based on time zones"
                    st.write(st.session_state["A_BestCaseLocation"])
                
                    st.session_state["response"] = openai_chatbot(client, system_content=system_prompt, user_content=user_prompt)
                    st.write(st.session_state["response"])
                    time_analysis_saved = st.session_state["response"] 

                    st.session_state["df"] = df
                    st.write(st.session_state["df"])

                ##################
                ### B: Filter out data
                ##################

                with TabB:
                    # Filter data
                    st.session_state["B_Header"] = "# B. Filter out simple"
                    st.write(st.session_state["B_Header"])

                    df = df.head(number_rows) # narrow to 1500

                    # Extract the 'Content' column
                    content_list = df['Content'].tolist()

                    # Define system and user prompts
                    system_prompt = prompt_sect2_quest1_system
                    user_prompts = [content for content in content_list]

                    # Run parallel processing
                    responses = parallel_process_openai_chatbot(client, system_prompt, user_prompts, lambda c, s, u: openai_chatbot_json(c, s, u, response_format=ContentCompliance))

                    # Add a new column 'is_simple' to the DataFrame
                    try:
                        df.loc[:, 'is_simple'] = [response.is_simple if response is not None else False for response in responses]
                    except Exception as e:
                        st.write(f"**Error: There was an issue with the OpenAI API. Please try again.**\n{e}")
                        st.write(responses)

                    # Calculate and visualize the percentage of rows filtered out
                    total_rows = len(df)
                    filtered_rows = df['is_simple'].sum()
                    filtered_percentage = filtered_rows / total_rows * 100  # Adjusted calculation

                    st.session_state["B_PercentageOfRowsFilteredOut"] = "### Percentage of Rows Filtered Out"
                    st.write(st.session_state["B_PercentageOfRowsFilteredOut"])

                    st.session_state["B_PercentageOfRowsFilteredOut_Text"] = f"{filtered_percentage:.2f}% of rows were filtered out."
                    st.write(st.session_state["B_PercentageOfRowsFilteredOut_Text"])

                    # Visualize the percentage of rows filtered out
                    st.bar_chart(pd.DataFrame({
                        'Status': ['Filtered Out', 'Remaining'],
                        'Count': [filtered_rows, total_rows - filtered_rows]  # Adjusted order
                    }).set_index('Status'))

                    st.session_state["B_FilteredDF"] = df[df['is_simple'] == False]
                    st.write(st.session_state["B_FilteredDF"])

                ##################
                ### C: Categorize prompts
                ##################

                with TabC:
                    # Filter out rows where 'is_simple' is False
                    complex_contents = df[df['is_simple'] == False]['Content'].tolist()

                    # Define system prompt for categorization
                    system_prompt_risk = prompt_sect4_risks # "Your task is to categorize the content based on risk."
                    system_prompt_clue = prompt_sect4_clues # "Your task is to categorize the content based on clues."

                    # Run parallel processing for RiskRating
                    if include_risk:
                        risk_responses = parallel_process_openai_chatbot(
                            client, system_prompt_risk, complex_contents, 
                            lambda c, s, u: openai_chatbot_json(c, s, u, response_format=RiskRating)
                        )
                    else: 
                        risk_responses = ["Unknown"] * len(complex_contents)

                    # Run parallel processing for ClueRating
                    if include_clue:
                        clue_responses = parallel_process_openai_chatbot(
                            client, system_prompt_clue, complex_contents, 
                            lambda c, s, u: openai_chatbot_json(c, s, u, response_format=ClueRating)
                        )
                    else: 
                        clue_responses = ["Unknown"] * len(complex_contents)

                    # Add new columns 'risk_rating' and 'clue_rating' to the DataFrame
                    # df.loc[df['is_simple'] == False, 'risk_rating'] = [response.rating for response in risk_responses]
                    df.loc[df['is_simple'] == False, 'risk_rating'] = [
                        response.rating if hasattr(response, 'rating') else "Unknown" for response in risk_responses
                    ]
                    # df.loc[df['is_simple'] == False, 'clue_rating'] = [response.rating for response in clue_responses]
                    df.loc[df['is_simple'] == False, 'clue_rating'] = [
                        response.rating if hasattr(response, 'rating') else "Unknown" for response in clue_responses
                    ]

                    # Visualize the percentage of each rating for risk and clue
                    st.session_state["C_Header"] = "# C. Risk and Clue Analysis"
                    st.write(st.session_state["C_Header"])

                    st.session_state["C_RiskRatingDistribution"] = "### Risk Rating Distribution\n"
                    st.write(st.session_state["C_RiskRatingDistribution"])

                    st.session_state["C_RiskRatingDistribution_ScoringSystem_Text"] = '''
                    Scoring System: \n
                    (A) The user's text suggests an immediate, tangible action against a child\n
                    (B) The user's text refers generally to an action against a child\n
                    (C) The user's text likely qualifies as CSAM but not an action\n
                    (D) The user's text is not related to CSAM\n
                    '''
                    st.write(st.session_state["C_RiskRatingDistribution_ScoringSystem_Text"])
                    df_complex = df[df['is_simple'] == False]
                    risk_counts = df_complex['risk_rating'].dropna().value_counts()
                    st.bar_chart(risk_counts)

                    st.session_state["C_ClueRatingDistribution"] = "### Clue Rating Distribution"
                    st.write(st.session_state["C_ClueRatingDistribution"])

                    st.session_state["C_ClueRatingDistribution_ScoringSystem_Text"] = '''
                    Scoring System: \n
                    (A) The user's text has an obvious hint that investigators should take a closer look at\n
                    (B) The user's text has a discernable hint at language, location, etc.\n
                    (C) The user's text has a very faint hint at language, location, etc.\n
                    (D) The user's text provides absolutely no relevant information on language, location, etc.\n
                    '''
                    st.write(st.session_state["C_ClueRatingDistribution_ScoringSystem_Text"])

                    clue_counts = df_complex['clue_rating'].dropna().value_counts()
                    st.bar_chart(clue_counts)

                    st.session_state["C_FilteredDF"] = df_complex
                    st.write(st.session_state["C_FilteredDF"])

                ##################
                ### D: Run prompts one by one (on the complex filtered df!)
                ##################

                with TabC:
                    st.session_state["D_Header"] = "# D. Clue and Risk Analysis (Detailed)"
                    st.write(st.session_state["D_Header"])

                    ratings_to_include = []
                    if include_a:
                        ratings_to_include.append('A')
                    if include_b:
                        ratings_to_include.append('B')
                    if include_c:
                        ratings_to_include.append('C')
                    if include_d:
                        ratings_to_include.append('D')

                    df_complex = df_complex.reset_index().rename(columns={'index': 'id'})
                    df_risk = df_complex[df_complex['risk_rating'].isin(ratings_to_include)]
                    df_clue = df_complex[df_complex['clue_rating'].isin(ratings_to_include)]
                    # df_complex_contents_with_risk_or_clue = df_complex[(df_complex['risk_rating'].isin(['A'])) | (df_complex['clue_rating'].isin(['A']))] 
                    list_risk = df_risk["Content"].tolist()
                    list_clue = df_clue["Content"].tolist()
                    # complex_contents_with_risk_or_clue_list = df_complex_contents_with_risk_or_clue["Content"].tolist()

                    # Generate risk justifications
                    if include_risk: 
                        risk_responses = parallel_process_openai_chatbot(
                            client, 
                            system_prompt_risk, 
                            list_risk, 
                            lambda c, s, u: openai_chatbot(c, s, u))
                    else: 
                        risk_responses = ["Unknown"] * len(list_risk)

                    risk_responses_df = pd.DataFrame({
                        'id': df_risk['id'].values,  # Ensure this matches the order of risk_responses
                        'risk_report': risk_responses
                    })

                    # Generate clue justifications
                    if include_clue: 
                        clue_responses = parallel_process_openai_chatbot(
                            client, 
                            system_prompt_clue, 
                            list_clue, 
                            lambda c, s, u: openai_chatbot(c, s, u)
                        )
                    else: 
                        clue_responses = ["Unknown"] * len(list_clue) 

                    clue_responses_df = pd.DataFrame({
                        'id': df_clue['id'].values,  # Ensure this matches the order of risk_responses
                        'clue_report': clue_responses
                    })

                    # Add new columns 'risk_rating' and 'clue_rating' to the DataFrame
                    df_complex = df_complex.merge(risk_responses_df, on='id', how='left')
                    df_complex = df_complex.merge(clue_responses_df, on='id', how='left')

                    st.session_state["D_DetailedDF"] = df_complex
                    st.write(st.session_state["D_DetailedDF"])

                ##################
                ### E: Build a meta-prompt for risk and clues
                ##################

                # Define variables
                df_high_risk_msgs = df_complex[df_complex['risk_rating'] == 'A']
                df_high_clue_msgs = df_complex[df_complex['clue_rating'] == 'A']

                # Create a list of formatted strings
                high_risk_contents_list = [
                    f"Timestamp: {row['Datetime']}, Message: {row['Content']}, Risk Message: {row['risk_report']}"
                    for _, row in df_high_risk_msgs.iterrows()
                ]

                high_clue_contents_list = [
                    f"Timestamp: {row['Datetime']}, Message: {row['Content']}, Clue Message: {row['clue_report']}"
                    for _, row in df_high_clue_msgs.iterrows()
                ]
                # Run inference on high risk contents
                if include_risk: 
                    overall_meta_response_risk = openai_chatbot_o1(
                        client, 
                        user_content=f"{prompt_sect5_meta_risk}\nHere are the high risk contents: {high_risk_contents_list}"
                        )
                else: 
                    overall_meta_response_risk = "Skip risk section"
                with TabE:  
                    st.session_state["E_MetaReportRisk_Text"] = overall_meta_response_risk
                    st.write(st.session_state["E_MetaReportRisk_Text"])

                # Run inference on high clue contents
                if include_clue:
                    overall_meta_response_clue = openai_chatbot_o1(
                        client, 
                        user_content=f"{prompt_sect5_meta_clue}\nHere are the high clue contents: {high_clue_contents_list}. Additional location context from previous analysis: {time_analysis_saved}"
                        )
                else: 
                    overall_meta_response_clue = "Skip clue section"
                with TabF:
                    st.session_state["E_MetaReportClue_Text"] = overall_meta_response_clue
                    st.write(st.session_state["E_MetaReportClue_Text"])

                with st.sidebar:
                    st.write("**Analysis now complete!** You can now download the reporting as a ZIP file in the last tab.")

                ##################
                ### F: Export CSV and Two Reports
                ##################

                # @st.fragment
                def download_zip_file():
                    zip_buffer = io.BytesIO()
                    with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED) as zf:
                        # Add the risk and clue report to the zip
                        zf.writestr("risk_and_clue_report.txt", f"{overall_meta_response_risk}\n{overall_meta_response_clue}")
                        # Add the CSV data to the zip
                        zf.writestr("post-processed_data.csv", df.to_csv(index=False))

                    # Ensure the buffer's position is at the start
                    zip_buffer.seek(0)

                    # Streamlit download button for the zip file
                    st.download_button(
                        label="Download Reports as ZIP",
                        data=zip_buffer,
                        file_name="reports.zip",
                        mime="application/zip"
                    )
                with TabG:
                    download_zip_file()

if __name__ == "__main__":
    main()