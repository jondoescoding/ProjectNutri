from datetime import datetime
import streamlit as st
import sqlite3
import logging
import os
import easyocr
import numpy as np
import pypdfium2 as pdfium
from dotenv import load_dotenv
from random import randint
from PIL import Image
from io import BytesIO
import json

## LANGCHAIN IMPORTS
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain import hub
from langchain_community.tools.pubmed.tool import PubmedQueryRun
from langchain_openai import ChatOpenAI
from langchain.schema import ChatMessage



# Configure logging
logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')

# Initialization
# DATABASE
DB_FILE = r'D:\CODING\LOCAL\projectNutri\tests\agents\client_health_data.db'

load_dotenv()

# API KEYS
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
LANGCHAIN_API_KEY = os.getenv('LANGCHAIN_API_KEY')
openai_api_key = os.getenv('OPENAI_API_KEY')
tavily_api_key = os.getenv("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")

## LLM
llm = OpenAI(temperature=0.5, model="gpt-3.5-turbo-instruct", api_key=openai_api_key)
#tools = load_tools(["pubmed"])
#tools = [PubmedQueryRun]
tool = PubmedQueryRun()

### ALL FUNCTIONS ####

## DATABASE FUNCTIONS ###
# Function to insert new client data into the database with a random 4 digit client ID
def insert_client_data(client_info, health_info):
    try:
        with sqlite3.connect(DB_FILE) as conn:
            c = conn.cursor()
            # Generate a random 4 digit client ID
            client_id = randint(1000, 9999)
            
            # Check if the generated client_id already exists in the database
            c.execute('SELECT * FROM clients WHERE client_id = ?', (client_id,))
            while c.fetchone():
                # If it exists, generate a new one
                client_id = randint(1000, 9999)
                c.execute('SELECT * FROM clients WHERE client_id = ?', (client_id,))
            
            # Insert into clients table with the new random client_id
            c.execute('''
                INSERT INTO clients (
                    client_id, first_name, last_name, gender, dob, occupation, address, phone
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (client_id,) + client_info)

            # Insert into healthprofile table
            c.execute('''
                INSERT INTO healthprofile (
                    client_id, height, allergies, weight, body_fat, waist_circumference,
                    fat_percentage, skin_fold, medications, supplements
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (client_id,) + health_info)
            conn.commit()
        logging.info(f"Client data inserted successfully with client ID: {client_id}")
        return client_id
    except sqlite3.Error as e:
        logging.error(f"Database error: {e}")
        raise
    except Exception as e:
        logging.error(f"Insertion error: {e}")
        raise

# Function to insert or update biomedical data into the database
def insert_biomedical_data(client_id, biomedical_data):
    with sqlite3.connect(DB_FILE) as conn:
        c = conn.cursor()
        # Ensure client_id is an integer
        client_id = int(client_id)
            
        # If biomedical_data is a dictionary or list, convert it to a string
        if isinstance(biomedical_data, (dict, list)):
            biomedical_data = json.dumps(biomedical_data)

        biomedical_data = str(biomedical_data)
            
        # Check if the client_id already has biomedical data
        c.execute('SELECT * FROM biomedical_data WHERE client_id = ?', (client_id,))
        existing_data = c.fetchone()
        
        if existing_data:
            # If data exists, update it or notify the user
            # For example, to update you could use:
            c.execute('''
                UPDATE biomedical_data
                SET biomedical_data = ?
                WHERE client_id = ?
            ''', (biomedical_data, client_id))
            action = "updated"
        else:
            # If no data exists, insert the new data
            c.execute('''
                INSERT INTO biomedical_data (
                    client_id, biomedical_data
                ) VALUES (?, ?)
            ''', (client_id, biomedical_data))
            action = "inserted"
        
        conn.commit()
        logging.info(f"Biomedical data {action} successfully for client ID: {client_id}")
        return action

### HELPER FUNCTIONS ###
# Function to retrieve biomedical data from the database
def get_biomedical_data(client_id):
    with sqlite3.connect(DB_FILE) as conn:
        c = conn.cursor()
        c.execute('SELECT biomedical_data FROM biomedical_data WHERE client_id = ?', (client_id,))
        data = c.fetchone()
        return data[0] if data else None

# Function to generate probable diagnosis using OpenAI's LLM
def generate_probable_diagnosis(symptoms, biomedical_data):
    template = """
    You are an expert at diagnosing illnesses.

    Having throught it through step by step give a possible and given both: the symptoms of an individual and their biomedical data, you should infer what illness they have.

    Patient Details
    Symptoms
    {symptoms}
    
    BioMedical Data
    {biomedical_data}

    Possible Illness:
    """
    
    prompt = PromptTemplate.from_template(template=template)

    chain = LLMChain(llm=llm, prompt=prompt)

    results = chain.invoke({"symptoms": symptoms, "biomedical_data": biomedical_data})['text']

    return results

# Function to update diagnosis in the database
def update_diagnosis(client_id, diagnosis, doctor_name, medical_centre, timestamp=False):
    with sqlite3.connect(DB_FILE) as conn:
        c = conn.cursor()
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S") if timestamp else None
        c.execute('''
            INSERT INTO diagnosis (client_id, diagnosis, doctor_name, medical_centre, timestamp)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(client_id) DO UPDATE SET
            diagnosis=excluded.diagnosis,
            doctor_name=excluded.doctor_name,
            medical_centre=excluded.medical_centre,
            timestamp=excluded.timestamp
        ''', (client_id, diagnosis, doctor_name, medical_centre, current_time))
        conn.commit()


## OTHER FUNCTIONS ###
# Convert PDF file into images via pypdfium2
def convert_pdf_to_images(file_path, scale=300/72):

    pdf_file = pdfium.PdfDocument(file_path)

    page_indices = [i for i in range(len(pdf_file))]

    renderer = pdf_file.render(
        pdfium.PdfBitmap.to_pil,
        page_indices=page_indices,
        scale=scale,
    )

    final_images = []

    for i, image in zip(page_indices, renderer):

        image_byte_array = BytesIO()
        image.save(image_byte_array, format='jpeg', optimize=True)
        image_byte_array = image_byte_array.getvalue()
        final_images.append(dict({i: image_byte_array}))

    return final_images

# Extract text from images using EasyOCR
def extract_text_from_img(list_dict_final_images):
    reader = easyocr.Reader(['en'])  # Initialize the EasyOCR reader for English language
    image_content = []

    for image_dict in list_dict_final_images:
        for image_index, image_bytes in image_dict.items():
            image = Image.open(BytesIO(image_bytes))
            results = reader.readtext(np.array(image))  # Use EasyOCR to read text from the image array
            text_output = ' '.join([result[1] for result in results])  # Extract the text from the results
            image_content.append(text_output)

    return "\n".join(image_content)

# Extract structured info from text via LLM
def extract_structured_data(content: str):
    template = """
    You are a medical data extraction specialist. Below is the medical data content of a client

    {content}

        
    Extracting the following: 
    
    # Format
    Client Name
    
    Test Done
    
    Test Results 
    
    Unit  
    
    Reference Range
    """

    prompt = PromptTemplate.from_template(template=template)

    chain = LLMChain(llm=llm, prompt=prompt)

    results = chain.invoke({"content": content})['text']

    return results

### STREAMLIT SECTION ####
# Streamlit form for new client data - 1
def client_data_entry_form():
    st.title('Client Health Data Entry')
    st.write("Enter new client data:")
    # Client information
    with st.form("new_client_form"):
        first_name = st.text_input("First Name")
        last_name = st.text_input("Last Name")
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        dob = st.date_input("Date of Birth")
        occupation = st.text_input("Occupation")
        address = st.text_area("Address")
        phone = st.text_input("Phone")
        
        # Health profile information
        height = st.number_input("Height (cm)")
        allergies = st.text_area("Allergies")
        weight = st.number_input("Weight (kg)")
        body_fat = st.number_input("Body Fat (%)")
        waist_circumference = st.number_input("Waist Circumference (cm)")
        fat_percentage = st.number_input("Fat Percentage (%)")
        skin_fold = st.number_input("Skin Fold (mm)")
        medications = st.text_area("Medications")
        supplements = st.text_area("Supplements")
        
        submitted = st.form_submit_button("Submit")
        if submitted:
            client_info = (
                first_name, last_name, gender, dob, occupation, address, phone
            )
            health_info = (
                height, allergies, weight, body_fat, waist_circumference,
                fat_percentage, skin_fold, medications, supplements
            )
            client_id = insert_client_data(client_info, health_info)
            st.success(f"Client data submitted successfully! Client ID: {client_id}")

# New Streamlit page for biomedical data entry - 2
def biomedical_data_entry_page():
    st.title('Patient Bio Medical Data')
    
    # Initialize session state variables if they don't exist
    if 'client_id' not in st.session_state:
        st.session_state.client_id = ''
    
    client_id = st.text_input("Enter the client's ID", value=st.session_state.client_id)
    st.session_state.client_id = client_id  # Update session state
    
    # Directly upload a PDF file without asking the user
    uploaded_file = st.file_uploader("Upload the client's health data (PDF only)", type=['pdf'])
    
    if uploaded_file is not None:
        # Button to trigger the processing of the uploaded file
        if st.button('Process Biomedical Data'):
            with st.spinner('Processing... Please wait'):
                images_list = convert_pdf_to_images(uploaded_file)
                extracted_text = extract_text_from_img(images_list)
                structured_data = extract_structured_data(extracted_text)
                st.session_state.structured_data = structured_data  # Update session state
                st.subheader("Extracted AI Text")
                st.write(structured_data)
            
            action = insert_biomedical_data(client_id, structured_data)

            # Button to submit the structured data to the database
            if st.button('Submit Biomedical Data'):
                if action == "inserted":
                    st.success("Biomedical data submitted successfully!")
                elif action == "updated":
                    st.success("Biomedical data updated successfully!")

# New Streamlit page for diagnosis - 3
def diagnosis_page():
    st.title('Patient Diagnosis')

    # Initialize session state variables if they don't exist
    if 'client_id' not in st.session_state:
        st.session_state.client_id = ''

    client_id = st.text_input("Enter the client's ID", value=st.session_state.client_id)
    st.session_state.client_id = client_id  # Update session state

    # Ask for symptoms
    symptoms = st.text_area("Enter the symptoms shown by the client")

    # Ask if the patient has already been diagnosed
    already_diagnosed = st.radio("Has the patient already been diagnosed?", ("Yes", "No"))

    if already_diagnosed == "Yes":
        diagnosis_given = st.text_input("Enter the diagnosis given")
        doctor_name = st.text_input("Enter the name of the doctor who gave the diagnosis")
        medical_centre = st.text_input("Enter the name of the medical centre")

        if st.button('Submit Diagnosis'):
            # Update the diagnosis in the database
            update_diagnosis(client_id, diagnosis_given, doctor_name, medical_centre)
            st.success("Diagnosis information updated successfully!")
    else:
        if st.button('Generate Probable Diagnosis'):
            with st.spinner('Generating diagnosis... Please wait'):
                # Retrieve biomedical data
                biomedical_data = get_biomedical_data(client_id)
                # Generate probable diagnosis using OpenAI's LLM
                probable_diagnosis = generate_probable_diagnosis(symptoms, biomedical_data)
                st.write(probable_diagnosis)
                # Update the diagnosis in the database with the current timestamp
                update_diagnosis(client_id,probable_diagnosis, "AI Generated", "N/A", timestamp=True)

# Streamlit page for treatment recommendations - 4
def treatment_page():
    st.title('Patient Treatment and Meal Plan')

    # Initialize session state variables if they don't exist
    if 'client_id' not in st.session_state:
        st.session_state.client_id = ''
    if 'recommendations' not in st.session_state:
        st.session_state.recommendations = ''
    if 'recommendations_generated' not in st.session_state:  # Track whether recommendations have been generated
        st.session_state.recommendations_generated = False

    client_id = st.text_input("Enter the client's ID", value=st.session_state.client_id)
    st.session_state.client_id = client_id  # Update session state

    # Prompt for food diary PDF upload
    food_diary_pdf = st.file_uploader("Upload the client's food diary (PDF only)", type=['pdf'])

    # Button to generate recommendations
    if st.button('Generate Recommendations'):
        with st.spinner('Generating recommendations... Please wait'):
            # Retrieve client's diagnosis and personal data
            diagnosis = get_biomedical_data(client_id)
            personal_data = get_client_personal_data(client_id)
            
            # Extract text from food diary PDF if uploaded
            food_diary_text = ""
            if food_diary_pdf is not None:
                images_list = convert_pdf_to_images(food_diary_pdf)
                food_diary_text = extract_text_from_img(images_list)
            
            # Generate recommendations
            recommendations = generate_recommendations(diagnosis, personal_data, food_diary_text)
            st.session_state.recommendations = recommendations
            st.session_state.recommendations_generated = True  # Update the state to indicate that recommendations have been generated
            st.write(recommendations)

    # Button to submit the recommendations to the database
    if st.session_state.recommendations_generated:  # Check if recommendations have been generated
        if st.button('Submit Recommendations'):
            # Update the recommendations in the database with the current timestamp
            update_treatment_and_meal_plan(client_id, st.session_state.recommendations, timestamp=True)
            st.success("Recommendations generated and saved successfully!")


# Function to get client's personal data
def get_client_personal_data(client_id):
    with sqlite3.connect(DB_FILE) as conn:
        c = conn.cursor()
        c.execute('''
            SELECT first_name, last_name, gender, dob, occupation, address, phone, medications, supplements
            FROM clients
            JOIN healthprofile ON clients.client_id = healthprofile.client_id
            WHERE clients.client_id = ?
        ''', (client_id,))
        data = c.fetchone()
        return data if data else None

# Function to generate a 4-week meal plan based on food recommendations
def generate_meal_plan(food_recommendations):
    template = """
    You are a Jamaican nutritionist.

    Create a 5 day Jamaican based food ONLY meal plan using the following food recommendations. 
    
    Ensure the meal plan is in bullet list style.

    Food Recommendations:
    {food_recommendations}

    Meal Plan:
    Breakfast
    - 
    
    Lunch
    - 
    
    Dinner
    
    -
    """

    prompt = PromptTemplate.from_template(template=template)

    chain = LLMChain(llm=llm, prompt=prompt)

    meal_plan = chain.invoke({
        "food_recommendations": food_recommendations
    })['text']

    return meal_plan

# Function to generate recommendations based on diagnosis, personal data, and food diary
def generate_recommendations(diagnosis, personal_data, food_diary_text):
    template = """
    You are a health consultant.

    Based on the following client's diagnosis, personal data, and their food diary, provide a complete breakdown of this individual's current health situation and lifestyle recommendations.

    Diagnosis:
    {diagnosis}

    Personal Data:
    {personal_data}

    Food Diary:
    {food_diary_text}

    Recommendations:
    """

    prompt = PromptTemplate.from_template(template=template)

    chain = LLMChain(llm=llm, prompt=prompt)

    recommendations = chain.invoke({
        "diagnosis": diagnosis,
        "personal_data": personal_data,
        "food_diary_text": food_diary_text
    })['text']

    return recommendations

# Function to update treatment and meal plan in the database
def update_treatment_and_meal_plan(client_id, treatment_and_meal_plan, timestamp=False):
    with sqlite3.connect(DB_FILE) as conn:
        c = conn.cursor()
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S") if timestamp else None
        c.execute('''
            INSERT INTO treatment_plan (client_id, treatment_and_meal_plan, timestamp)
            VALUES (?, ?, ?)
            ON CONFLICT(client_id) DO UPDATE SET
            treatment_and_meal_plan=excluded.treatment_and_meal_plan,
            timestamp=excluded.timestamp
        ''', (client_id, treatment_and_meal_plan, current_time))
        conn.commit()

# Function to retrieve treatment and meal plan from the database
def get_recommendations(client_id):
    with sqlite3.connect(DB_FILE) as conn:
        c = conn.cursor()
        c.execute('SELECT diagnosis FROM diagnosis WHERE client_id = ?', (client_id,))
        result = c.fetchone()
        return result[0] if result else None

# Function to interact with OpenAI's LLM in a chat-like interface
def interact_with_llm(user_input, context):
    # Combine the context and user input into a single string
    combined_context = "\n".join(context + [f"User: {user_input}"])
    
    # Create the prompt for the LLM
    prompt = combined_context + "\nLLM:"

    # Invoke the LLM with the prompt
    # Assuming `llm` is an instance of `ChatOpenAI` or similar
    response = llm.invoke(prompt=prompt, max_tokens=150)

    # Extract the text from the response
    llm_response = response['choices'][0]['text'].strip()

    return llm_response

def adjusted_meal_plan(original_meal_plan, user_adjustments):
    # Define the prompt template for the LLM to make adjustments to the meal plan
    template = """
    You are a nutritionist who has provided a meal plan for a client. 

    You are given some adjustments for the client's meal plan. 

    Please update the meal plan by replacing foods. 

    Original Meal Plan:
    {original_meal_plan}

    Adjustments:
    {user_adjustments}

    Updated Meal Plan:
    
    """

    prompt = PromptTemplate.from_template(template=template)

    chain = LLMChain(llm=llm, prompt=prompt)

    # Invoke the LLM with the original meal plan and the user's adjustments
    updated_meal_plan = chain.invoke({
        "original_meal_plan": original_meal_plan,
        "user_adjustments": user_adjustments
    })['text']

    return updated_meal_plan

# Modify the meal_plan_page function to include a call to adjusted_meal_plan
def meal_plan_page():
    st.title('Patient Meal Plan Generation')

    # Input field for client ID
    client_id = st.text_input("Enter the client's ID")

    # Button to retrieve and generate meal plan
    if st.button('Generate Meal Plan'):
        if validate_client_id(client_id):
            recommendations = get_recommendations(client_id)
            if recommendations:
                meal_plan = generate_meal_plan(recommendations)
                st.session_state['meal_plan'] = meal_plan  # Save the generated meal plan in session state
                st.markdown(meal_plan)  # Display the generated meal plan as markdown
            else:
                st.error("No recommendations found for this client.")
        else:
            st.error("Invalid client ID. Please try again.")


# Function to validate client ID
def validate_client_id(client_id):
    with sqlite3.connect(DB_FILE) as conn:
        c = conn.cursor()
        c.execute('SELECT EXISTS(SELECT 1 FROM clients WHERE client_id=?)', (client_id,))
        return c.fetchone()[0]


# Main function to run the Streamlit app
if __name__ == "__main__":
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Client Data Entry", "Biomedical Data Entry", "Diagnosis", "Treatment", "Meal Plan"])

    if page == "Client Data Entry":
        client_data_entry_form()
    elif page == "Biomedical Data Entry":
        biomedical_data_entry_page()
    elif page == "Diagnosis":
        diagnosis_page()
    elif page == "Treatment":
        treatment_page()
    elif page == "Meal Plan":
        meal_plan_page()