import streamlit as st
import sqlite3
from random import randint

# Database initialization
DB_FILE = r'D:\CODING\projectNutri\tests\agents\client_health_data.db'

# Function to insert new client data into the database with a random 4 digit client ID
import logging

# Configure logging
logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')

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

# Streamlit form for new client data
with st.form("new_client_form"):
    st.title('Client Health Data Entry')
    st.write("Enter new client data:")
    # Client information
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