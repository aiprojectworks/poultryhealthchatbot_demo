__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import streamlit as st
import sqlite3
from datetime import datetime
from myagent import sql_dev, extract_data, Crew, Process, data_analyst, \
    analyze_data, data_insert_validator, validate_insert_data, alert_agent,alert_task, \
    data_json_validator,validate_json_data,store_data,alert_json_task,update_data
from openai import OpenAI
import openai
import base64
import json
import psycopg2
from dotenv import load_dotenv
import os
import random
import string
import time

import sys
sys.path.append('./')

# Load environment variables from .env
load_dotenv()

# Fetch variables
USER = os.getenv("user")
PASSWORD = os.getenv("password")
HOST = os.getenv("host")
PORT = os.getenv("port")
DBNAME = os.getenv("dbname")

def generate_custom_id():
    timestamp = str(int(time.time()))
    random_chars = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
    return f"ID-{timestamp}-{random_chars}"

client = OpenAI()  # Uses OPENAI_API_KEY from environment

# Database setup
# def init_db():
#     conn = sqlite3.connect('poultry_health.db')
#     c = conn.cursor()
    
#     # Create tables if they don't exist
#     c.execute('''CREATE TABLE IF NOT EXISTS poultry_health_records
#                  (id INTEGER PRIMARY KEY AUTOINCREMENT,
#                   body_weight REAL,
#                   body_temperature REAL,
#                   vaccination_records TEXT,
#                   symptoms TEXT,
#                   image_analysis TEXT,
#                   created_at TIMESTAMP)''')
                  
#     c.execute('''CREATE TABLE IF NOT EXISTS biosecurity_records
#                  (id INTEGER PRIMARY KEY AUTOINCREMENT,
#                   location TEXT,
#                   violation TEXT,
#                   image_analysis TEXT,
#                   created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
#     conn.commit()
#     conn.close()

# # Initialize database
# init_db()

# Main app
st.title("Poultry Farm Management System")

# Sidebar navigation
menu = st.sidebar.selectbox("Select Function", 
                           ["Poultry Health Entry",
                           "Poultry Health Entry 1",
                            "Poultry Health Update", 
                            "Biosecurity Entry", 
                            "Database Query"])

if menu == "Poultry Health Entry":
    st.header("Poultry Health Data Entry")
    with st.form("health_form"):
    
        body_weight = st.number_input("Body Weight (kg)", min_value=0.0)
        body_temp = st.number_input("Body Temperature (°C)", min_value=0.0)
        vaccines = st.text_input("Vaccination Records")
        symptoms = st.text_input("Symptoms")
        uploaded_image = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
        image_analysis = ""
        if uploaded_image is not None:
            # Display the uploaded image
            st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
            # Read image and encode as base64
            image_bytes = uploaded_image.read()
            image_base64 = base64.b64encode(image_bytes).decode("utf-8")
            prompt = "Describe what you see in this poultry health image."
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are an expert poultry farm assistant. Describe the content of the uploaded image for health analysis."},
                    {"role": "user", "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                    ]}
                ],
                max_tokens=256
            )
            image_analysis = response.choices[0].message.content
            st.info(f"Image Analysis: {image_analysis}")
        else:
            image_analysis = "No Photo"
        submitted = st.form_submit_button("Submit")
        
        if submitted:
            query_prompt = f"""
            INSERT INTO poultry_health_records 
            (body_weight, body_temperature, vaccination_records, symptoms, image_analysis, created_at)
            VALUES 
            ({body_weight}, {body_temp}, '{vaccines}', '{symptoms}', '{image_analysis}', '{datetime.now()}');
            """
            
            crew = Crew(
                agents=[sql_dev, data_insert_validator,alert_agent],
                tasks=[extract_data, validate_insert_data,alert_task],
                process=Process.sequential,
                verbose=True,
                memory=False,
                output_log_file="crew.log",
            )
            
            inputs = {"query": query_prompt}
            result = crew.kickoff(inputs=inputs)
            
            st.success("Record processed successfully!")
            st.write("Query Result:")
            st.code(result)
elif menu == "Poultry Health Entry 1":
    st.header("Poultry Health Data Entry 1")
    gen_case_id = generate_custom_id()
    with st.form("health_form 1"):
        case_id = st.text_input("Case ID", value=gen_case_id, disabled=True)
        body_weight = st.number_input("Body Weight (kg)", min_value=0.0)
        body_temp = st.number_input("Body Temperature (°C)", min_value=0.0)
        vaccines = st.text_input("Vaccination Records")
        symptoms = st.text_input("Symptoms")
        uploaded_image = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
        image_analysis = ""
        submitted = st.form_submit_button("Submit")
        if submitted:
            if uploaded_image is not None:
                st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
                image_bytes = uploaded_image.read()
                image_base64 = base64.b64encode(image_bytes).decode("utf-8")
                prompt = "Describe what you see in this poultry health image."
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "You are an expert poultry farm assistant. Describe the content of the uploaded image for health analysis."},
                        {"role": "user", "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                        ]}
                    ],
                    max_tokens=256
                )
                image_analysis = response.choices[0].message.content
                st.info(f"Image Analysis: {image_analysis}")
            else:
                image_analysis = "No Photo"



            # Collect data into JSON format
            record_json = {
                "case_id": {"value": case_id, "requirement": "mandatory"},
                "body_weight": {"value": body_weight, "requirement": "mandatory"},
                "body_temperature": {"value": body_temp, "requirement": "mandatory"},
                "vaccination_records": {"value": vaccines, "requirement": "mandatory"},
                "symptoms": {"value": symptoms, "requirement": "optional"},
                "image_analysis": {"value": image_analysis, "requirement": "optional"},
                "created_at": {"value": str(datetime.now()), "requirement": "mandatory"},
                "SQL_type": {"value": "Insert", "requirement": "mandatory"},
                "Database_table": {"value": "poultry_health_records", "requirement": "mandatory"}
            }
            
            # Validation example
            missing_fields = [k for k, v in record_json.items() if v["requirement"] == "mandatory" and not v["value"]]
            if missing_fields:
                st.error(f"Missing mandatory fields: {', '.join(missing_fields)}")
            else:
                st.success("All mandatory fields are present.")
                st.json(record_json)
            st.write("Collected JSON:")
            st.json(record_json)

            # query_prompt = f"""
            # INSERT INTO poultry_health_records 
            # (body_weight, body_temperature, vaccination_records, symptoms, image_analysis, created_at)
            # VALUES 
            # ({body_weight}, {body_temp}, '{vaccines}', '{symptoms}', '{image_analysis}', '{datetime.now()}');
            # """

            crew_input = {"prompt": json.dumps(record_json)}

            crew = Crew(
                # agents=[data_json_validator,alert_agent,sql_dev],
                # tasks=[validate_json_data,alert_task,store_data],
                agents=[data_json_validator,alert_agent, sql_dev],
                tasks=[validate_json_data,alert_json_task, store_data],
                process=Process.sequential,
                verbose=True,
                memory=False,
                output_log_file="crew.log",
            )
            inputs = {"query": crew_input}
            result = crew.kickoff(inputs=inputs)
            st.write("Query Result:")
            st.code(result)

elif menu == "Biosecurity Entry":
    st.header("Biosecurity Data Entry")
    with st.form("bio_form"):
        location = st.text_input("Location")
        violation = st.text_input("Violation")
        image_analysis = st.text_area("Image Analysis")
        submitted = st.form_submit_button("Submit")
        
        if submitted:
            query_prompt = f"""
            INSERT INTO biosecurity_records 
            (location, violation, image_analysis, created_at)
            VALUES 
            ('{location}', '{violation}', '{image_analysis}', '{datetime.now()}');
            """
            
            crew = Crew(
                agents=[sql_dev],
                tasks=[extract_data],
                process=Process.sequential,
                verbose=True,
                memory=False,
                output_log_file="crew.log",
            )
            
            inputs = {"query": query_prompt}
            result = crew.kickoff(inputs=inputs)
            
            st.success("Record processed successfully!")
            st.write("Query Result:")
            try:
                # Try to parse result as JSON
                if isinstance(result, str):
                    parsed_result = json.loads(result)
                else:
                    parsed_result = result
                st.code(json.dumps(parsed_result, indent=2))
            except Exception:
                # Fallback: just display the raw result
                st.code(result)
            st.success("Record added successfully!")
            st.write(f"Added record: Location={location}, Violation={violation}")

elif menu == "Database Query":
    st.header("Database Query")
    query = st.text_area("Enter your query (SQL or natural language)")
    
    if st.button("Execute Query"):
        # Initialize Crew with the query
        crew = Crew(
            agents=[sql_dev, data_analyst],
            tasks=[extract_data, analyze_data],
            process=Process.sequential,
            verbose=True,
            memory=False,
            output_log_file="crew.log",
        )
        
        # Execute the query through CrewAI
        inputs = {"query": query}
        result = crew.kickoff(inputs=inputs)
        
        st.write("Query Results:")
        st.code(result)
        st.success("Query executed successfully!")

elif menu == "Poultry Health Update":
    st.header("Poultry Health Data Update")
    # Initialize session state for record_found and form fields
    if "record_found" not in st.session_state:
        st.session_state.record_found = False
    if "update_fields" not in st.session_state:
        st.session_state.update_fields = {}

    with st.form("health_update_form"):
        # record_id = st.number_input("Enter Record ID to Update", min_value=1, step=1)
        # Connect to the database

        connection = psycopg2.connect(
                        user=USER,
                        password=PASSWORD,
                        host=HOST,
                        port=PORT,
                        dbname=DBNAME
                    )

        print("Connection successful!")
        # Create a cursor to execute SQL queries
        cursor = connection.cursor()
        cursor.execute("SELECT case_id FROM poultry_health_records")
        case_id_rows = cursor.fetchall()
        # Close the cursor and connection
        cursor.close()
        connection.close()
        print("Connection closed.")


        case_id_list = [row[0] for row in case_id_rows]
        if case_id_list:
            case_record_id = st.selectbox("Select Record ID to Update", case_id_list)
            st.write(f"You selected case ID: {case_record_id}")
        else:
            st.warning("No records found in poultry_health_records.")

        fetch_btn = st.form_submit_button("Fetch Record")

        if fetch_btn:
            # Fetch the record from the database

            #SQLite codes
            # conn = sqlite3.connect('poultry_health.db')
            # c = conn.cursor()
            # c.execute("SELECT body_weight, body_temperature, vaccination_records, symptoms, image_analysis FROM poultry_health_records WHERE id=?", (record_id,))
            # row = c.fetchone()
            # conn.close()
            # if row:
            #     st.session_state.record_found = True
            #     st.session_state.update_fields = {
            #         "body_weight": row[0],
            #         "body_temp": row[1],
            #         "vaccines": row[2],
            #         "symptoms": row[3],
            #         "image_analysis": row[4]
            #     }
            #     st.success("Record found. You can now update the fields below.")
            
            #Supabase codes
            connection = psycopg2.connect(
                        user=USER,
                        password=PASSWORD,
                        host=HOST,
                        port=PORT,
                        dbname=DBNAME
                    )

            print("Connection successful!")
            # Create a cursor to execute SQL queries
            cursor = connection.cursor()
    
            cursor.execute(
                "SELECT body_weight, body_temperature, vaccination_records, symptoms, image_analysis FROM poultry_health_records WHERE case_id = %s",
                (case_record_id,))
            row = cursor.fetchone()
            # Close the cursor and connection
            cursor.close()
            connection.close()
            print("Connection closed.")
            if row:
                st.session_state.record_found = True
                st.session_state.update_fields = {
                    "body_weight": row[0],
                    "body_temp": row[1],
                    "vaccines": row[2],
                    "symptoms": row[3],
                    "image_analysis": row[4]
                }
                st.success("Record found. You can now update the fields below.")          
            else:
                st.session_state.record_found = False
                st.session_state.update_fields = {}
                st.error("Record not found. Please check the ID.")

        # Only display the update form if a record was found
        if st.session_state.record_found:
            update_fields = st.session_state.update_fields
            case_id = st.text_input("Case ID", value=case_record_id, disabled=True)
            new_body_weight = st.number_input("Body Weight (kg)", min_value=0.0, value=update_fields.get("body_weight", 0.0), key="update_weight")
            new_body_temp = st.number_input("Body Temperature (°C)", min_value=0.0, value=update_fields.get("body_temp", 0.0), key="update_temp")
            new_vaccines = st.text_input("Vaccination Records", value=update_fields.get("vaccines", ""), key="update_vaccines")
            new_symptoms = st.text_input("Symptoms", value=update_fields.get("symptoms", ""), key="update_symptoms")
            new_image_analysis = st.text_input("Image Analysis", value=update_fields.get("image_analysis", ""), key="update_image_analysis")
            uploaded_image = st.file_uploader("Upload New Image (optional)", type=["jpg", "jpeg", "png"], key="update_image")
            new_image_analysis = update_fields.get("image_analysis", "")

            #Submit button for update
            update_btn = st.form_submit_button("Update Record")
            if update_btn:
                #Call VLLM to get the new image analysis 
                if uploaded_image is not None:
                    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
                    image_bytes = uploaded_image.read()
                    image_base64 = base64.b64encode(image_bytes).decode("utf-8")
                    prompt = "Describe what you see in this poultry health image."
                    response = client.chat.completions.create(
                        model="gpt-4o",
                        messages=[
                            {"role": "system", "content": "You are an expert poultry farm assistant. Describe the content of the uploaded image for health analysis."},
                            {"role": "user", "content": [
                                {"type": "text", "text": prompt},
                                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                            ]}
                        ],
                        max_tokens=256
                    )
                    new_image_analysis = response.choices[0].message.content
                    st.info(f"Image Analysis: {new_image_analysis}")
                else:
                    new_image_analysis = "No Photo"
                
                # Collect data into JSON format
                record_json = {
                    "case_id": {"value": case_id, "requirement": "mandatory"},
                    "body_weight": {"value": new_body_weight, "requirement": "mandatory"},
                    "body_temperature": {"value": new_body_temp, "requirement": "mandatory"},
                    "vaccination_records": {"value": new_vaccines, "requirement": "mandatory"},
                    "symptoms": {"value": new_symptoms, "requirement": "optional"},
                    "image_analysis": {"value": new_image_analysis, "requirement": "optional"},
                    "created_at": {"value": str(datetime.now()), "requirement": "mandatory"},
                    "SQL_type": {"value": "Update", "requirement": "mandatory"},
                    "Database_table": {"value": "poultry_health_records", "requirement": "mandatory"}
                }
                
                # Validation example
                missing_fields = [k for k, v in record_json.items() if v["requirement"] == "mandatory" and not v["value"]]
                if missing_fields:
                    st.error(f"Missing mandatory fields: {', '.join(missing_fields)}")
                else:
                    st.success("All mandatory fields are present.")
                    st.json(record_json)
                st.write("Collected JSON:")
                st.json(record_json)

                crew_input = {"prompt": json.dumps(record_json)}

                # Use Crew agent to construct and execute the UPDATE query
                # query_prompt = f"""
                # UPDATE poultry_health_records
                # SET body_weight={new_body_weight},
                #     body_temperature={new_body_temp},
                #     vaccination_records='{new_vaccines}',
                #     symptoms='{new_symptoms}',
                #     image_analysis='{new_image_analysis}'
                # WHERE case_id={case_record_id};
                # """


                crew = Crew(
                    # agents=[sql_dev, data_insert_validator],
                    # tasks=[extract_data, validate_insert_data],
                    agents=[data_json_validator,alert_agent, sql_dev],
                    tasks=[validate_json_data,alert_json_task, update_data],
                    process=Process.sequential,
                    verbose=True,
                    memory=False,
                    output_log_file="crew.log",
                )
                inputs = {"query": crew_input}
                result = crew.kickoff(inputs=inputs)
                st.success("Record updated successfully!")
                st.write("Query Result:")
                st.code(result)





