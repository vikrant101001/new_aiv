


from flask import Flask, request, jsonify
from geopy.geocoders import MapBox
from geopy.distance import geodesic
from pathlib import Path
import faiss
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
#from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
import pickle
import os
import json
from langchain import LLMChain
#from langchain_community.llms import OpenAIChat
from langchain.chat_models import ChatOpenAI
from langchain.prompts import Prompt
import time
from flask_cors import CORS

import re
import config 
from PyPDF2 import PdfReader
import helper
import database
import random


from sqlalchemy import create_engine, text
from sqlalchemy import Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker


from collections import defaultdict
import threading


from crewai import Agent, Crew
from crewai import Task
from langchain_community.tools import DuckDuckGoSearchRun
import os
from langchain_anthropic import ChatAnthropic


from anthropic import Anthropic
import crews
import trainer



app = Flask(__name__)
cors = CORS(app)


openai_api_key = config.OPENAI_API_KEY
mapbox_api_key = config.MAPBOX_API_KEY
API_SECRET = config.API_SECRET


geocoder = MapBox(api_key=mapbox_api_key)


last_api_call_time = 0
history = []
llmChain = None

# inserting data into ai advocate history
username = ""
password = ""
hostname = ""
database_name = ""


# Construct the connection URL
db_url = f"postgresql://{username}:{password}@{hostname}/{database_name}"

# Define the SQLAlchemy model
Base = declarative_base()


class advocatehistory(Base):
  __tablename__ = 'aiadvocatehistory'  # Adjust table name as needed

  # Add a dummy primary key
  id = Column(Integer, primary_key=True, autoincrement=True)

  user_question = Column(String)
  bot_answer = Column(String)
  caregiver_id = Column(String)
  careteam_id = Column(String)




searched = 0
previous_response = ""

def clean_response(response):
    # Define a regex pattern to match valid characters
    pattern = r'[^a-zA-Z0-9,. ]'
    
    # Use re.sub to replace invalid characters with an empty string
    cleaned_response = re.sub(pattern, '', response)
    
    return cleaned_response


def reset_history(careteam_id):
    careteam_histories[careteam_id] = []

@app.route("/", methods=["GET"])
def index():
  return "API Online"


# Dictionary to store unique histories for each caregiver_id
careteam_histories = {}

previous_response = {}

searched = {}
last_api_call_times = {}
agent = 'undefined'



@app.route('/')
def backend_start():
    return 'Backend is up and running!'

@app.route("/chat_text", methods=["POST"])
def ask():
    global last_api_call_time
    global llmChain
    global count1
    global agent

    username = ""
    password = ""
    hostname = ""
    database_name = ""
    db_connection_url = f"postgresql://{username}:{password}@{hostname}/{database_name}"

    api_secret_from_frontend = request.headers.get('X-API-SECRET')
    if api_secret_from_frontend != API_SECRET:
        return jsonify({'error': 'Unauthorized access'}), 401

    careteam_id = request.headers.get('careteamid')
    caregiver_id = request.headers.get('userid')

    if careteam_id == "not implied" or caregiver_id == "not implied":
        return jsonify({'message': "Caregiver or careteam id not implied"})

    try:
        reqData = request.get_json()
        user_question = reqData['question']
        user_address = request.headers.get('userprimaddress')
        print(f"All Headers: {request.headers}")
        
        metadata = reqData['metadata']
        agent_recieved = reqData['agent']

        current_time = time.time()
        last_api_call_time_for_caregiver = last_api_call_times.get(careteam_id, 0)

        careteam_history = careteam_histories.setdefault(careteam_id, [])


        if current_time - last_api_call_time_for_caregiver > 600:
            reset_history(careteam_id)

            # """
        # History = "We are refering to the image () with the description of the image being ()"
        # question = "Refer to the image ()"
        # """

            last_api_call_times[careteam_id] = current_time
            # Only confirm address if the question is related to a search
            user_location = trainer.get_coordinates(user_address)
            print("1")

            count1 = trainer.train(user_location)  # Train based on user location for the first call of a session
            print(count1)


        if len(metadata) > 0:
            careteam_history.append(f"Human: {user_question}")
            description_of_docs, status_code = helper.text_extraction(metadata, careteam_id)
            if status_code == 200:
                print(description_of_docs)
                answer = crews.new_document_analyser(description_of_docs, user_question)
                careteam_history.append(f"Bot: Description of Image: {description_of_docs} , Answer for User_Question: {answer}")
                agent = "document"
                return jsonify({"answer": answer, "current agent": agent,"success": True}), 200
            else:
                return jsonify({"answer": "Sorry, we aren't able to help you right now, try again after some time.", "current agent": agent, "success": False}), 400

        if not llmChain:
            # Initialize llmChain if it's not initialized yet
            with open("training/master.txt", "r") as f:
                promptTemplate = f.read()
            print("i read master text")

            prompt = Prompt(template=promptTemplate, input_variables=["history", "context", "question"])
            llmChain = LLMChain(prompt=prompt, llm=ChatOpenAI(temperature=0.2,
                                                             model_name="gpt-4o", max_tokens = 150,

                                                             openai_api_key=openai_api_key))
            print("i reached LLmchain definition")
            

        current_agent = crews.basedecidercrew(careteam_history, agent_recieved, user_question)



        # if agent == 'base':
        #     #The base decider agent decides when and which agent to give the task to
        #     new_agent = crews.basedecidercrew(careteam_history)
        #     print(new_agent)
        #     if 'Base' not in new_agent:
        #         agent = new_agent
        #     else:
        #         agent = "base"
        careteam_history.append(f"Human: {user_question}")
        if 'MEDICATOR' in current_agent:
            print("i reached medicator agent")
            # agent = 'Medicator'
            response = crews.medicator_crew(careteam_history, user_question)
            # if response == 'base':
            #     agent = 'base'
            careteam_history.append(f"Bot: {response}")
            agent = current_agent
            return jsonify({"answer": response, "current agent": agent,"success": True}), 200

        if 'DOCUMENT' in current_agent:
            print("i reached document agent")
            # agent = 'Medicator'
            response = crews.document_processor_crew(careteam_history, user_question)
            # if response == 'base':
            #     agent = 'base'
            careteam_history.append(f"Bot: {response}")
            agent = current_agent
            return jsonify({"answer": response, "current agent": agent, "success": True}), 200

        response = llmChain.predict(question=user_question, context="\n\n".join(careteam_history), history=careteam_history)
        careteam_history.append(f"Bot: {response}")
        return jsonify({"answer": response, "current agent": current_agent,"success": True}), 200


        
        # Adjust insert_conversation to handle caregiver-specific history
        #database.insert_conversation(user_question, response, careteam_id, caregiver_id)
    except Exception as e:
        return jsonify({"answer": None, "success": False, "message": str(e)}), 400




@app.route("/visitai_summary", methods=["POST"])
def visitai_summary():
    try:
        payload = request.get_json()
    except Exception as e:
        return jsonify({"answer": "Sorry, we aren't able to help you right now, try again after some time.", "message": str(e)}), 400

if __name__ == '__main__':
  app.run(host='0.0.0.0', port=3000)

