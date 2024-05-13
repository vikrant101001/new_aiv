


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


@app.route("/", methods=["GET"])
def index():
  return "API Online"


# Dictionary to store unique histories for each caregiver_id
careteam_histories = {}

previous_response = {}

searched = {}
last_api_call_times = {}
agent = 'undefined'





@app.route("/", methods=["POST"])
def ask():
    global last_api_call_time
    global llmChain
    global count1
    global agent

    username = "aiassistantevvaadmin"
    password = "EvvaAi10$"
    hostname = "aiassistantdatabase.postgres.database.azure.com"
    database_name = "aidatabasecombined"
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

        current_time = time.time()
        last_api_call_time_for_caregiver = last_api_call_times.get(careteam_id, 0)
        if current_time - last_api_call_time_for_caregiver > 600:
            
            
        # """
        # History = "We are refering to the image () with the description of the image being ()"
        # question = "Refer to the image ()"
        # """

            #if len(metadata) > 0:
            #    description_of_docs = helper.text_extraction(metadata)
            #    print(description_of_docs)


            #agent = 'base'
            #careteam_histories[careteam_id] = []
            #last_api_call_times[careteam_id] = current_time


            #searched[careteam_id] = 0

            # Only confirm address if the question is related to a search
            user_location = trainer.get_coordinates(user_address)
            count1 = trainer.train(user_location)  # Train based on user location for the first call of a session
            print(count1)
        


        print(agent)
        if not llmChain:
            # Initialize llmChain if it's not initialized yet
            with open("training/master.txt", "r") as f:
                promptTemplate = f.read()
            print("i read master text")

            prompt = Prompt(template=promptTemplate, input_variables=["history", "context", "question"])
            llmChain = LLMChain(prompt=prompt, llm=ChatOpenAI(temperature=0.5,
                                                             model_name="gpt-4-1106-preview",
                                                             openai_api_key=openai_api_key))
            print("i reached LLmchain definition")
            
        careteam_history = careteam_histories.setdefault(careteam_id, [])
        careteam_history.append(f"Human: {user_question}")


        if len(metadata) > 0:
            description_of_docs = helper.text_extraction(metadata)
            print(description_of_docs)
            agent = 'Image Processor'
            response = crews.image_processor_crew(description_of_docs, user_question)
            if response == 'base':
                agent = 'base'

        
        if agent == 'base':
            #The base decider agent decides when and which agent to give the task to
            new_agent = crews.basedecidercrew(careteam_history)
            print(new_agent)
            if 'Base' not in new_agent:
                agent = new_agent
            else:
                agent = "base"

        if 'Medicator' in agent:
            print("i reached medicator agent")
            agent = 'Medicator'
            response = crews.medicator_crew(careteam_history, user_question)
            if response == 'base':
                agent = 'base'

        if agent == 'base':
            print("i reached base agent")
            # Only confirm the user's address for search-related questions
            search_keywords = ["near my location"]

            

            if any(keyword in user_question.lower() for keyword in search_keywords):
                if searched.get(careteam_id, 0) == 0:
                    searched[careteam_id] = searched.get(careteam_id, 0) + 1
                    print(searched[careteam_id])
                    confirm_message = f"Do you want me to search near\n{user_address}\n\nReply with 'yes' or 'no'."
                    previous_response[careteam_id] = confirm_message
                    response = confirm_message
                    searched[careteam_id] = searched.get(careteam_id, 0) + 1
                    print(searched[careteam_id])
                else:



                    response = llmChain.predict(question=user_question, context="\n\n".join(careteam_history), history=careteam_history)
            elif previous_response.get(careteam_id, "").startswith("Do you want me to search near") and "yes" in user_question.lower():
                # Continue with the user's provided address
                if count1 < 1:
                    response = "I am sorry! 🙁 I couldn’t find any suitable results within 100 miles. Evva is only available in limited geographies. Please contact Team Evva at info@evva360.com to learn more about when your region may be next. Would you like me to search near a different location?.. \n Please Reply with 'yes' or 'no'"
                    previous_response[careteam_id] = "I am sorry! 🙁 I couldn’t find any suitable results within 100 miles. Evva is only available in limited geographies. Please contact Team Evva at info@evva360.com to learn more about when your region may be next."
                else:
                    response = llmChain.predict(question=user_question, context="\n\n".join(careteam_history), history=careteam_history)
            elif previous_response.get(careteam_id, "").startswith("Do you want me to search near") and "no" in user_question.lower():
                # Ask for a new location
                previous_response[careteam_id] = "Please enter the new location where you want to search"
                response = previous_response[careteam_id]
            elif previous_response.get(careteam_id, "").startswith("Do you want me to search near") and "yes" not in user_question.lower() and "no" not in user_question.lower():
                response = "Please include yes or no in your answer"

            elif previous_response.get(careteam_id, "").startswith("Please enter the new location where you want to search"):
                user_address = user_question
                user_location = trainer.get_coordinates(user_address)
                count2 = trainer.train(user_location)  # Train based on the new user location
                if count2 < 1:
                    response = "I am sorry! 🙁 I couldn’t find any suitable results within 100 miles. Evva is only available in limited geographies. Please contact Team Evva at info@evva360.com to learn more about when your region may be next. Would you like me to search near a different location?.. \n Please Reply with 'yes' or 'no'"
                    previous_response[careteam_id] = "I am sorry! 🙁 I couldn’t find any suitable results within 100 miles. Evva is only available in limited geographies. Please contact Team Evva at info@evva360.com to learn more about when your region may be next."
                else:
                    previous_response[careteam_id] = ""
                    response = llmChain.predict(question=user_question, context="\n\n".join(careteam_history), history=careteam_history)
            elif previous_response.get(careteam_id, "").startswith("I am sorry! 🙁 I couldn’t find ") and "no" in user_question.lower():
                response = llmChain.predict(question=user_question, context="\n\n".join(careteam_history), history=careteam_history)
                previous_response[careteam_id] = ""
            elif previous_response.get(careteam_id, "").startswith("I am sorry! 🙁 I couldn’t find ") and "yes" in user_question.lower():
                previous_response[careteam_id] = "Please enter the new location where you want to search"
                response = previous_response[careteam_id]
            elif previous_response.get(careteam_id, "").startswith("I am sorry! 🙁 I couldn’t find ") and "yes" not in user_question.lower() and "no" not in user_question.lower():
                response = "Please include yes or no in your answer"
            else:
                # Continue with the user's question for non-search queries
                response = llmChain.predict(question=user_question, context="\n\n".join(careteam_history), history=careteam_history)



        careteam_history.append(f"Bot: {response}")
        
        # Adjust insert_conversation to handle caregiver-specific history
        database.insert_conversation(user_question, response, careteam_id, caregiver_id)



        
        return jsonify({"answer": response, "current agent": agent ,"success": True})
    except Exception as e:
        return jsonify({"answer": None, "success": False, "message": str(e)}), 400




@app.route("/voiceadvocate", methods=["POST"])
def askvoice():
    global last_api_call_time
    global llmChain
    global count1

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

        current_time = time.time()
        last_api_call_time_for_caregiver = last_api_call_times.get(careteam_id, 0)
        if current_time - last_api_call_time_for_caregiver > 600:
            trainer.reset_history(careteam_id)
            last_api_call_times[careteam_id] = current_time


            searched[careteam_id] = 0

            # Only confirm address if the question is related to a search
            user_location = trainer.get_coordinates(user_address)
            count1 = trainer.train(user_location)  # Train based on user location for the first call of a session
            print(count1)
            

        if not llmChain:
            # Initialize llmChain if it's not initialized yet
            with open("training/mastervoice.txt", "r") as f:
                promptTemplate = f.read()

            prompt = Prompt(template=promptTemplate, input_variables=["history", "context", "question"])
            llmChain = LLMChain(prompt=prompt, llm=OpenAIChat(temperature=0.5,
                                                             model_name="gpt-4-1106-preview",
                                                             openai_api_key=openai_api_key))

        # Only confirm the user's address for search-related questions
        search_keywords = ["near my location"]

        careteam_history = careteam_histories.setdefault(careteam_id, [])

        if any(keyword in user_question.lower() for keyword in search_keywords):
            if searched.get(careteam_id, 0) == 0:
                searched[careteam_id] = searched.get(careteam_id, 0) + 1
                print(searched[careteam_id])
                confirm_message = f"Do you want me to search near\n{user_address}\n\nReply with 'yes' or 'no'."
                previous_response[careteam_id] = confirm_message
                response = confirm_message
                searched[careteam_id] = searched.get(careteam_id, 0) + 1
                print(searched[careteam_id])
            else:
                response = llmChain.predict(question=user_question, context="\n\n".join(careteam_history), history=careteam_history)
        elif previous_response.get(careteam_id, "").startswith("Do you want me to search near") and "yes" in user_question.lower():
            # Continue with the user's provided address
            if count1 < 1:
                response = "I am sorry! 🙁 I couldn’t find any suitable results within 100 miles. Evva is only available in limited geographies. Please contact Team Evva at info@evva360.com to learn more about when your region may be next. Would you like me to search near a different location?.. \n Please Reply with 'yes' or 'no'"
                previous_response[careteam_id] = "I am sorry! 🙁 I couldn’t find any suitable results within 100 miles. Evva is only available in limited geographies. Please contact Team Evva at info@evva360.com to learn more about when your region may be next."
            else:
                response = llmChain.predict(question=user_question, context="\n\n".join(careteam_history), history=careteam_history)
        elif previous_response.get(careteam_id, "").startswith("Do you want me to search near") and "no" in user_question.lower():
            # Ask for a new location
            previous_response[careteam_id] = "Please enter the new location where you want to search"
            response = previous_response[careteam_id]
        elif previous_response.get(careteam_id, "").startswith("Do you want me to search near") and "yes" not in user_question.lower() and "no" not in user_question.lower():
            response = "Please include yes or no in your answer"

        elif previous_response.get(careteam_id, "").startswith("Please enter the new location where you want to search"):
            user_address = user_question
            user_location = trainer.get_coordinates(user_address)
            count2 = trainer.train(user_location)  # Train based on the new user location
            if count2 < 1:
                response = "I am sorry! 🙁 I couldn’t find any suitable results within 100 miles. Evva is only available in limited geographies. Please contact Team Evva at info@evva360.com to learn more about when your region may be next. Would you like me to search near a different location?.. \n Please Reply with 'yes' or 'no'"
                previous_response[careteam_id] = "I am sorry! 🙁 I couldn’t find any suitable results within 100 miles. Evva is only available in limited geographies. Please contact Team Evva at info@evva360.com to learn more about when your region may be next."
            else:
                previous_response[careteam_id] = ""
                response = llmChain.predict(question=user_question, context="\n\n".join(careteam_history), history=careteam_history)
        elif previous_response.get(careteam_id, "").startswith("I am sorry! 🙁 I couldn’t find ") and "no" in user_question.lower():
            response = llmChain.predict(question=user_question, context="\n\n".join(careteam_history), history=careteam_history)
            previous_response[careteam_id] = ""
        elif previous_response.get(careteam_id, "").startswith("I am sorry! 🙁 I couldn’t find ") and "yes" in user_question.lower():
            previous_response[careteam_id] = "Please enter the new location where you want to search"
            response = previous_response[careteam_id]
        elif previous_response.get(careteam_id, "").startswith("I am sorry! 🙁 I couldn’t find ") and "yes" not in user_question.lower() and "no" not in user_question.lower():
            response = "Please include yes or no in your answer"
        else:
            # Continue with the user's question for non-search queries
            response = llmChain.predict(question=user_question, context="\n\n".join(careteam_history), history=careteam_history)

        careteam_history.append(f"Bot: {response}")
        careteam_history.append(f"Human: {user_question}")
        # Adjust insert_conversation to handle caregiver-specific history
        database.insert_conversation(user_question, response, careteam_id, caregiver_id)

        
        response = clean_response(response)
        
        return jsonify({"answer": response, "previous_response": previous_response.get(careteam_id, ""), "searched": searched.get(careteam_id, 0), "success": True})
    except Exception as e:
        return jsonify({"answer": None, "success": False, "message": str(e)}), 400





if __name__ == '__main__':
  app.run(host='0.0.0.0', port=3000)

