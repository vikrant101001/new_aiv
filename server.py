


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




app = Flask(__name__)
cors = CORS(app)


os.environ["OPENAI_API_KEY"] = ''
os.environ["MAPBOX_API_KEY"] = ''

os.environ["API_SECRET"] = 'my secret'

openai_api_key = os.environ["OPENAI_API_KEY"]
mapbox_api_key = os.environ["MAPBOX_API_KEY"]
geocoder = MapBox(api_key=mapbox_api_key)

API_SECRET = os.environ["API_SECRET"]

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


def insert_conversation(user_question, bot_answer, careteam_id, caregiver_id):
  try:
    # Create a SQLAlchemy engine and session
    engine = create_engine(db_url)
    Session = sessionmaker(bind=engine)
    session = Session()

    # Create a Conversation object
    conversation = advocatehistory(user_question=user_question,
                                   bot_answer=bot_answer,
                                   careteam_id=careteam_id,
                                   caregiver_id=caregiver_id)

    # Add the Conversation object to the session and commit the transaction
    session.add(conversation)
    session.commit()

    # Close the session
    session.close()

  except Exception as e:
    # Handle exceptions (e.g., database errors)
    print(f"Error inserting conversation: {e}")


# AI CARE MANAGER insert data: weekly checkin


class checkinhistory(Base):
  __tablename__ = 'acmcheckinhistory'  # Adjust table name as needed

  # Add a dummy primary key
  id = Column(Integer, primary_key=True, autoincrement=True)

  checkin_question = Column(String)
  user_answer = Column(String)
  caregiver_id = Column(String)
  careteam_id = Column(String)


def insert_checkin(checkin_question, user_answer, caregiver_id, careteam_id):
  try:
    # Create a SQLAlchemy engine and session
    engine = create_engine(db_url)
    Session = sessionmaker(bind=engine)
    session = Session()

    # Create a Conversation object
    conversation = checkinhistory(checkin_question=checkin_question,
                                  user_answer=user_answer,
                                  caregiver_id=caregiver_id,
                                  careteam_id=careteam_id)

    # Add the Conversation object to the session and commit the transaction
    session.add(conversation)
    session.commit()

    # Close the session
    session.close()

  except Exception as e:
    # Handle exceptions (e.g., database errors)
    print(f"Error inserting conversation: {e}")


# AI CARE MANAGER insert data: Functional assessmeent checkin


class fahistory(Base):
  __tablename__ = 'acmfahistory'  # Adjust table name as needed

  # Add a dummy primary key
  id = Column(Integer, primary_key=True, autoincrement=True)

  fa_question = Column(String)
  fa_answer = Column(String)
  fa_title = Column(String)
  fa_score = Column(Integer)
  caregiver_id = Column(String)
  careteam_id = Column(String)


def insert_fa(fa_question, fa_answer, fa_title, fa_score, caregiver_id,
              careteam_id):
  try:
    # Create a SQLAlchemy engine and session
    engine = create_engine(db_url)
    Session = sessionmaker(bind=engine)
    session = Session()

    # Create a Conversation object
    conversation = fahistory(fa_question=fa_question,
                             fa_answer=fa_answer,
                             fa_title=fa_title,
                             fa_score=fa_score,
                             caregiver_id=caregiver_id,
                             careteam_id=careteam_id)

    # Add the Conversation object to the session and commit the transaction
    session.add(conversation)
    session.commit()

    # Close the session
    session.close()

  except Exception as e:
    # Handle exceptions (e.g., database errors)
    print(f"Error inserting conversation: {e}")



def reset_history(careteam_id):
    careteam_histories[careteam_id] = []


def get_coordinates(address):
    mapbox_api_key = os.environ["mapbox_api_key"]
    geocoder = MapBox(api_key=mapbox_api_key)

    # Convert the address to the central zipcode
    location = geocoder.geocode(address)
    
    if location:
        latitude, longitude = location.latitude, location.longitude
        print(f"Confirmed:\nAddress: {address}\nCoordinates: {latitude}, {longitude}")
        return latitude, longitude
    else:
        raise ValueError("Could not retrieve location information from the address.")


def calculate_distance(coord1, coord2):
    return geodesic(coord1, coord2).miles

def train(user_location):
    print("i reached train user location")
    count = 0
    print(count)
    try:
        #os.remove("faiss.pkl")
        #os.remove("training.index")
        print("Removed existing faiss.pkl and training.index")
    except FileNotFoundError:
        pass 


    # Check there is data fetched from the database
    training_data_folders = list(Path("training/facts/").glob("**/latitude*,longitude*"))

    # Check there is data in the trainingData folder
    if len(training_data_folders) < 1:
        print("The folder training/facts should be populated with at least one subfolder.")
        return

    
    latitude, longitude = user_location
    user_coordinates = (latitude, longitude)

    data = []
    for folder in training_data_folders:
        folder_coordinates = folder.name.replace('latitude', '').replace('longitude', '').split(',')
        folder_latitude, folder_longitude = map(float, folder_coordinates)

        folder_coords = (folder_latitude, folder_longitude)
        distance = calculate_distance(user_coordinates, folder_coords)
        print(f" the distance between {user_coordinates} and {folder.name} is {distance}.")

        if distance < 100:
            count = count +1
            print(f"Added {folder.name}'s contents to training data.")
            for json_file in folder.glob("*.json"):
                with open(json_file) as f:
                    data.extend(json.load(f))
                    print(f"  Added {json_file.name} to training data.")
 
    if count == 0:
       print("No relevant data found within 50 miles.")
       print(count)
       return count
        

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=0)

    docs = []
    for entry in data:
    	address = entry.get('address', '')
    	if address is not None:
    		print(f"Address to split: {address}")
    		docs.extend(text_splitter.split_text(address))

    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    store = FAISS.from_texts(docs, embeddings)

    #faiss.write_index(store.index, "training.index")
    #store.index = None
    #print(dir(store))
    print(count)
    #with open("faiss.pkl", "wb") as f:
    #    pickle.dump(store, f)
    
    return count

searched = 0
# ...
previous_response = ""
# ...

def clean_response(response):
    # Define a regex pattern to match valid characters
    pattern = r'[^a-zA-Z0-9,. ]'
    
    # Use re.sub to replace invalid characters with an empty string
    cleaned_response = re.sub(pattern, '', response)
    
    return cleaned_response


@app.route("/", methods=["GET"])
def index():
  return "API Online"

#last_api_call_time = time.time()

# Dictionary to store unique histories for each caregiver_id
careteam_histories = {}

previous_response = {}

searched = {}
last_api_call_times = {}
agent = 'undefined'
os.environ["ANTHROPIC_API_KEY"] = ""

medicatorllm = ChatAnthropic(temperature=0, model_name="claude-3-haiku-20240307")


Medicator = Agent(
    role='Expert regarding Medicines',
    goal='Answer Question regarding Medicines',
    backstory="""As a Medicator, you specialize in pharmaceuticals and healthcare. Your role is to provide accurate information on medications, leveraging open-source data to ensure precision. Your expertise is crucial in guiding users with clarity and staying updated on medical advancements to offer valuable insights for informed decisions. You will use open source to find information regarding anything to do with medicines and answer questions properly. U also have knowledge regarding finding appropriate resources. You will use your own knowledge to find resources and not tell the user to use search tools and reply in proper format. Note: it is very very crucial that the datas are correct""",
    verbose=True,
    allow_delegation=False,
    llm=medicatorllm,
)


def basedecidercrew(careteam_history):
    # Instantiate your crew with a sequential process
   
    client = Anthropic()
    message = client.messages.create(
        model = 'claude-3-haiku-20240307',
        max_tokens=1024,
        messages=[
        {"role":"user",
         "content":f"Analyze the following chat history between the user and the AI://starts here {careteam_history}  ends here//  . The chat history includes the user's latest question. Based on the content of the chat history, decide whether to transfer the conversation to Agent 1 (Medicator) or keep it with the base agent (Agent 0) base agent is really good in resource finding also. so for searching for resources dont switch to the other agents.If the conversation should be transferred to the Medicator, reply with 'Agent 1. Medicator'. If the conversation should remain with the base agent, reply with 'Agent 0. Base'. Provide only the exact response without any additional explanations."         },
         ]
    ).content[0].text

    print(message)
    return(message)


def medicator_crew(careteam_history, current_question):
    medicator1 = Task(
        description=f"""You (medicator) has been assigned the task of answering the user's queries. You will first decide if the conversation should still stay with you or if it should go to another agent. You can check this by considering the fact that you only deal with medical/medicinal based queries. take a look at the entire conversation which includes the latest question too: {careteam_history}.  Only answer with Yes or No, and dont add anything else""",
        agent=Medicator
    )

    client = Anthropic()
    message = client.messages.create(
        model = 'claude-3-haiku-20240307',
        max_tokens=1024,
        messages=[
        {"role":"user",
         "content":f"Analyze the following chat history between the user and the AI://starts here {careteam_history}  ends here//. The chat history includes the user's latest question. Based on the content of the latest question, decide whether to keep the conversation with Agent 1 (Medicator) or send it back to the base agent (Agent 0). If the conversation should be transferred to the Medicator, reply with 'Agent 1. Medicator'. If the conversation should remain with the base agent, reply with 'Agent 0. Base'. Provide only the exact response without any additional explanations."         },
         ]
    ).content[0].text

    print(message)

    if 'Medicator' in message:

        medicator2 = Task(
            description=f"""answer their medication related question quickly but efficiently and also in short no more than required . This is the past conversations (conversation starts here ----{careteam_history}-----end) and the lastest question : {current_question}. and answer it as fast as possible dont take time at all. The answer doesnt need to be the best. Also very important note: the answer MUST be short, no more than 2 lines""",
            agent=Medicator
        )

        crew = Crew(
            agents=[Medicator],
            tasks=[medicator2],
            verbose=1, 
        )

    # Get your crew to work!
        result = crew.kickoff()
    else:
        result = 'base'
        
    print("######################")
    print(result)

    return result

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

        current_time = time.time()
        last_api_call_time_for_caregiver = last_api_call_times.get(careteam_id, 0)
        if current_time - last_api_call_time_for_caregiver > 600:
            agent = 'base'
            reset_history(careteam_id)
            last_api_call_times[careteam_id] = current_time


            searched[careteam_id] = 0

            # Only confirm address if the question is related to a search
            user_location = get_coordinates(user_address)
            count1 = train(user_location)  # Train based on user location for the first call of a session
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

        
        if agent == 'base':
            #The base decider agent decides when and which agent to give the task to
            new_agent = basedecidercrew(careteam_history)
            print(new_agent)
            if 'Base' not in new_agent:
                agent = new_agent
            else:
                agent = "base"

        if 'Medicator' in agent:
            print("i reached medicator agent")
            agent = 'Medicator'
            response = medicator_crew(careteam_history, user_question)
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
                user_location = get_coordinates(user_address)
                count2 = train(user_location)  # Train based on the new user location
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
        insert_conversation(user_question, response, careteam_id, caregiver_id)

        #return jsonify({"answer": response, "previous_response": previous_response.get(careteam_id, ""), "searched": searched.get(careteam_id, 0), "current agent": agent ,"success": True})
        if 'finding' or 'Miami' in user_question:
            return jsonify({"answer": response, "current agent": 'resource finder' ,"success": True})

        
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
            reset_history(careteam_id)
            last_api_call_times[careteam_id] = current_time


            searched[careteam_id] = 0

            # Only confirm address if the question is related to a search
            user_location = get_coordinates(user_address)
            count1 = train(user_location)  # Train based on user location for the first call of a session
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
            user_location = get_coordinates(user_address)
            count2 = train(user_location)  # Train based on the new user location
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
        insert_conversation(user_question, response, careteam_id, caregiver_id)

        
        response = clean_response(response)
        
        return jsonify({"answer": response, "previous_response": previous_response.get(careteam_id, ""), "searched": searched.get(careteam_id, 0), "success": True})
    except Exception as e:
        return jsonify({"answer": None, "success": False, "message": str(e)}), 400





if __name__ == '__main__':
  app.run(host='0.0.0.0', port=3000)

