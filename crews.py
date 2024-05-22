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
        model='claude-3-haiku-20240307',
        max_tokens=1024,
        messages=[
            {"role": "user",
             "content": f"Analyze the following chat history between the user and the AI://starts here {careteam_history}  ends here//  . The chat history includes the user's latest question. Based on the content of the chat history, decide whether to transfer the conversation to Agent 1 (Medicator) or keep it with the base agent (Agent 0) base agent is really good in resource finding also. so for searching for resources dont switch to the other agents.If the conversation should be transferred to the Medicator, reply with 'Agent 1. Medicator'. If the conversation should remain with the base agent, reply with 'Agent 0. Base'. Provide only the exact response without any additional explanations."},
        ]
    ).content[0].text

    print(message)
    return (message)


def medicator_crew(careteam_history, current_question):

    client = Anthropic()
    message = client.messages.create(
        model='claude-3-haiku-20240307',
        max_tokens=1024,
        messages=[
            {"role": "user",
             "content": f"Analyze the following chat history between the user and the AI://starts here {careteam_history}  ends here//. The chat history includes the user's latest question. Based on the content of the latest question, decide whether to keep the conversation with Agent 1 (Medicator) or send it back to the base agent (Agent 0). If the conversation should be transferred to the Medicator, reply with 'Agent 1. Medicator'. If the conversation should remain with the base agent, reply with 'Agent 0. Base'. Provide only the exact response without any additional explanations."},
        ]
    ).content[0].text

    print(message)

    if 'Medicator' in message:

        medicator2 = Task(
            description=f"""answer their medication related question quickly but efficiently and also in short no more than required . This is the past conversations (conversation starts here ----{careteam_history}-----end) and the lastest question : {current_question}. and answer it as fast as possible dont take time at all. The answer doesnt need to be the best. Also very important note: the answer MUST be short, no more than 2 lines """,
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



def image_processor_crew(description_of_docs, current_question):

    client = Anthropic()
    message = client.messages.create(
        model='claude-3-haiku-20240307',
        max_tokens=1024,
        messages=[
            {"role": "user",
             "content": f"Analyze the following chat history between the user and the AI://starts here {careteam_history}  ends here//. The chat history includes the user's latest question. Based on the content of the latest question, decide whether to keep the conversation with Agent 2 (Image Processor) or send it back to the base agent (Agent 0). If the conversation should be transferred to the Image Processor, reply with 'Agent 1. Image Processor'. If the conversation should remain with the base agent, reply with 'Agent 0. Base'. Provide only the exact response without any additional explanations."},
        ]
    ).content[0].text

    print(message)

    if 'Image' in message:

        imager1 = Task(
            description=f""" Consider the Data of the Image and then answer their questions properly. This is the past conversations (conversation starts here ----{careteam_history}-----end) and the lastest question : {current_question}. Also very important note: the answer MUST be short, no more than 2 lines """,
            agent=Medicator
        )

        crew = Crew(
            agents=[Medicator],
            tasks=[imager1],
            verbose=1,
        )

        # Get your crew to work!
        result = crew.kickoff()
    else:
        result = 'base'

    print("######################")
    print(result)

    return result