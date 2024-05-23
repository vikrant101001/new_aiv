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




def basedecidercrew(careteam_history, current_agent, user_question):
    # Instantiate your crew with a sequential process

    client = Anthropic()
    message = client.messages.create(
        model='claude-3-haiku-20240307',
        max_tokens=1024,
        system = """<ROLE>You are expert in analyzing the conversation provided between the user and AI Bot.</ROLE> \n <TASK> You will be provied with the conversation between the User and the AI Bot along with the next question asked by the user and which agent answered the last question. You have to predict which agent is going to answer the next questions. </TASK>
        <AGENTS> 1. BASE: User wants to have a normal conversation with the AI. \n 2. MEDICATOR: User wants have any medical related conversation with AI. \n 3. DOCUMENT: User wants to continue the document related conversation with the AI. </AGENTS> \n <INSTRUCTIONS> You must read the complete chat history and the next question asked by the User and give only the ONE WORD RESPONSE, telling which agent to answer the next question. <INSTRUCTIONS> <MUST>Only Name of the AGENT is expected in the response </MUST>""",
        messages=[
            {"role": "user",
             "content": f"""Analyze the following chat history between the user and the AI: \n {careteam_history} \n. AGENT THAT ANSWERED LAST QUESTION: {current_agent} \n NEXT QUESTION ASKED BY THE USER: {user_question}. Tell me which AGENT will answer the NEXT QUESTION. Only give name of the agent."""},
        ]
    ).content[0].text

    print("AGENT DECIDED -- ",message)
    return message


def medicator_crew(careteam_history, current_question):

    # client = Anthropic()
    # message = client.messages.create(
    #     model='claude-3-haiku-20240307',
    #     max_tokens=1024,
    #     messages=[
    #         {"role": "user",
    #          "content": f"Analyze the following chat history between the user and the AI://starts here {careteam_history}  ends here//. The chat history includes the user's latest question. Based on the content of the latest question, decide whether to keep the conversation with Agent 1 (Medicator) or send it back to the base agent (Agent 0). If the conversation should be transferred to the Medicator, reply with 'Agent 1. Medicator'. If the conversation should remain with the base agent, reply with 'Agent 0. Base'. Provide only the exact response without any additional explanations."},
    #     ]
    # ).content[0].text
    #
    # print(message)

    # if 'Medicator' in message:

    Medicator = Agent(
        role='Expert regarding Medicines',
        goal='Answer Question regarding Medicines',
        backstory="""As a Medicator, you specialize in pharmaceuticals and healthcare. Your role is to provide accurate information on medications, leveraging open-source data to ensure precision. Your expertise is crucial in guiding users with clarity and staying updated on medical advancements to offer valuable insights for informed decisions. You will use open source to find information regarding anything to do with medicines and answer questions properly. U also have knowledge regarding finding appropriate resources. You will use your own knowledge to find resources and not tell the user to use search tools and reply in proper format. Note: it is very very crucial that the datas are correct""",
        verbose=True,
        allow_delegation=False,
        llm=medicatorllm,
    )

    medicator2 = Task(
        description=f"""answer their medication related question quickly but efficiently and also in short no more than required . This is the past conversations (conversation starts here ----{careteam_history}-----end) and the latest question : {current_question}. and answer it as fast as possible dont take time at all. The answer doesnt need to be the best. Also very important note: the answer MUST be short, no more than 2 lines """,
        agent=Medicator,
        expected_output="Provide answer for the User's medical related query",

    )

    crew = Crew(
        agents=[Medicator],
        tasks=[medicator2],
        verbose=1,
    )

    result = crew.kickoff()


    print("######################")
    print(result)

    return result




def new_document_analyser(description, user_question):
    client = Anthropic()
    message = client.messages.create(
        model='claude-3-haiku-20240307',
        max_tokens=1024,
        system="""<ROLE>You are expert in analyzing the document description provided by User and answering the User Question.</ROLE> \n <TASK> You will be provied with the document description along with the question asked by the User. You are expected to give the detailed answer based on the description of the document provided only. <INSTRUCTION> Only Answer based on the descirption provided for the docuemnt. No External Information in the response is expected.</INSTRUCTION>""",
        messages=[
            {"role": "user",
             "content": f"""The description of the document is {description} and User_Question: {user_question}. Please provided detailed answer for the question based on the description provided."""},
        ]
    ).content[0].text

    print("New Document Analysis Result -- ", message)
    return message

def pdf_description(pdf_data):
    client = Anthropic()
    message = client.messages.create(
        model='claude-3-haiku-20240307',
        max_tokens=1024,
        system="""<ROLE>You are expert in analyzing the PDF document text provided by User.</ROLE> \n <TASK> You will be provied with the Text present in the PDF Document. You are expected to give the detailed description of the document by extracting all the information provided in the document. <INSTRUCTION>Generate the detailed description of the document only from the text provided</INSTRUCTION>""",
        messages=[
            {"role": "user",
             "content": f"""The text present in the PDF Document : {pdf_data}. Create a detailed description of the document"""},
        ]
    ).content[0].text

    if message:
        print("Document Description -- ", message)
        return message, 200
    else:
        print("Document description not created")
        message = "Failed"
        return message, 400

def document_processor_crew(careteam_history, current_question):

    Document = Agent(
        role='Expert in Analyzing Documents and Answering user specific questions on document',
        goal='Answering Questions asked by user regarding the Documents ',
        backstory="""As a Data Analyst, You have an expertize in analysing different kind of documents including images and pdfs. Your expertise is crucial in guiding users with clarity and answering User's query based on the documents . You only to use the information provided regarding the Document description""",
        verbose=True,
        allow_delegation=False,
        llm=medicatorllm,
    )

    document_task = Task(
        description=f"""You are provided with the chat_history of the conversation between user and AI regarding the discussion of document. Chat History: \n {careteam_history}. Analyse the chat history provided. Get the detailed understanding of the document. Now ANS the question asked by the USER regarding the document. User_Question: {current_question}""",
        agent=Document,
        expected_output="Provide answer for the User's medical related query",

    )

    crew = Crew(
        agents=[Document],
        tasks=[document_task],
        verbose=1,
    )

    result = crew.kickoff()

    print("######################")
    print(result)

    return result

