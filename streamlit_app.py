# Imports
import os
import time
from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st
import fitz
import re
import docx
import openai
import json
import boto3
import sagemaker
from msal import ConfidentialClientApplication
from functions import parse_pdf, parse_docx, parse_txt, text_to_docs, test_embed, configure_openai,clear_functions
from langchain.document_loaders import Docx2txtLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain, RetrievalQA, VectorDBQA, RetrievalQAWithSourcesChain, ConversationChain
from langchain.chat_models import AzureChatOpenAI
from langchain.llms import OpenAI, AzureOpenAI
from langchain.vectorstores import FAISS, Chroma
from langchain.docstore.document import Document
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from questions import questions_10k,questions_def14a
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
# Function to get the list of folders in the specified directory

def get_folders_in_directory(directory):
    return [folder for folder in os.listdir(directory) if os.path.isdir(os.path.join(directory, folder))]

# Function to get the list of files in the specified folder
def get_files_in_folder(folder_path):
    return [file for file in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, file))]


# Constants
CHUNK_SIZE = 7000
GET_SOURCE = False
MODEL = "gpt-35-turbo-16k"
Final_Prompt=""" As a Corporate Governance and Financial Analyst, your role involves a meticulous analysis of corporate governance structures and financial practices. Avoid generalizations; focus on the provided context. Provide well-structured responses to specific queries. Take into account the given information:

Context:
{context}

Respond to the question directly. Refrain from assumptions unless explicitly specified. Also give the source from the document for the answer. 

Question:
{question}

"""

Question= """As a Corporate Governance and Financial Analyst, your role involves a meticulous analysis of corporate governance structures and financial practices. Improve the question to suit serve your needs. Do not change the core of the question, just improve it to streamline the results we get

Question: {question}"""


st.title('IS EDGAR SAMPLE DEMO')

# Default folder
data_folder = "Data"


# Get the list of folders
folders = get_folders_in_directory("./Data/")

# Add a None option to the list of folders
folders_with_none = [None] + folders

# Dropdown for selecting the folder with a default value
selected_folder = st.selectbox("Select a Folder", folders_with_none, index=folders_with_none.index(data_folder) if data_folder in folders_with_none else 0)

# Check if a folder is selected
if selected_folder is not None:
    # Get the path of the selected folder
    folder_path = os.path.join(".", selected_folder)

    # Get the list of files in the selected folder
    files = get_files_in_folder(folder_path)

    # Add a None option to the list of files
    files_with_none = [None] + files

    # Dropdown for selecting a file within the selected folder
    selected_file = st.selectbox("Select a File", files_with_none, index=files_with_none.index(data_folder) if data_folder in files_with_none else 0)
    # Check if a file is selected
    if selected_file is not None:
        # Read the content of the selected file
        MY_DOC = os.path.join(".", data_folder, selected_folder, selected_file) 
        doc = parse_pdf(MY_DOC)
        pages = text_to_docs(doc)
        # st.write("texttodoc - file ", MY_DOC)
        docsearch = test_embed(pages)
        # st.write("embedded file ", MY_DOC)
        api_key = configure_openai()
        llm = AzureChatOpenAI(deployment_name=MODEL, temperature=0)
        retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 15, 'lambda_mult': 0.70, "fetch_k": 50})
        PROMPT_mq = PromptTemplate(template=Question, input_variables=["question"])
        chain_type_kwargs = {"prompt": PROMPT_mq}
        # qa_chain = .from_chain_type(llm, chain_type_kwargs=chain_type_kwargs, return_source_documents=GET_SOURCE)
        PROMPT = PromptTemplate(template=Final_Prompt, input_variables=["context", "question"])
        chain_type_kwargs = {"prompt": PROMPT}
        qa_chain = RetrievalQA.from_chain_type(llm, chain_type="stuff", retriever=retriever, chain_type_kwargs=chain_type_kwargs, return_source_documents=GET_SOURCE)
        # Check if the selected folder is "10k"
        if selected_folder.lower() == "10k":
            # Display 10-K questions in a dropdown
            questions_10k_list = ['None'] + questions_10k['Question'].tolist()
            selected_question = st.selectbox("Select a 10-K Question", questions_10k_list)

            # Check if "None" is selected
            if selected_question == 'None':
                # Hide the dropdown and show a text input instead
                selected_question = st.text_input("Enter your custom question:", value="")

            # Process the question when the "Process Question" button is pressed
            if st.button("Process Question"):
                if selected_question.lower() == 'q':
                    st.subheader("Quit")
                    st.stop()
                    
                prompt = ChatPromptTemplate.from_template(Question)
                chain = prompt | llm
                Modified_question=chain.invoke({"question": selected_question})
                matches = re.findall(r'Question:\s*(.*)', str(Modified_question))
                matches = [selected_question] if not matches or all(match.strip() == '' for match in matches) else matches
                result = qa_chain({'query': str(matches)})
                st.write(result['result'])
        # Check if the selected folder is "def14a"
        elif selected_folder.lower() == "def14a":
            # Display DEF14A questions in a dropdown
            questions_def14a_list = ['None'] + questions_def14a['Question'].tolist()
            selected_question = st.selectbox("Select a DEF14A Question", questions_def14a_list)
  # Check if "None" is selected
            if selected_question == 'None':
                # Hide the dropdown and show a text input instead
                selected_question = st.text_input("Enter your custom question:", value="")

            # Process the question when the "Process Question" button is pressed
            if st.button("Process Question"):
                if selected_question.lower() == 'q':
                    st.subheader("Quit")
                    st.stop()
                    
                prompt = ChatPromptTemplate.from_template(Question)
                chain = prompt | llm
                Modified_question=chain.invoke({"question": selected_question})
                matches = re.findall(r'Question:\s*(.*)', str(Modified_question))
                matches = [selected_question] if not matches or all(match.strip() == '' for match in matches) else matches
                st.write(str(matches))
                result = qa_chain({'query': str(matches)})
                st.write(result['result'])

    else:
        st.warning("Please select a file.")

else:
    st.warning("Please select a folder.")
