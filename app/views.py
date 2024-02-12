# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""

# Flask modules
from flask   import render_template, request
from jinja2  import TemplateNotFound


# Import necessary libraries
from flask import Flask, render_template, request, redirect,jsonify

import os
import time
import os
from langchain.vectorstores import Milvus
from langchain.embeddings import HuggingFaceEmbeddings
from PIL import Image
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
import os
from langchain.schema.runnable import RunnablePassthrough,RunnableLambda
from langchain.chat_models import ChatOpenAI
import base64
from pathlib import Path
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.vectorstores import FAISS
from operator import itemgetter
from langchain.chains import ConversationalRetrievalChain 
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.agents import AgentExecutor, Tool,initialize_agent
from langchain.agents.types import AgentType
from langchain.schema.runnable import RunnableMap
import tempfile
from langchain_community.document_loaders import UnstructuredFileLoader
from unstructured.cleaners.core import clean_extra_whitespace


import os
import shutil



def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# Set the OpenAI API key
print(os.getenv("OPENAI_API_KEY"))
os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(model_name="gpt-3.5-turbo-1106", temperature=0.9)

# Define the name of the bot
name = 'SolonX'

# Define the role of the bot
role = 'Assistant'

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")



# App modules
from app import app

# App main route + generic routing
@app.route('/', defaults={'path': 'chat.html'})
@app.route('/<path>')
def index(path):

    try:

        # Detect the current page
        segment = get_segment( request )

        # Serve the file (if exists) from app/templates/FILE.html
        return render_template( path, segment=segment )
    
    except TemplateNotFound:
        return render_template('page-404.html'), 404

def get_segment( request ): 

    try:

        segment = request.path.split('/')[-1]

        if segment == '':
            segment = 'index'

        return segment    

    except:
        return None  




def generate_response(prompt_input):
    vectorstore = FAISS.load_local("faiss_index", embeddings)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    template = """you are an Assistant Use the following pieces of context to answer the question at the end,formatted in HTML. 
    If you don't know the answer, just reply with exact words "I do not know", don't try to make up an answer. 
    Use  five  sentences maximum  keep the answer as concise as possible. 

    Always say "thanks for asking!" at the end of the answer. 
    {context}
    Question: {question}
    Helpful Answer:
    Tips: Use proper HTML formatting while writing the answer.
    """
    rag_prompt_custom = PromptTemplate.from_template(template)


    rag_chain_from_docs = (
    {
        "context": lambda input: format_docs(input["documents"]),
        "question": itemgetter("question"),
    }
    | rag_prompt_custom
    | llm
    | StrOutputParser()
    )
    rag_chain_with_source = RunnableMap(
        {"documents": retriever, "question": RunnablePassthrough()}
    ) | {
        "documents": lambda input: [doc.metadata for doc in input["documents"]],
        "answer": rag_chain_from_docs,
    }

    response= rag_chain_with_source.invoke(prompt_input)
    print(response)

    if ("don't have information" in response['answer']) or ("I do not know" in response['answer']) or ("not known" in response['answer']) or ("can't answer" in response['answer']):
        source_docs=""
    else:
        source_docs="""<br><strong>Documents Link: </strong>\n\n<a href="https://drive.google.com/drive/folders/1uzn3E3JOqAZ54ZltPRfx0zEfdt6cbkXZ?usp=sharing"> Drive Link</a>  <br>\n\n <strong> Source Documents: </strong> <br>\n\n"""+"<br>\n\n".join([f"Document: {doc['source']} Page Number:{doc['page_number']}" for doc in response['documents']])
    
    full_response = f"""{response['answer']} \n 
        """
    return full_response,source_docs


# Initialize variables for chat history
explicit_input = ""
chatgpt_output = 'Chat log: /n'
cwd = os.getcwd()
i = 1


# Initialize chat history
chat_history = ''



# Function to handle user chat input
def chat(user_input):
    global chat_history, name, chatgpt_output
    current_day = time.strftime("%d/%m", time.localtime())
    current_time = time.strftime("%H:%M:%S", time.localtime())
    chat_history += f'\nUser: {user_input}\n'
    chatgpt_raw_output,source_docs = generate_response(user_input)
    chatgpt_output = f"""{name}: {chatgpt_raw_output}"""
    chat_history += chatgpt_output + '\n'
    output= f"""{chatgpt_raw_output} \n\n {source_docs}"""

    return output

# Function to get a response from the chatbot
def get_response(userText):
    return chat(userText)



@app.route("/upload",methods=['GET','POST'])
def upload():
    if request.method=='GET':
        return render_template("upload.html")
    if request.method=='POST':
        uploaded_files = request.files.getlist("files")
        for file in uploaded_files:
            # Ensure the filename is safe
            file_extension = os.path.splitext(file.filename)[1]
            # Create a temporary file
            temp_file = tempfile.NamedTemporaryFile(delete=False,suffix=file_extension)
            # Save the uploaded file to the temporary file
            file.save(temp_file.name)
            loader = UnstructuredFileLoader(file_path=temp_file.name, post_processors=[
                                        clean_extra_whitespace], mode='elements')
            data = loader.load()
            for doc in data:
                doc.metadata['source']=file.filename 
            db=FAISS.from_documents(data, embeddings)
            db.save_local("faiss_index")
        return render_template('chat.html')



@app.route("/get")
# Function for the bot response
def get_bot_response():
    userText = request.args.get('msg')
    return str(get_response(userText))

@app.route('/refresh')
def refresh():
    time.sleep(600) # Wait for 10 minutes
    return redirect('/refresh')


@app.route('/test')
def test():
    return jsonify({"msg":"test success"})
# Run the Flask app
if __name__ == "__main__":
    app.run()