

###############################################################################
# This Python program is to access OpenAI's ChatGPT.
# Fine tuning of ChatGPT with One-Shot document to adapt to post 
# September 2021 content and achieve Semi-Adaptive AI. Fine-tuning improves
# on domain-specific few-shot learning by training on many more examples 
# that cannot be managed through prompt engineering. 
# It helps ChatGPT achieve better results on tasks that it is unfamiliar with.
######################
# In the Python code we run a FLASK Web server with our OpenAI/GPT3 API
# The FLASK code depending on GET ot POST runs different functions within
# the program
###############################################################################

# Import all necessary libraries
import os
import openai
import langchain
import pypdf
import pickle
from langchain.llms import AzureOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.docstore.document import Document
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import AzureChatOpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain

# Set up Azure & OpenAI environment
openai.api_type = "azure"
openai.api_base = "https://gpt-demo1.openai.azure.com/"
openai.api_version = "2022-12-01"
api_key = "d6420559b1154f5d82f8364ec4c77b55"

os.environ["OPENAI_API_KEY"] = api_key
os.environ["AZURE_OPENAI_API_KEY"] = api_key
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://gpt-demo1.openai.azure.com/"
os.environ["OPENAI_ENGINES"] = "GPT-35-Demo1"
os.environ["OPENAI_EMBEDDINGS_ENGINE_DOC"] = "text-embedding-ada"
os.environ["OPENAI_EMBEDDINGS_ENGINE_QUERY"] = "text-embedding-ada"
os.environ["OPENAI_API_BASE"] = "https://gpt-demo1.openai.azure.com"
os.environ["OPENAI_ENGINES"] = "GPT35-Demo1"
os.environ["OPENAI_API_VERSION"] = "2023-03-15-preview"

#d6420559b1154f5d82f8364ec4c77b55
#os.environ["OPENAI_API_KEY"] = "d6420559b1154f5d82f8364ec4c77b55"

###############################################################################
# This function is for logging messages
# Messages include all messages entered by user
# And, all messages displayed by the program
######################
def logmessage(user_message, history):
    # Get response from QA chain
    response = qa({"question": user_message, "chat_history": history})
    # Append user message and response to chat history
    history.append((user_message, response["answer"]))

###############################################################################
# This function is for PDF file embedding.
# This function takes only one parameter
# INPUT:
#   pdfDoc: The input PDF document to be embedded
# OUTPUT: None
# RETURN:
#  The embedded file name is passed to the calling program as return value
# EXAMPLE:
#   pdfEmbedding("data/Dengue-National-Guidelines-2014.pdf")
######################
def pdfEmbedding(pdfDoc):
  # Load PDF documents into the Document format that we will use downstream
  # https://python.langchain.com/en/latest/modules/indexes/document_loaders/examples/pdf.html
  time_before = datetime.now()
  loader = PyPDFLoader(pdfDoc)
  pages = loader.load_and_split()
  # Create an empty container
  merged_db = FAISS.from_documents([Document(page_content=" ")], OpenAIEmbeddings(model="text-embedding-ada-002"))
  # Read and add each PDF page into embedding container separately
  search_indices = []
  for i, chunk in enumerate(pages):
    data = [Document(page_content=str(chunk))]
    var_name = f"search_index_{i}"
    locals()[var_name] = FAISS.from_documents(data, OpenAIEmbeddings(model="text-embedding-ada-002"))
    search_indices.append(locals()[var_name])
    #merged_db.merge_from(locals()[var_name])
  # Now merge the separate emdeddings into one seamless embedding
  for db in search_indices:
    merged_db.merge_from(db)
  # Return the embedded object
  vectorfile = "vectorstore.pkl"
  with open(vectorfile, "wb") as f:
    pickle.dump(merged_db,f)
  f.close()
  time_after = datetime.now()
  print("Embedding Time taken:",time_after - time_before)
  return(vectorfile)

###############################################################################

# During Q&A we use both Native GPT3 and One-shot GPT3 for comparison
# This functions takes TWO parameters
# INPUT:
#   question: the question as entered by the user
#   vectorfile: the vector embedding file
# OUTPUT:
#   None
# RETURN:
#   The Zero-shot response from GPT3
#   The One-shot response from GPT3
# EXAMPLE:
#   gpt3resp("bla bla","vector.pkl")
######################

from datetime import datetime

def gpt3resp(question,vectorfile):
  chat_history = []
  #question1 = "What is the volume of fluid to be given in 24 hours maintenance for 30 kgs bodyweight"
  #question2 = "Use Table 1. Requirement of fluid based on bodyweight. Tell me the volume of fluid to be given in 24 hrs Maintenance for bodyweight=30 kgs"
  #llm = OpenAI(model_name="text-davinci-003",temperature=0.9)
  # Try the GPT3 with default settings
  question = str(question)
  llm = AzureOpenAI(deployment_name="text-davinci-003", model_name="text-davinci-003",temperature=0.4)
  responseZero = llm(question)
  time_before = datetime.now()
  with open(vectorfile,"rb") as f:
    vectorstore = pickle.load(f)
  f.close()
  qa = ConversationalRetrievalChain.from_llm(AzureChatOpenAI(deployment_name="GPT35-Demo1",model_name="gpt-35-turbo",temperature=0.4),vectorstore.as_retriever())
  time_after = datetime.now()
  print("Q&A Time taken:",time_after - time_before)
  responseOne = qa({"question": question, "chat_history": ""})
  return responseOne
  #return(responseZero,responseOne["answer"])
  #return(responseOne["answer"])


###############################################################################
# This is the FLASK part of the program. The above functions are called 
#   through FLUSK iinternet server.
# INPUT:
# Here we have implemented two HTTP POST functions.
# 1) embedding: this function is running as a HTTP server at port 8000
#    This function accepts the pdf document name as a JSON input and
#    sends back the embedded file name as JSON output
# 2) qanda: this function is running as a HTTP server at port 8000
#    This function accests the Question and the embedded vector file
#    as JSON input and sends back the Zero-shor and Once-shot answers back
#    as JSON output
# OUTPUT:
#   None
# RETURN:
#   embedding function returns the vector file
#   qanda function retirns Question, Zero-shot answer, and One-shot answer
# EXAMPLE:
######################
from flask import Flask
from flask import request
from flask import jsonify
from datetime import datetime

app = Flask(__name__)

##############################
# The General GET command processing
@app.route('/', methods = ['GET'])
def index():
    return("\n=====================\nWelcome to BRTGPT\nCONGRATULATIONS you have SUCCESSFULLY CONNECTED\n=====================\n\n")

##############################
# The embedding POST command processing
@app.route('/embedding', methods = ['GET','POST'])
def postEmbeddingHandler():
    print (request.is_json)
    print("Parse Embedding JSON")
    content = request.get_json()
    document = content['document']
    print(document)
    vectorfile = pdfEmbedding(document)
    return(jsonify(vectorfile=vectorfile))

##############################
# The qanda POST command processing
@app.route('/qanda', methods = ['POST'])
def postJsonHandler():
    print (request.is_json)
    print("Parse Q&A JSON")
    print(request.form)
    content = request.get_json()
    #question = request.form['question']
   # document =request.form['document']
    #document='vectorstore.pkl'
    question = content['question']
    document= content['document']
    print(question,document) 
    responseOne=gpt3resp(question,document)
    #responseZero,responseOne = gpt3resp(question,document)
    #return(jsonify(question=question,responseZero=responseZero,responseOne=responseOne))
    return (responseOne['answer'])
##############################
# RUN the FLUSK program in debugging mode#


context = ('cert.pem','key.pem')

#app.run(debug=True, host='10.0.0.4', port=8000,ssl_context = context)

app.run(host='0.0.0.0',port=8000,debug=True)

#########################################################################
