from flask import Flask, jsonify, request
from helper_lib import *
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import AzureChatOpenAI
import openai
import os
import pickle
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

def gpt3resp(question,vectorfile):
    #blob_instance = Blob_storage_access()
    #vectorstore = blob_instance.read_vector(vectorfile)
    with open(vectorfile,"rb") as f:
        vectorstore = pickle.load(f)
    qa = ConversationalRetrievalChain.from_llm(AzureChatOpenAI(deployment_name="GPT35-Demo1",model_name="gpt-35-turbo",temperature=0.4),vectorstore.as_retriever())
    response = qa({"question": question, "chat_history": ""}) 
    return(response["answer"])

app = Flask(__name__)

@app.route('/', methods=['GET'])
def welcome():
    return jsonify({'message': 'Welcome to BRT-GPT'})

@app.route('/question', methods=['POST'])
def process_question():
    print (request.is_json)
    question = request.form['question']
    detailed_option = request.form['detailed-option']
    vector_path = "codecatalyst_ug_vectorstore.pkl"
    print (question,detailed_option,vector_path)
    # Process the question and generate a response
    if detailed_option == 'yes':
        print ("Detailed: yes")
        question = question + ". Provide a detailed explanation."
        print (question)

    response = gpt3resp(question,vector_path)
    print ("Response generated.")
    return response

if __name__ == '__main__':
    context = ('cert.pem', 'key.pem') #certificate and key files
    app.run(host='0.0.0.0',port=8000,debug=False,ssl_context=context)
    #app.run(port=8000,debug=False)
