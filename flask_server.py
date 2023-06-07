from flask import Flask, jsonify, request
#import pyodbc
from helper_lib import *

app = Flask(__name__)

@app.route('/', methods=['GET'])
def welcome():
    return jsonify({'message': 'Welcome to BRT-GPT'})

@app.route('/question', methods=['POST'])
def process_question():
    data = request.get_json()
    question = data['question']

    # Process the question and generate a response
    response = get_response(query=question,use_merged=True)

    return jsonify({'response': response})

@app.route('/embed', methods=['POST'])
def embed_document():
    data = request.get_json()
    file_path = data['file_path']
    try:
        uploaded_path = embed_and_upload(file_path)
    except pyodbc.IntegrityError as e:
        # Handle the IntegrityError exception here
        error_message = "Error occurred while embedding and uploading the document.\n"+ str(e)
        return jsonify({'error': error_message}), 500

    return jsonify({'uploaded_path': uploaded_path})

# @app.route('/embed', methods=['POST'])
# def embed_document():
#     data = request.get_json()
#     file_path = data['file_path']
#     uploaded_path = embed_and_upload(file_path)

#     return jsonify({'uploaded_path': uploaded_path})
    
if __name__ == '__main__':
    context = ('cert.pem', 'key.pem') #certificate and key files
    app.run(host='0.0.0.0',port=8000,debug=False,ssl_context=context)
    #app.run(port=8000,debug=False)
