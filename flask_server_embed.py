from flask import Flask, jsonify, request
from vm_helper_lib import *

app = Flask(__name__)

@app.route('/', methods=['GET'])
def welcome():
    return jsonify({'message': 'Welcome to BRT-GPT'})

@app.route('/question', methods=['POST'])
def process_question():
    question = request.form['question']

    # Process the question and generate a response
    response = get_response(query=question,vector_path='',use_merged=True)

    return (response)

@app.route('/embed', methods=['POST'])
def embed():
	file_path = request.form['file_path']

	embed_and_upload(file_path)

if __name__ == '__main__':
    context = ('cert.pem', 'key.pem') #certificate and key files
    app.run(host='0.0.0.0',port=8000,debug=False,ssl_context=context)
    #app.run(port=8000,debug=False)
