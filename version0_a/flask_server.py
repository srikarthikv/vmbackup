from flask import Flask, jsonify, request
from vm_helper_lib import *

app = Flask(__name__)

@app.route('/', methods=['GET'])
def welcome():
    return jsonify({'message': 'Welcome to BRT-GPT'})

@app.route('/question', methods=['POST'])
def process_question():
    print (request.is_json)
    question = request.form['question']
    detailed_option = request.form['detailed-option']
    vector_path = request.form['vectorstore-option']
    print (question,detailed_option,vector_path)
    # Process the question and generate a response
    if detailed_option == 'yes':
        print ("Detailed: yes")
        question = question + ". Provide a detailed explanation."
        print (question)

    response = get_response(question,vector_path,False)
    print ("Response generated.")
    return response

if __name__ == '__main__':
    context = ('cert.pem', 'key.pem') #certificate and key files
    app.run(host='0.0.0.0',port=8000,debug=False,ssl_context=context)
