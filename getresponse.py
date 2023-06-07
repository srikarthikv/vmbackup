def gpt3resp(question,vectorfile): 
  with open(vectorfile,"rb") as f:
    vectorstore = pickle.load(f)
  qa = ConversationalRetrievalChain.from_llm(AzureChatOpenAI(deployment_name="GPT35-Demo1",model_name="gpt-35-turbo",temperature=0.4),vectorstore.as_retriever())
  response = qa({"question": question, "chat_history": ""}) 
  return(response["answer"])
