#!/usr/bin/env python
# coding: utf-8

# In[1]:

# def main():
#     #required constants
#     KEY_FILE = '/home/azureuser/cloudfiles/code/keys/openai_key.txt'
#     KNOWLEDGE_DOCS = '/home/azureuser/cloudfiles/code/Users/richard_malchi/Database/documents'
#     DEFAULT_EMBED_BATCH_SIZE = 1
#     EMBEDDING_MODEL = 'text-embedding-ada-002'
#     EMBEDDING_CTX_LENGTH = 8191
#     EMBEDDING_ENCODING = 'cl100k_base'
#     RESOURCE_ENDPOINT = ""

#required constants
#KEY_FILE = '/home/azureuser/cloudfiles/code/keys/openai_key.txt'
#KNOWLEDGE_DOCS = '/home/azureuser/cloudfiles/code/Users/richard_malchi/Database/documents'
DEFAULT_EMBED_BATCH_SIZE = 1
EMBEDDING_MODEL = 'text-embedding-ada-002'
EMBEDDING_CTX_LENGTH = 8191
EMBEDDING_ENCODING = 'cl100k_base'
RESOURCE_ENDPOINT = ""
# In[ ]:


# """
# This class provides access to OpenAI models and functionalities for language processing tasks.
# It includes methods for initializing models, embeddings, and chains, loading and processing documents,
# generating responses, and more.
# """
class OpenaAI:
    def __init__(self):
        import os
        import openai
        import langchain
        #with open(KEY_FILE, 'r') as file:
            #api_key = file.read().replace('\n', '')

        # Set OpenAI API configuration
        api_key = "d6420559b1154f5d82f8364ec4c77b55"
        openai.api_type = "azure"
        openai.api_base = "https://gpt-demo1.openai.azure.com/"
        openai.api_version = "2022-12-01"
        os.environ["OPENAI_API_KEY"] = api_key
        os.environ["AZURE_OPENAI_API_KEY"] = api_key
        os.environ["AZURE_OPENAI_ENDPOINT"] = "https://gpt-demo1.openai.azure.com/"
        os.environ["OPENAI_EMBEDDINGS_ENGINE_DOC"] = "text-embedding-ada"
        os.environ["OPENAI_EMBEDDINGS_ENGINE_QUERY"] = "text-embedding-ada"
        os.environ["OPENAI_API_BASE"] = "https://gpt-demo1.openai.azure.com"
        os.environ["OPENAI_ENGINES"] = "GPT35-Demo1"
        os.environ["OPENAI_API_VERSION"] = "2023-03-15-preview"
        RESOURCE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")

    # """
    # Initialize the OpenAI model.
    # Args:
    #     model_type (str): The type of model to initialize. Defaults to "Chat".
    # Returns:
    #     object: The initialized model object.
    # """
    def init_model(self, model_type="Chat"):
        import openai
        import os
        from langchain.llms import AzureOpenAI
        from langchain.chat_models import AzureChatOpenAI, ChatOpenAI

        if model_type == "Chat":
            model = AzureChatOpenAI(
                openai_api_base=openai.api_base,
                openai_api_version="2023-03-15-preview",
                deployment_name=os.environ["OPENAI_ENGINES"],
                openai_api_key=openai.api_key,
                openai_api_type=openai.api_type
            )
        else:
            model = AzureOpenAI(
                openai_api_base=openai.api_base,
                deployment_name="text-davinci-003",
                openai_api_key=openai.api_key,
            )
        return model

    # """
    # Initialize the OpenAI embeddings.
    # Returns:
    #     object: The initialized embeddings object.
    # """
    def init_embeddings(self):

        from langchain.embeddings import OpenAIEmbeddings

        embedding = OpenAIEmbeddings(model=EMBEDDING_MODEL, chunk_size=1)

        return embedding

    # """
    # Initialize the language model chain.
    # Args:
    #     prompt (str): The prompt to use for the chain.
    # Returns:
    #     object: The initialized language model chain object.
    # """
    def init_chain(self, prompt):
        from langchain.chains import LLMChain

        model = self.init_model(model_type="Chat")

        chatgpt_chain = LLMChain(
            llm=model,
            prompt=prompt,
            verbose=True
        )

        return chatgpt_chain

    # """
    # Load a PDF document.
    # Args:
    #     docpath (str): The path to the PDF document.
    #     online (bool): Indicates whether the document is loaded online. Defaults to False.
    # Returns:
    #     object: The loaded PDF document data.
    # """
    def pdf_loader(self, docpath, online=False):
        if not online:
            from langchain.document_loaders import PyMuPDFLoader
            loader = PyMuPDFLoader(docpath)
            data = loader.load()
            data[0]
        else:
            from langchain.document_loaders import OnlinePDFLoader
            loader = OnlinePDFLoader(docpath)
            data = loader.load()
            data[0]

        return data

    # """
    # Get the appropriate document loader based on the file type.
    # Args:
    #     file_path_or_url (str): The path or URL of the document.
    # Returns:
    #     object: The appropriate document loader object.
    # """
    def get_loader(self, file_path_or_url):
        import mimetypes
        from langchain.document_loaders import TextLoader, BSHTMLLoader, WebBaseLoader, PyMuPDFLoader, CSVLoader, UnstructuredWordDocumentLoader, WebBaseLoader

        if file_path_or_url.startswith("http://") or file_path_or_url.startswith("https://"):
            handle_website = URLHandler()
            return WebBaseLoader(handle_website.extract_links_from_websites([file_path_or_url]))
        else:
            mime_type, _ = mimetypes.guess_type(file_path_or_url)

            if mime_type == 'application/pdf':
                return PyMuPDFLoader(file_path_or_url)
            elif mime_type == 'text/csv':
                return CSVLoader(file_path_or_url)
            elif mime_type in ['application/msword',
                               'application/vnd.openxmlformats-officedocument.wordprocessingml.document']:
                return UnstructuredWordDocumentLoader(file_path_or_url)
            elif mime_type == 'text/plain':
                return TextLoader(file_path_or_url)
            elif mime_type == 'text/html':
                return BSHTMLLoader(file_path_or_url)
            else:
                raise ValueError(f"Unsupported file type: {mime_type}")

    # """
    # Ingest and process documents.
    # Args:
    #     file_path_or_url (str): The path or URL of the document.
    # Returns:
    #     list: A list of processed documents.
    # """
    def ingest_docs(self, file_path_or_url):
        from langchain.text_splitter import TokenTextSplitter

        loader = self.get_loader(file_path_or_url)
        raw_documents = loader.load()
        text_splitter = TokenTextSplitter(chunk_size=1000, chunk_overlap=200)
        documents = text_splitter.split_documents(raw_documents)

        return documents

    # """
    # Generate embeddings for the document.
    # Args:
    #     file_path_or_url (str): The path or URL of the document.
    # Returns:
    #     object: The generated embeddings.
    # """
    def embed(self, file_path_or_url):
        from langchain.vectorstores import FAISS
        import os
        import pickle

        embeddings = self.init_embeddings()
        documents = self.ingest_docs(file_path_or_url)
        vectorstore = FAISS.from_documents(documents, embeddings)

        return vectorstore

    # """
    # Generate a response based on the query.
    # Args:
    #     query (str): The query text.
    #     vectorstore (object): The vectorstore object containing document embeddings. Defaults to None.
    #     use_merged (bool): Indicates whether to use merged documents for retrieval. Defaults to True.
    # Returns:
    #     str: The generated response.
    # """
    def generate_response(self, query, vectorstore=None, use_merged=True):
        from langchain.chains import ConversationalRetrievalChain
        from langchain.chains.question_answering import load_qa_chain
        from langchain.prompts import PromptTemplate

        if vectorstore is not None:
            if use_merged:
                model = self.init_model("Chat")
                prompt = PromptTemplate(
                    input_variables=["context", "question"],
                    template="""Use the following pieces of context to answer the question. If you don't know the answer, just say that you don't know, don't try to make up an answer.
                    {context}
                    Question: {question}
                    Helpful Answer: """
                )
                chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
                docs = vectorstore.similarity_search(query)
                response = chain.run(input_documents=docs, question=query)
            else:
                llm = self.init_model("Chat")
                qa = ConversationalRetrievalChain.from_llm(llm, vectorstore.as_retriever())
                response = qa({"question": query, "chat_history": ""})['answer']
        else:
            llm = self.init_model("native")
            response = llm(query)

        return response


# In[ ]:


# """
# This class provides access to Azure Blob storage using the Azure Machine Learning file system.
# It allows listing files, downloading blobs, uploading blobs, and reading vector data from the storage.
# """
class Blob_storage_access:
    # """
    # Initialize the Blob_storage_access class and set up the Azure Machine Learning file system.
    # """
    def __init__(self):
        from azureml.fsspec import AzureMachineLearningFileSystem

        subid = '4986991a-fc7f-4d80-8c34-f2a4d0c66059'
        resource_group = 'ranjan_goal-rg'
        workspace = 'gptdemo'  # Workspace name
        datastore_name = 'workspaceblobstore'  # Datastore name
        documents_on_datastore = '/documents'  # Path to documents on the datastore
        vectors_on_datastore = '/vectors'  # Path to vectors on the datastore

        # Create the URI for the Azure Machine Learning file system
        uri = f'azureml://subscriptions/{subid}/resourcegroups/{resource_group}/workspaces/{workspace}/datastores/{datastore_name}/paths'

        # Initialize the Azure Machine Learning file system
        self.FS = AzureMachineLearningFileSystem(uri)

    # """
    # List files in the specified path on the datastore.
    # Args:
    #     path_on_datastore (str): The path on the datastore.
    # Returns:
    #     list: A list of file paths.
    # """
    def list_files(self, path_on_datastore):
        # Get the file system object
        fs = self.FS
        # List files in the specified path on the datastore
        file_list = fs.ls(path_on_datastore)

        return file_list

    # """
    # Download a blob from the remote path to the local path.
    # Args:
    #     local_path (str): The local path to save the downloaded blob.
    #     remote_path (str): The remote path of the blob to download.
    # Returns:
    #     str: The local path of the downloaded blob.
    # """
    def download_blob(self, local_path, remote_path):
        import os

        # Get the file system object
        fs = self.FS

        # Download the blob from the remote path to the local path
        fs.download(rpath=remote_path, lpath=local_path, recursive=False, **{'overwrite': 'MERGE_WITH_OVERWRITE'})

        # Get the name of the downloaded blob
        doc_name = os.path.basename(remote_path)

        # Return the local path of the downloaded blob
        return local_path + '/' + doc_name

    # """
    # Upload a blob from the local path to the remote path.
    # Args:
    #     local_path (str): The local path of the blob to upload.
    #     remote_path (str): The remote path to save the uploaded blob.
    # Returns:
    #     str: The remote path of the uploaded blob.
    # """
    def upload_blob(self, local_path, remote_path):
        import os

        # Get the file system object
        fs = self.FS

        # Upload the blob from the local path to the remote path
        fs.upload(lpath=local_path, rpath=remote_path, recursive=False, **{'overwrite': 'MERGE_WITH_OVERWRITE'})

        # Get the name of the uploaded blob
        vector_name = os.path.basename(local_path)

        # Return the remote path of the uploaded blob
        return remote_path + '/' + vector_name

    # """
    # Read the vector data from the specified path on the datastore.
    # Args:
    #     path_on_datastore (str): The path on the datastore to read the vector data from.
    # Returns:
    #     object: The vectorstore object.
    # """
    def read_vector(self, path_on_datastore):
        import pickle

        # Get the file system object
        fs = self.FS
        # Read the vector data from the specified path on the datastore
        with fs.open(path_on_datastore, "rb") as f:
            vectorstore = pickle.loads(f.read())

        return vectorstore


# In[ ]:


# """
# This class provides access to an SQL database and contains functions to execute SQL queries and perform database operations.
# """
class Sql_database_access:
    # """
    # Initialize the Sql_database_access class and establish the connection to the SQL database.
    # """
    def __init__(self):
        server = 'tcp:sqldatabaseserverbrt.database.windows.net,1433'
        database = 'sqldatabasebrt'
        username = 'CloudSA9c26d955'
        password = 'BRTpassword1'
        driver = '{ODBC Driver 18 for SQL Server}'

        # Establish the database connection
        self.conn_string = f"DRIVER={driver};SERVER={server};DATABASE={database};UID={username};PWD={password};Encrypt=yes;TrustServerCertificate=no;"
        # connection = pyodbc.connect(conn_string)

    # """
    # Execute the given SQL query.
    # Args:
    #     query (str): The SQL query to execute. Default is "select * from knowledge".
    #     display (bool): Whether to display the query results. Default is False.
    #     values (tuple): The parameter values for the query. Default is None.
    # """
    def execute_query(self, query="select * from knowledge", display=False, values=None):
        import pyodbc

        # Establish the database connection
        connection = pyodbc.connect(self.conn_string)
        # Create a cursor object to execute SQL queries
        cursor = connection.cursor()

        if values is None:
            cursor.execute(query)
        else:
            cursor.execute(query, values)

        if display:
            rows = cursor.fetchall()
            # Process the query results
            for row in rows:
                print(row)

        # Close the cursor and connection
        connection.commit()
        cursor.close()
        connection.close()

    # """
    # Insert a row into the SQL database.
    # Args:
    #     doc_name (str): The name of the document.
    #     doc_path (str): The path of the document.
    #     vector_path (str): The path of the vector file.
    # """
    def insert_row(self, doc_name, doc_path, vector_path):
        from datetime import datetime

        # Define the SQL query to insert values into the table
        query = """INSERT INTO knowledge (type, docname, date, release, url, provenance, embeddinglocation, file_location, docid) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)"""

        # Define the parameter values to be inserted
        type = 'pdf'
        docname = doc_name
        date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # Convert datetime to string
        release = 'release'
        url = 'url'
        docpath = doc_path
        docid = "docid"
        provenance = 'Provenance Value'
        vectorpath = vector_path

        values = (type, docname, date, release, url, provenance, vectorpath, docpath, docid)
        self.execute_query(query, values=values)


# In[ ]:


# """
# Method: embed_and_upload
# Embeds a list of documents using OpenAI and uploads the resulting vectorstores to Blob storage
# Parameters:
# - file_list: List of document paths or a single document path (string)
# - merge: Boolean flag indicating whether to merge the vectorstores or not (default: False)
# Returns: None
# """
def embed_and_upload(file_list, merge=False):
    import os
    import pickle

    # Initialize instances of required classes
    blob_instance = Blob_storage_access()  # Blob storage access instance
    openai_instance = OpenaAI()  # OpenAI instance
    sql_instance = Sql_database_access()  # SQL database access instance

    # Ensure file_list is a list
    if not isinstance(file_list, list):
        file_list = list({file_list})

    # Define local and remote directories for storing documents and vectors
    local_docs_dir = "/home/virtualmachineuser/temp_database/documents/"
    local_vectors_dir = "/home/virtualmachineuser/temp_database/vectors/"
    remote_vectors_dir = "vectors"
    local_merged_vector = local_vectors_dir + "merged_vectorstore.pkl"

    # Check if merge flag is True
    if merge:
        # Load merged vectorstore from remote storage
        with blob_instance.FS.open("vectors/merged_vectorstore.pkl", "rb") as f:
            merged_vectorstore = pickle.load(f)
    else:
        merged_vectorstore = ''  # Initialize merged vectorstore as empty string

    # Process each file in the file_list
    for file in file_list:
        # print(file)

        remote_doc = file  # Remote document path
        local_download = blob_instance.download_blob(local_docs_dir, remote_doc)  # Download document locally
        #print("download done")

        vectorstore = openai_instance.embed(local_download)  # Embed the downloaded document
        #print("embedding done")

        # Extract document and vector names
        doc_name_w_ext = os.path.basename(local_download)
        doc_name = os.path.splitext(doc_name_w_ext)[0]
        vector_name = doc_name + "_vectorstore.pkl"

        local_vector = local_vectors_dir + vector_name  # Local vector path
        with open(local_vector, "wb") as f:
            pickle.dump(vectorstore, f)  # Save vectorstore to local file

        if merge:
            merged_vectorstore.merge_from(vectorstore)  # Merge vectorstore with the merged vectorstore

        remote_upload = blob_instance.upload_blob(local_vector, remote_vectors_dir)  # Upload vectorstore to remote storage
        #print(f"{vector_name} is uploaded to {remote_vectors_dir}")

        # Insert metadata into the SQL database
        #sql_instance.insert_row(doc_name=doc_name_w_ext, doc_path=local_download, vector_path=remote_upload)
        #print("record meta data to SQL database is done")

    if merge:
        with open(local_merged_vector, "wb") as f:
            pickle.dump(merged_vectorstore, f)  # Save merged vectorstore to local file

        blob_instance.upload_blob(local_merged_vector, remote_vectors_dir)  # Upload merged vectorstore to remote storage

    return remote_vectors_dir+'/'+vector_name

# In[ ]:


# """
# Method: get_response
# Generates a response to a given query using OpenAI, optionally using a specific vectorstore
# Parameters:
# - query: The query to generate a response for
# - vector_path: Path to a specific vectorstore (default: None)
# - is_merged: Boolean flag indicating whether the vectorstore is merged or not (default: True)
# Returns:
# - response: The generated response
# """
def get_response(query, vector_path=None, use_merged=False):
    blob_instance = Blob_storage_access()  # Blob storage access instance
    openai_instance = OpenaAI()  # OpenAI instance

    # Check if vector_path is provided
    if vector_path is not None:
        # Determine the path based on whether it is a merged vectorstore or not
        if use_merged:
            path = "vectors/merged_vectorstore.pkl"  # Path to the merged vectorstore

        else:
            path = vector_path  # Path to the specific vectorstore

        vectorstore = blob_instance.read_vector(path_on_datastore=path)  # Read the vectorstore from Blob storage
        response = openai_instance.generate_response(query=query, vectorstore=vectorstore)  # Generate response using the provided vectorstore

    else:
        response = openai_instance.generate_response(query=query)  # Generate response without using a specific vectorstore

    return response


# if __name__ == "__main__":
#     main()
