import pandas as pd
import csv
from dotenv import load_dotenv, dotenv_values
import os
from langchain.document_loaders import UnstructuredPDFLoader, OnlinePDFLoader, PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')


# data = pd.read_csv('/Users/eddie/VSC/RAGExpirement/RAGTutorial/chatgpt_reviews.csv')
# data = data.drop(columns=['reviewId','userName','at','appVersion','score','thumbsUpCount','reviewCreatedVersion'])
# print(data['content'].to_list())
# with open('/Users/eddie/VSC/RAGExpirement/RAGTutorial/chatgpt_reviews.csv') as file_obj: 
      
#     # Create reader object by passing the file  
#     # object to reader method 
#     reader_obj = csv.reader(file_obj) 
      
#     # Iterate over each row in the csv  
#     # file using reader object 
#     for row in reader_obj: 
#         print(row)
# hi = ''
# data = [[1,2],[3,4],[5,6]]
# hi = data

# print(len(hi))

load_dotenv()


# print(os.getenv('MY_KEY'))
loader = TextLoader(file_path='/Users/eddie/VSC/RAGExpirement/RAGTutorial/data/txt/vb.txt')
# cambridge = PyPDFLoader("/Users/eddie/VSC/RAGExpirement/RAGTutorial/PDF/Cambridge - Wikipedia.pdf")
# oxford = PyPDFLoader("/Users/eddie/VSC/RAGExpirement/RAGTutorial/PDF/Oxford - Wikipedia.pdf")
# earth = PyPDFLoader("/Users/eddie/VSC/RAGExpirement/RAGTutorial/PDF/Earth - Wikipedia.pdf")
# python = PyPDFLoader("/Users/eddie/VSC/RAGExpirement/RAGTutorial/PDF/Python (programming language) - Wikipedia.pdf")

data = loader.load()

# print (f'You have {len(data)} document(s) in your data')
# print (f'There are {len(data[0].page_content)} characters in your sample document')
# print (f'Here is a sample: {data[0].page_content[:200]}')

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
texts = text_splitter.split_documents(data)

string_text = [texts[i].page_content for i in range(len(texts))]

print(string_text[0])

# print (f'Now you have {len(texts)} documents')

####

# api_key = os.environ.get(os.getenv('MY_KEY')) or os.getenv('MY_KEY')
# pc = Pinecone(api_key=api_key)

# spec = ServerlessSpec(cloud='aws', region= 'us-east-1')
# index_name = 'cambridge'

# if index_name not in pc.list_indexes().names():
#     dimensions = 384
#     pc.create_index(
#     name=index_name,
#     dimension=dimensions,
#     metric="cosine",
#     spec=spec
#     )


# index = pc.Index(index_name)

# vectors = model.encode(texts)
# index.upsert(vectors=zip(["doc_" + str(i) for i in range(len(vectors))], vectors))

# # Query embedding and similarity search
# query = input('What university is Cambridge known for housing?')
# query_vector = model.encode([query])
# results = index.query(vector=query_vector.tolist(), top_k=1, include_metadata=True)

# # Retrieve and process the results
# try:
#     retrieved_info = [texts[int(result['id'].split("_")[1])] for result in results['matches']]
#     context = " ".join(retrieved_info)

#     def generate_response(query, context):
#         return f"Context: {context}"

#     # Generate and return the response
#     response = generate_response(query, context)
#     if(context != " "):
#         print(response)
#     else:
#         print("no match")
# except:
#     print('No matches found')

# OPENAI_API_KEY = os.getenv('OPENAI_Key')
# embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# from langchain.vectorstores import Chroma

# vectorstore = Chroma.from_documents(texts, embeddings)

# query = "Cambridge is well known as the city that is the home of which college?"
# docs = vectorstore.similarity_search(query)

# for doc in docs:
#     print (f"{doc.page_content}\n")


# # initialize pinecone
# pinecone_api_key = os.environ.get(os.getenv('MY_KEY')) or os.getenv('MY_KEY')
# pc = Pinecone(api_key=pinecone_api_key)

# index_name = 'langchain'
# spec = ServerlessSpec(cloud='aws', region= 'us-east-1')
# dimensions = 384

# if index_name not in pc.list_indexes().names():
#     dimensions = 384
#     pc.create_index(
#         name=index_name,
#         dimension=dimensions,
#         metric="cosine",
#         spec=spec
#     )

# index = pc.Index(index_name)

# docsearch = index.from_texts([t.page_content for t in texts], embeddings, index_name=index_name)