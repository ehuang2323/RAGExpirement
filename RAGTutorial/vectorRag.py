from sentence_transformers import SentenceTransformer
import pinecone
from pinecone import Pinecone

# Initialize the embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')
index_name = "test"
pc = Pinecone(api_key='YOUR_API_KEY')

# Connect to the Pinecone vector database
pc = Pinecone(api_key="9703970d-8243-4909-8d81-f7b9799fdd7b")
index = pc.Index("test")
index_info = pc.describe_index(index_name)
index_host = index_info['status']['host']
index = pinecone.Index(index_name, host=index_host)


# Vectorize and index the knowledge base
knowledge_base = [
    "Python is a high-level, interpreted programming language.",
    "It emphasizes code readability and simplicity.",
    "Python supports multiple programming paradigms, including object-oriented and functional programming.",
]
vectors = model.encode(knowledge_base)
index.upsert(vectors=zip(["doc_" + str(i) for i in range(len(vectors))], vectors))

# Query embedding and similarity search
query = "What is Python?"
query_vector = model.encode([query])[0]
results = index.query(query_vector, top_k=2, include_metadata=True)

# Retrieve and process the results
retrieved_info = [knowledge_base[int(result['id'].split("_")[1])] for result in results['matches']]
context = " ".join(retrieved_info)

# Generate and return the response
response = generate_response(query, context)
print(response)