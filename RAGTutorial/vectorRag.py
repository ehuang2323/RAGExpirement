from sentence_transformers import SentenceTransformer
import pinecone
from pinecone import Pinecone
import os
from pinecone import ServerlessSpec, PodSpec
import pandas as pd
import time

# Initialize the embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')
index_name = "test"

####
use_severless = True

api_key = os.environ.get('d52a0a0f-20b9-4a19-be54-e64daeeacc17') or 'd52a0a0f-20b9-4a19-be54-e64daeeacc17'
pc = Pinecone(api_key=api_key)

if use_severless:
    spec = ServerlessSpec(cloud='aws', region= 'us-east-1')
else:
    spec = PodSpec(environment=environment)

index_name = 'test'

if index_name not in pc.list_indexes().names():
    dimensions = 384
    pc.create_index(
    name=index_name,
    dimension=dimensions,
    metric="cosine",
    spec=spec
    )

while not pc.describe_index(index_name).status['ready']:
    time.sleep(1)

index = pc.Index(index_name)


# Vectorize and index the knowledge base
knowledge_base = [
    'https://www.python.org/: Python is a high-level, interpreted programming language. It emphasizes code readability and simplicity. Python supports multiple programming paradigms, including object-oriented and functional programming.',
    'https://en.wikipedia.org/wiki/Earth: Earth has seven continents. It is a planet within the solar system, Earth can support organic life forms',
     "https://weather.com/: The weather is lovely today. It's so sunny outside!"
]
vectors = model.encode(knowledge_base)
index.upsert(vectors=zip(["doc_" + str(i) for i in range(len(vectors))], vectors))

# Query embedding and similarity search
query = input('Question: ')
query_vector = model.encode([query])
results = index.query(vector=query_vector.tolist(), top_k=1, include_metadata=True)

# Retrieve and process the results
try:
    retrieved_info = [knowledge_base[int(result['id'].split("_")[1])] for result in results['matches']]
    context = " ".join(retrieved_info)

    def generate_response(query, context):
        return f"Context: {context}"

    # Generate and return the response
    response = generate_response(query, context)
    if(context != " "):
        print(response)
    else:
        print("no match")
except:
    print('No matches found')