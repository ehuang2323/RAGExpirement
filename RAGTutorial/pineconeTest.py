import os
from pinecone import Pinecone
from pinecone import ServerlessSpec, PodSpec
import pandas as pd
import time

use_severless = True

api_key = os.environ.get('9703970d-8243-4909-8d81-f7b9799fdd7b') or '9703970d-8243-4909-8d81-f7b9799fdd7b'
pc = Pinecone(api_key=api_key)

if use_severless:
    spec = ServerlessSpec(cloud='aws', region= 'us-east-1')
else:
    spec = PodSpec(environment=environment)

index_name = 'hello-pinecone'

if index_name in pc.list_indexes().names():
    pc.delete_index(index_name)

dimensions = 3
pc.create_index(
    name=index_name,
    dimension=dimensions,
    metric="cosine",
    spec=spec
)

while not pc.describe_index(index_name).status['ready']:
    time.sleep(1)

index = pc.Index(index_name)

df = pd.DataFrame(
    data={
        "id": ["A", "B"],
        "vector": [[1., 1., 1.], [1., 2., 3.]]
    }
)

index.upsert(vectors=zip(df.id, df.vector))

index.describe_index_stats()

index.query(
    vector=[2., 2., 2.],
    top_k=5,
    include_values=True
)

pc.delete_index(index_name)