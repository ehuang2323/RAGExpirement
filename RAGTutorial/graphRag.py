from py2neo import Graph
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

graph = Graph("bolt://44.204.218.109:7687", auth=("neo4j", "state-centimeter-crash"))

model_name = "t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def query_graph(query):
  # Construct a Cypher query based on the user's input
  cypher_query = f"""
  MATCH (e:Entity)-[r]-(n)
  WHERE e.name =~ '(?i).{query}.'
  RETURN e, r, n
  """
  # Execute the query and retrieve the relevant subgraphs
  results = graph.run(cypher_query).data()
  
  # Process the retrieved subgraphs
  context = ""
  for result in results:
      entity = result['e']['name']
      relationship = type(result['r']).__name__
      neighbor = result['n']['name']
      context += f"{entity} {relationship} {neighbor}. "
  
  return context

def generate_response(query):
  # Query the knowledge graph
  context = query_graph(query)