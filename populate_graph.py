import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_neo4j import Neo4jGraph
from langchain_experimental.graph_transformers import LLMGraphTransformer

load_dotenv()

print("Connecting to Knowledge Graph...")
graph = Neo4jGraph(
    url=os.environ["NEO4J_URI"],
    username=os.environ["NEO4J_USERNAME"],
    password=os.environ["NEO4J_PASSWORD"],
    database=os.environ.get("NEO4J_DATABASE", "neo4j")
)

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0)
llm_transformer = LLMGraphTransformer(llm=llm)

# Loading documents from a 'data' folder
print("Loading documents from ./data directory...")
# Data Ingestion
loader = DirectoryLoader("./data", glob="**/*.txt", loader_cls=TextLoader)
raw_documents = loader.load()

if not raw_documents:
    print("No documents found in ./data. Please add text files.")
    raise SystemExit(1)

# Text Splitting
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
documents = text_splitter.split_documents(raw_documents)

# Graph Transformation
print(f"Processing {len(documents)} chunks and extracting multi-hop relationships...")
graph_documents = llm_transformer.convert_to_graph_documents(documents)

#Data Injection
print("Extracted nodes and relationships. Pushing to Neo4j...")
graph.add_graph_documents(graph_documents)
print("Data successfully injected into Neo4j! The graph is alive.")
