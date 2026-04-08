import os
import ssl
import certifi
from dotenv import load_dotenv
from typing import Any, Dict, List

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore
from langchain_tavily import TavilyCrawl, TavilyExtract, TavilyMap

from mistralai.client import Mistral

from logger import (Colors, log_info, log_success, log_error, log_warning, log_header)

load_dotenv()

ssl_context = ssl.create_default_context(cafile=certifi.where())
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

mistra_api_key = os.getenv("MISTRAL_API_KEY")
mistra_model = "mistral-embed"
mistra_client = Mistral(api_key=mistra_api_key)

embeddings = mistra_client.embeddings.create(
    model=mistra_model,
    inputs=["Embed this sentence.", "As well as this one."],
    )

vectorstore = PineconeVectorStore(index_name="langchain-docs", embedding=embeddings)
tavily_extract = TavilyExtract()
tavily_map = TavilyMap(max_depth=5, max_breath=20, max_pages=1000)