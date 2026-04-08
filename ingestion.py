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