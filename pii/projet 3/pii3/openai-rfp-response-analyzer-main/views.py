from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import os
import shutil
from django.views.decorators.http import require_http_methods
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
import logging
from logging.handlers import RotatingFileHandler
from llama_parse import LlamaParse
from openai import OpenAI
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.memory import ConversationBufferMemory
from langchain.tools import tool
from langchain.schema import Document
from typing import List, Dict, Any, Optional, Union
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from typing_extensions import Literal
import traceback
import sys
from functools import lru_cache
from django.conf import settings
import json

# Load environment variables
load_dotenv()

# Constants
TOP_K = 6
OUTPUT_FOLDER = 'parsed_pdfs'
FAISS_INDEX_FOLDER = 'faiss_index'
ALLOWED_EXTENSIONS = {'pdf'}

# Create necessary directories
for folder in [OUTPUT_FOLDER, FAISS_INDEX_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# Initialize clients
openai_api_key = os.getenv("OPENAI_API_KEY")
llama_cloud_api_key = os.getenv("LLAMA_CLOUD_API_KEY")

client = OpenAI(api_key=openai_api_key)

# Initialize LlamaParse
parser = LlamaParse(
    api_key=llama_cloud_api_key,
    api_result_type="markdown",
    use_vendor_multimodal_model=True,
    vendor_multimodal_model_name="anthropic-sonnet-3.5",
    num_workers=4,
    verbose=True,
    language="en"
)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def index(request):
    return render(request, 'index.html')

@csrf_exempt
@require_http_methods(["POST"])
def process_documents(request):
    if 'rfp' not in request.FILES or 'response' not in request.FILES:
        return JsonResponse({"error": "Both RFP and Response files are required"}, status=400)

    rfp_file = request.FILES['rfp']
    response_file = request.FILES['response']

    # Validate file extensions
    for file in [rfp_file, response_file]:
        if not file.name.lower().endswith('.pdf'):
            return JsonResponse({"error": "Only PDF files are allowed"}, status=400)

    # Save uploaded files temporarily
    rfp_path = os.path.join(settings.BASE_DIR, "temp_rfp.pdf")
    response_path = os.path.join(settings.BASE_DIR, "temp_response.pdf")

    try:
        with open(rfp_path, 'wb+') as destination:
            for chunk in rfp_file.chunks():
                destination.write(chunk)
        with open(response_path, 'wb+') as destination:
            for chunk in response_file.chunks():
                destination.write(chunk)

        processor = DocumentProcessor()
        rfp_result = processor.parse_pdf(rfp_path, "rfp_parsed")
        response_result = processor.parse_pdf(response_path, "response_parsed")

        return JsonResponse({
            "rfp_result": rfp_result,
            "response_result": response_result,
            "message": "Documents processed successfully"
        })
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)
    finally:
        # Clean up temporary files
        for path in [rfp_path, response_path]:
            if os.path.exists(path):
                os.remove(path)

@csrf_exempt
@require_http_methods(["POST"])
def generate_report(request):
    try:
        retriever = DocumentRetriever()
        rfp_retriever = retriever.initialize_retriever("rfp_parsed")
        response_retriever = retriever.initialize_retriever("response_parsed")

        if not rfp_retriever or not response_retriever:
            raise ValueError("Documents must be processed before generating a report")

        rfp_content = retriever.retrieve_documents("Retrieve all relevant RFP content.", rfp_retriever)
        response_content = retriever.retrieve_documents("Retrieve all relevant Response content.", response_retriever)

        analyzer = Analyzer()
        raw_analysis = analyzer.analyze_gap(f"RFP Content:\n{rfp_content}\n\nResponse Content:\n{response_content}")
        raw_insights = analyzer.generate_insights(f"RFP Content:\n{rfp_content}\n\nResponse Content:\n{response_content}")

        raw_report = f"""
        # RFP and Response Analysis Report

        ## Part 1: Gap Analysis
        {raw_analysis}

        ## Part 2: Detailed Insights
        {raw_insights}
        """

        formatter = ReportFormatter()
        formatted_report = formatter.format_report(raw_report)
        return JsonResponse({"structured_report": formatted_report})

    except ValueError as e:
        return JsonResponse({"error": str(e)}, status=400)
    except Exception as e:
        return JsonResponse({"error": "An internal error occurred"}, status=500)

@csrf_exempt
@require_http_methods(["POST"])
def chat(request):
    try:
        data = json.loads(request.body)
        query = data.get('query')
        if not query:
            raise ValueError("No query provided")

        agent = AgentTools.setup_agent()
        if not agent:
            raise ValueError("Failed to initialize the agent")

        # Retrieve relevant documents
        rfp_docs = AgentTools.retrieve_rfp_documents(query)
        response_docs = AgentTools.retrieve_response_documents(query)

        enhanced_query = f"""
        Considering the following document contents:

        RFP Documents:
        {rfp_docs}

        Response Documents:
        {response_docs}

        Please answer the following query:
        {query}
        """

        result = agent.run(input=enhanced_query)
        return JsonResponse({"response": result})

    except ValueError as e:
        return JsonResponse({"error": str(e)}, status=400)
    except Exception as e:
        return JsonResponse({"error": "An internal error occurred"}, status=500) 