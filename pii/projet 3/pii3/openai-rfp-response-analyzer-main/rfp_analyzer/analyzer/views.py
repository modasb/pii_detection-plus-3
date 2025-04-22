import json
from pathlib import Path
from django.conf import settings
from django.http import JsonResponse
from django.shortcuts import render, redirect
from django.urls import reverse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.core.files.storage import default_storage
from django.contrib import messages
from asgiref.sync import sync_to_async
from .models import Document, Analysis, ChatMessage
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from llama_parse import LlamaParse
from openai import OpenAI
from pii_service.pii_detector import PIIDetector
import logging

logger = logging.getLogger(__name__)

# Initialize clients
parser = LlamaParse(
    api_key=settings.LLAMA_CLOUD_API_KEY,
    api_result_type="markdown",
    use_vendor_multimodal_model=True,
    vendor_multimodal_model_name="anthropic-sonnet-3.5",
    num_workers=4,
    verbose=True,
    language="en"
)

openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)

async def index(request):
    return render(request, 'analyzer/index.html')

@csrf_exempt
@require_http_methods(["POST"])
async def process_documents(request):
    try:
        rfp_file = request.FILES.get('rfp')
        response_file = request.FILES.get('response')
        
        if not rfp_file or not response_file:
            return JsonResponse({
                'error': 'Both RFP and Response files are required'
            }, status=400)
            
        if not all(f.name.lower().endswith('.pdf') for f in [rfp_file, response_file]):
            return JsonResponse({
                'error': 'Only PDF files are allowed'
            }, status=400)

        # Save files
        rfp_path = default_storage.save(f'documents/rfp/{rfp_file.name}', rfp_file)
        response_path = default_storage.save(f'documents/response/{response_file.name}', response_file)
        
        # Process documents asynchronously
        rfp_doc = await sync_to_async(Document.objects.create)(
            title=rfp_file.name,
            document_type='RFP',
            file=rfp_path
        )
        
        response_doc = await sync_to_async(Document.objects.create)(
            title=response_file.name,
            document_type='RESPONSE',
            file=response_path
        )
        
        # Parse documents
        rfp_content = await sync_to_async(parser.load_data)(default_storage.path(rfp_path))
        response_content = await sync_to_async(parser.load_data)(default_storage.path(response_path))
        
        # Create FAISS indexes
        embeddings = OpenAIEmbeddings(openai_api_key=settings.OPENAI_API_KEY)
        
        # RFP index
        rfp_texts = [chunk.text for chunk in rfp_content]
        rfp_metadatas = [chunk.metadata for chunk in rfp_content]
        rfp_index = await sync_to_async(FAISS.from_texts)(
            texts=rfp_texts,
            embedding=embeddings,
            metadatas=rfp_metadatas
        )
        
        # Response index
        response_texts = [chunk.text for chunk in response_content]
        response_metadatas = [chunk.metadata for chunk in response_content]
        response_index = await sync_to_async(FAISS.from_texts)(
            texts=response_texts,
            embedding=embeddings,
            metadatas=response_metadatas
        )
        
        # Save indexes
        rfp_index_path = Path(settings.FAISS_INDEX_FOLDER) / f"rfp_{rfp_doc.id}"
        response_index_path = Path(settings.FAISS_INDEX_FOLDER) / f"response_{response_doc.id}"
        
        await sync_to_async(rfp_index.save_local)(str(rfp_index_path))
        await sync_to_async(response_index.save_local)(str(response_index_path))

        return JsonResponse({
            'message': 'Documents processed successfully!',
            'rfp_id': rfp_doc.id,
            'response_id': response_doc.id,
            'redirect_url': reverse('analyzer:chat_view')
        })

    except Exception as e:
        logger.error(f"Error processing documents: {str(e)}")
        return JsonResponse({
            'error': str(e)
        }, status=500)

@csrf_exempt
@require_http_methods(["POST"])
async def generate_report(request):
    try:
        # Get the latest processed documents
        rfp_doc = await sync_to_async(lambda: Document.objects.filter(
            document_type='RFP',
            processed=True
        ).latest('uploaded_at'))()
        
        response_doc = await sync_to_async(lambda: Document.objects.filter(
            document_type='RESPONSE',
            processed=True
        ).latest('uploaded_at'))()
        
        # Load FAISS indexes
        embeddings = OpenAIEmbeddings(openai_api_key=settings.OPENAI_API_KEY)
        rfp_index = await sync_to_async(FAISS.load_local)(
            rfp_doc.faiss_index_path,
            embeddings,
            allow_dangerous_deserialization=True
        )
        response_index = await sync_to_async(FAISS.load_local)(
            response_doc.faiss_index_path,
            embeddings,
            allow_dangerous_deserialization=True
        )
        
        # Generate report using OpenAI
        report_prompt = """
        Analyze the RFP requirements and the response document. Generate a structured report that includes:
        1. Executive Summary
        2. Key Requirements Analysis
        3. Response Completeness
        4. Gaps and Recommendations
        5. Overall Assessment
        
        Format the report in HTML with appropriate styling.
        """
        
        # Get relevant chunks from both documents
        rfp_chunks = await sync_to_async(rfp_index.similarity_search)(report_prompt, k=5)
        response_chunks = await sync_to_async(response_index.similarity_search)(report_prompt, k=5)
        
        # Combine contexts
        context = "\nRFP Content:\n" + "\n".join([chunk.page_content for chunk in rfp_chunks])
        context += "\nResponse Content:\n" + "\n".join([chunk.page_content for chunk in response_chunks])
        
        # Generate report
        completion = await sync_to_async(openai_client.chat.completions.create)(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert in analyzing RFP documents and responses. Generate a detailed, well-structured report."},
                {"role": "user", "content": f"{report_prompt}\n\nContext:\n{context}"}
            ]
        )
        
        report_content = completion.choices[0].message.content
        
        # Save analysis
        analysis = await sync_to_async(Analysis.objects.create)(
            rfp_document=rfp_doc,
            response_document=response_doc,
            report={'content': report_content}
        )
        
        return JsonResponse({
            'structured_report': report_content,
            'analysis_id': analysis.id
        })
        
    except Document.DoesNotExist:
        return JsonResponse({
            'error': 'No processed documents found'
        }, status=404)
    except Exception as e:
        logger.error(f"Error generating report: {str(e)}")
        return JsonResponse({
            'error': str(e)
        }, status=500)

@csrf_exempt
@require_http_methods(["POST"])
async def chat(request):
    try:
        data = json.loads(request.body)
        query = data.get('query')
        
        if not query:
            return JsonResponse({
                'error': 'Query is required'
            }, status=400)
        
        # Initialize PII detector
        pii_detector = PIIDetector()
        
        # Check query for PII
        query_pii_results = pii_detector.detect_pii(query)
        has_query_pii = len(query_pii_results.get('entities', [])) > 0
        
        # Get the latest analysis
        analysis = await sync_to_async(lambda: Analysis.objects.select_related(
            'rfp_document',
            'response_document'
        ).latest('created_at'))()
        
        # Load FAISS indexes
        embeddings = OpenAIEmbeddings(openai_api_key=settings.OPENAI_API_KEY)
        rfp_index = await sync_to_async(FAISS.load_local)(
            analysis.rfp_document.faiss_index_path,
            embeddings,
            allow_dangerous_deserialization=True
        )
        response_index = await sync_to_async(FAISS.load_local)(
            analysis.response_document.faiss_index_path,
            embeddings,
            allow_dangerous_deserialization=True
        )
        
        # Get relevant chunks from both documents
        rfp_chunks = await sync_to_async(rfp_index.similarity_search)(query, k=3)
        response_chunks = await sync_to_async(response_index.similarity_search)(query, k=3)
        
        # Combine contexts
        context = "\nRFP Content:\n" + "\n".join([chunk.page_content for chunk in rfp_chunks])
        context += "\nResponse Content:\n" + "\n".join([chunk.page_content for chunk in response_chunks])
        
        # Generate answer
        completion = await sync_to_async(openai_client.chat.completions.create)(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert in analyzing RFP documents and responses. Provide clear and concise answers based on the document content."},
                {"role": "user", "content": f"Question: {query}\n\nContext:\n{context}"}
            ]
        )
        
        answer = completion.choices[0].message.content
        
        # Check answer for PII
        answer_pii_results = pii_detector.detect_pii(answer)
        has_answer_pii = len(answer_pii_results.get('entities', [])) > 0
        
        # Save chat message
        chat_message = await sync_to_async(ChatMessage.objects.create)(
            analysis=analysis,
            query=query,
            response=answer,
            has_pii=has_query_pii or has_answer_pii
        )
        
        # Prepare response with PII information
        response_data = {
            'answer': answer,
            'pii_detected': {
                'in_query': has_query_pii,
                'in_answer': has_answer_pii,
                'query_entities': query_pii_results.get('entities', []) if has_query_pii else [],
                'answer_entities': answer_pii_results.get('entities', []) if has_answer_pii else [],
            }
        }
        
        # If PII is detected, include anonymized versions
        if has_query_pii or has_answer_pii:
            response_data['anonymized'] = {
                'query': query_pii_results.get('anonymized_text') if has_query_pii else query,
                'answer': answer_pii_results.get('anonymized_text') if has_answer_pii else answer
            }
        
        return JsonResponse(response_data)
        
    except Analysis.DoesNotExist:
        return JsonResponse({
            'error': 'No analysis found'
        }, status=404)
    except Exception as e:
        logger.error(f"Error in chat: {str(e)}")
        return JsonResponse({
            'error': str(e)
        }, status=500)

async def chat_view(request):
    """
    Render the chat interface template.
    """
    return render(request, 'analyzer/chat.html') 