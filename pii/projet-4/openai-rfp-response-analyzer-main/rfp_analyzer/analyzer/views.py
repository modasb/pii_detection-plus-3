import json
from pathlib import Path
from django.conf import settings
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.core.files.storage import default_storage
from asgiref.sync import sync_to_async
from .models import Document, Analysis, ChatMessage
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from llama_parse import LlamaParse
from openai import OpenAI
import logging
from .services.pii_service import PIIServiceClient

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

# Initialize PII service client
pii_client = PIIServiceClient()

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
        
        # Update documents
        rfp_doc.processed = True
        rfp_doc.faiss_index_path = str(rfp_index_path)
        await sync_to_async(rfp_doc.save)()
        
        response_doc.processed = True
        response_doc.faiss_index_path = str(response_index_path)
        await sync_to_async(response_doc.save)()
        
        return JsonResponse({
            'message': 'Documents processed successfully',
            'rfp_id': rfp_doc.id,
            'response_id': response_doc.id
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
        
        # Check for PII in the query and context using async context manager
        async with PIIServiceClient() as pii_client:
            pii_result = await pii_client.detect_pii(query + "\n" + context)
        
        # Generate answer
        completion = await sync_to_async(openai_client.chat.completions.create)(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert in analyzing RFP documents and responses. Provide clear, concise answers."},
                {"role": "user", "content": f"Question: {query}\n\nContext:\n{context}"}
            ]
        )
        
        answer = completion.choices[0].message.content
        
        # Mask PII in the answer if detected
        masked_answer = answer
        if pii_result['has_pii']:
            # Get the entities from PII detection
            entities = pii_result.get('entities', [])
            # Sort entities by start position in reverse order to handle overlapping
            entities.sort(key=lambda x: x['start'], reverse=True)
            
            # Replace each PII entity with its type
            for entity in entities:
                start = entity['start']
                end = entity['end']
                pii_type = entity['type']
                masked_answer = masked_answer[:start] + f"[{pii_type}]" + masked_answer[end:]
        
        # Save chat message with original answer (for record keeping)
        await sync_to_async(ChatMessage.objects.create)(
            analysis=analysis,
            query=query,
            answer=answer,  # Store original answer
            pii_detected=pii_result['has_pii'],
            pii_risk_level=pii_result.get('risk_level', 'UNKNOWN')
        )
        
        return JsonResponse({
            'answer': masked_answer,  # Send masked answer to frontend
            'pii_detected': pii_result['has_pii'],
            'pii_risk_level': pii_result.get('risk_level', 'UNKNOWN'),
            'pii_recommendations': pii_result.get('recommendations', [])
        })
        
    except Document.DoesNotExist:
        return JsonResponse({
            'error': 'No processed documents found'
        }, status=404)
    except Exception as e:
        logger.error(f"Error in chat: {str(e)}")
        return JsonResponse({
            'error': str(e)
        }, status=500) 