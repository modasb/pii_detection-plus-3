from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import default_storage
from django.urls import reverse
from llama_parse import LlamaParse
from .faiss_service import FAISSService
import json
import os
from openai import OpenAI
from django.views.decorators.http import require_http_methods
import datetime

faiss_service = FAISSService()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Initialize LlamaParse
parser = LlamaParse(
    api_key=os.getenv('LLAMA_CLOUD_API_KEY'),
    api_result_type="markdown",
    use_vendor_multimodal_model=True,
    vendor_multimodal_model_name="anthropic-sonnet-3.5",
    num_workers=4,
    verbose=True,
    language="en",
    result_encoding="utf-8",
    max_retries=3,
    timeout=300
)

def generate_text_with_gpt(rfp_text, response_text):
    """
    Generate text using GPT based on the RFP and response.
    """
    try:
        # Prepare the prompt
        prompt = f"""Based on the following RFP and Response documents, generate a comprehensive analysis:

RFP Content:
{rfp_text}

Response Content:
{response_text}

Please provide:
1. A summary of the key points from both documents
2. Analysis of the alignment between RFP requirements and response
3. Any potential risks or concerns
4. Suggestions for improvement

Format the response in a clear, structured manner."""

        # Generate text using GPT
        response = client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {"role": "system", "content": "You are an expert RFP analyst."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=2000
        )

        return response.choices[0].message.content

    except Exception as e:
        print(f"Error generating text with GPT: {str(e)}")
        return None

@csrf_exempt
def process_documents(request):
    """Process uploaded RFP and Response documents."""
    if request.method != 'POST':
        return JsonResponse({'error': 'Method not allowed'}, status=405)
    
    try:
        # Get uploaded files
        rfp_file = request.FILES.get('rfp')
        response_file = request.FILES.get('response')
        
        if not rfp_file or not response_file:
            return JsonResponse({
                'error': 'Both RFP and Response files are required',
                'rfp_uploaded': bool(rfp_file),
                'response_uploaded': bool(response_file)
            }, status=400)
        
        # Validate file types
        if not rfp_file.name.lower().endswith('.pdf') or not response_file.name.lower().endswith('.pdf'):
            return JsonResponse({
                'error': 'Only PDF files are supported',
                'rfp_type': rfp_file.name.split('.')[-1].lower(),
                'response_type': response_file.name.split('.')[-1].lower()
            }, status=400)
        
        # Create directories if they don't exist
        os.makedirs('media/documents/rfp', exist_ok=True)
        os.makedirs('media/documents/response', exist_ok=True)
        
        # Save files
        rfp_path = os.path.join('media/documents/rfp', rfp_file.name)
        response_path = os.path.join('media/documents/response', response_file.name)
        
        with open(rfp_path, 'wb+') as destination:
            for chunk in rfp_file.chunks():
                destination.write(chunk)
        
        with open(response_path, 'wb+') as destination:
            for chunk in response_file.chunks():
                destination.write(chunk)
        
        # Extract text from PDF files using LlamaParse
        try:
            if not os.getenv('LLAMA_CLOUD_API_KEY'):
                return JsonResponse({
                    'error': 'LLAMA_CLOUD_API_KEY is not configured. Please check your environment variables.'
                }, status=500)
            
            print(f"Processing RFP file: {rfp_path}")
            print(f"Processing Response file: {response_path}")
            
            # Use LlamaParse to extract text from PDF files
            rfp_result = parser.load_data(rfp_path)
            response_result = parser.load_data(response_path)
            
            if not rfp_result or not response_result:
                return JsonResponse({
                    'error': 'Failed to parse PDF files. Please ensure the files are valid PDFs.'
                }, status=500)
            
            # Extract text from the results
            rfp_text = "\n\n".join([page.text for page in rfp_result])
            response_text = "\n\n".join([page.text for page in response_result])
            
            if not rfp_text or not response_text:
                return JsonResponse({
                    'error': 'Failed to extract text from PDF files. Please ensure the files are valid PDFs.'
                }, status=500)
            
            print("Successfully extracted text from PDF files")
                
        except Exception as e:
            print(f"Error extracting text from PDF files: {str(e)}")
            import traceback
            traceback.print_exc()  # This will print the full stack trace
            return JsonResponse({
                'error': f'Error extracting text from PDF files: {str(e)}'
            }, status=500)
        
        try:
            # Create FAISS index
            texts = [rfp_text, response_text]
            document_ids = ['rfp', 'response']
            faiss_service.create_index(texts, document_ids)
            
            # Save FAISS index
            os.makedirs('media/faiss_index', exist_ok=True)
            faiss_service.save_index('media/faiss_index')
            
            # Generate text using GPT
            if not os.getenv('OPENAI_API_KEY'):
                return JsonResponse({
                    'error': 'OPENAI_API_KEY is not configured. Please check your environment variables.'
                }, status=500)
                
            generated_text = generate_text_with_gpt(rfp_text, response_text)
            
            if not generated_text:
                return JsonResponse({
                    'error': 'Failed to generate text using GPT. Please try again.'
                }, status=500)
            
            # Store document data in session
            request.session['rfp_text'] = rfp_text
            request.session['response_text'] = response_text
            request.session['generated_text'] = generated_text
            
            return JsonResponse({
                'success': True,
                'message': 'Documents processed successfully',
                'generated_text': generated_text
            })
            
        except Exception as e:
            print(f"Error processing documents: {str(e)}")
            import traceback
            traceback.print_exc()
            return JsonResponse({
                'error': f'Error processing documents: {str(e)}'
            }, status=500)
            
    except Exception as e:
        print(f"Error handling file upload: {str(e)}")
        return JsonResponse({
            'error': f'Error handling file upload: {str(e)}'
        }, status=500)

def chat_view(request):
    """Render the chat interface."""
    return render(request, 'chat.html')

@require_http_methods(["POST"])
def chat_message(request):
    """Handle chat messages and return AI responses"""
    try:
        data = json.loads(request.body)
        message = data.get('message')
        chat_history = data.get('chat_history', [])
        
        # Get RFP and response text from session
        rfp_text = request.session.get('rfp_text', '')
        response_text = request.session.get('response_text', '')
        
        # Prepare the prompt with context
        prompt = f"""You are an AI assistant helping to analyze an RFP (Request for Proposal) and its response. 
        Here is the context:

        RFP Content:
        {rfp_text}

        Response Content:
        {response_text}

        Chat History:
        {json.dumps(chat_history, indent=2)}

        User Question: {message}

        Please provide a helpful response based on the RFP and response content. Focus on:
        1. Direct answers to the user's question
        2. Relevant information from both documents
        3. Clear and concise explanations
        4. Professional tone

        Response:"""

        # Get response from GPT
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an AI assistant helping to analyze RFP documents and responses."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1000
        )
        
        # Extract the response text
        ai_response = response.choices[0].message.content.strip()
        
        return JsonResponse({
            'response': ai_response,
            'status': 'success'
        })
        
    except Exception as e:
        print(f"Error in chat_message: {str(e)}")
        return JsonResponse({
            'error': str(e),
            'status': 'error'
        }, status=500)

@csrf_exempt
def regenerate_text(request):
    """Regenerate analysis text."""
    try:
        # Get stored document data
        rfp_text = request.session.get('rfp_text', '')
        response_text = request.session.get('response_text', '')
        
        if not rfp_text or not response_text:
            return JsonResponse({
                'error': 'No document data found in session. Please upload documents first.'
            }, status=400)
        
        # Generate new text
        generated_text = generate_text_with_gpt(rfp_text, response_text)
        
        if not generated_text:
            return JsonResponse({
                'error': 'Failed to regenerate text. Please try again.'
            }, status=500)
        
        # Update session
        request.session['generated_text'] = generated_text
        
        return JsonResponse({
            'success': True,
            'generated_text': generated_text
        })
        
    except Exception as e:
        print(f"Error regenerating text: {str(e)}")
        return JsonResponse({
            'error': str(e)
        }, status=500)

@csrf_exempt
def generate_report(request):
    """Generate a report based on the analysis."""
    if request.method != 'POST':
        return JsonResponse({'error': 'Method not allowed'}, status=405)
    
    try:
        # Get stored document data and analysis
        rfp_text = request.session.get('rfp_text', '')
        response_text = request.session.get('response_text', '')
        generated_text = request.session.get('generated_text', '')
        
        if not all([rfp_text, response_text, generated_text]):
            return JsonResponse({
                'error': 'Missing required data. Please process documents first.'
            }, status=400)
        
        # Prepare report data
        report = {
            'summary': generated_text,
            'timestamp': datetime.datetime.now().isoformat(),
            'document_info': {
                'rfp_length': len(rfp_text.split()),
                'response_length': len(response_text.split())
            }
        }
        
        return JsonResponse({
            'success': True,
            'report': report
        })
        
    except Exception as e:
        print(f"Error generating report: {str(e)}")
        return JsonResponse({
            'error': str(e)
        }, status=500) 