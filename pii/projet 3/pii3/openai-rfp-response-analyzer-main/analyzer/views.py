from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
import json
import os
from .models import Document, Analysis
from .utils import process_documents, generate_report

def index(request):
    return render(request, 'analyzer/index.html')

@csrf_exempt
@require_http_methods(["POST"])
def process_files(request):
    try:
        rfp_file = request.FILES.get('rfp')
        response_file = request.FILES.get('response')
        
        if not rfp_file or not response_file:
            return JsonResponse({'error': 'Both RFP and Response files are required'}, status=400)
            
        # Save files temporarily
        rfp_path = os.path.join('temp', rfp_file.name)
        response_path = os.path.join('temp', response_file.name)
        
        os.makedirs('temp', exist_ok=True)
        
        with open(rfp_path, 'wb+') as destination:
            for chunk in rfp_file.chunks():
                destination.write(chunk)
                
        with open(response_path, 'wb+') as destination:
            for chunk in response_file.chunks():
                destination.write(chunk)
        
        # Process documents
        analysis_result = process_documents(rfp_path, response_path)
        
        # Clean up temporary files
        os.remove(rfp_path)
        os.remove(response_path)
        
        return JsonResponse(analysis_result)
        
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

@csrf_exempt
@require_http_methods(["POST"])
def generate_analysis(request):
    try:
        data = json.loads(request.body)
        rfp_text = data.get('rfp_text')
        response_text = data.get('response_text')
        
        if not rfp_text or not response_text:
            return JsonResponse({'error': 'Both RFP and Response text are required'}, status=400)
            
        report = generate_report(rfp_text, response_text)
        return JsonResponse({'report': report})
        
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

@csrf_exempt
@require_http_methods(["POST"])
def chat(request):
    try:
        data = json.loads(request.body)
        message = data.get('message')
        
        if not message:
            return JsonResponse({'error': 'Message is required'}, status=400)
            
        # Process chat message and generate response
        response = process_chat_message(message)
        return JsonResponse({'response': response})
        
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500) 