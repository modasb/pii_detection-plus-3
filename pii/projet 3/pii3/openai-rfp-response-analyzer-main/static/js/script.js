document.addEventListener('DOMContentLoaded', function() {
    const documentForm = document.getElementById('document-form');
    const processBtn = document.getElementById('processBtn');
    const processingSpinner = document.getElementById('processingSpinner');
    const processingStatus = document.getElementById('processingStatus');
    const generateReportBtn = document.getElementById('generateReportBtn');
    const chatForm = document.getElementById('chatForm');
    const chatInput = document.getElementById('chat-input');
    const chatMessages = document.getElementById('chat-messages');

    // Get CSRF token from cookie
    function getCookie(name) {
        let cookieValue = null;
        if (document.cookie && document.cookie !== '') {
            const cookies = document.cookie.split(';');
            for (let i = 0; i < cookies.length; i++) {
                const cookie = cookies[i].trim();
                if (cookie.substring(0, name.length + 1) === (name + '=')) {
                    cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                    break;
                }
            }
        }
        return cookieValue;
    }

    // Process documents
    processBtn.addEventListener('click', async function() {
        const formData = new FormData(documentForm);
        
        // Show loading state
        processBtn.disabled = true;
        processingSpinner.classList.remove('hidden');
        processingStatus.textContent = 'Processing documents...';
        processingStatus.classList.remove('hidden');
        
        try {
            const response = await fetch('/analyzer/process-files/', {
                method: 'POST',
                body: formData,
                headers: {
                    'X-CSRFToken': getCookie('csrftoken')
                }
            });
            
            const data = await response.json();
            
            if (response.ok) {
                processingStatus.textContent = 'Documents processed successfully!';
                processingStatus.classList.remove('bg-blue-50', 'text-blue-700');
                processingStatus.classList.add('bg-green-50', 'text-green-700');
                generateReportBtn.classList.remove('hidden');
            } else {
                throw new Error(data.error || 'Failed to process documents');
            }
        } catch (error) {
            processingStatus.textContent = `Error: ${error.message}`;
            processingStatus.classList.remove('bg-blue-50', 'text-blue-700');
            processingStatus.classList.add('bg-red-50', 'text-red-700');
        } finally {
            processBtn.disabled = false;
            processingSpinner.classList.add('hidden');
        }
    });

    // Generate report
    generateReportBtn.addEventListener('click', async function() {
        try {
            const response = await fetch('/analyzer/generate-analysis/', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-CSRFToken': getCookie('csrftoken')
                },
                body: JSON.stringify({
                    rfp_text: document.getElementById('rfp-text').value,
                    response_text: document.getElementById('response-text').value
                })
            });
            
            const data = await response.json();
            
            if (response.ok) {
                document.getElementById('report').innerHTML = data.report;
            } else {
                throw new Error(data.error || 'Failed to generate report');
            }
        } catch (error) {
            alert(`Error: ${error.message}`);
        }
    });

    // Handle chat messages
    async function sendChatMessage(event) {
        event.preventDefault();
        const message = chatInput.value.trim();
        if (!message) return;

        // Add user message to chat
        addMessageToChat('user', message);
        chatInput.value = '';

        try {
            const response = await fetch('/analyzer/chat/', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-CSRFToken': getCookie('csrftoken')
                },
                body: JSON.stringify({ message })
            });
            
            const data = await response.json();
            
            if (response.ok) {
                addMessageToChat('assistant', data.response);
            } else {
                throw new Error(data.error || 'Failed to get response');
            }
        } catch (error) {
            addMessageToChat('error', `Error: ${error.message}`);
        }
    }

    // Add message to chat
    function addMessageToChat(type, content) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `chat-message ${type}-message mb-4 p-4 rounded-lg ${
            type === 'user' ? 'bg-indigo-50 ml-auto' : 
            type === 'assistant' ? 'bg-gray-50' : 
            'bg-red-50'
        } max-w-[80%] ${type === 'user' ? 'ml-auto' : 'mr-auto'}`;
        
        messageDiv.textContent = content;
        chatMessages.appendChild(messageDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    // Clear chat
    window.clearChat = function() {
        chatMessages.innerHTML = '';
    };
});
