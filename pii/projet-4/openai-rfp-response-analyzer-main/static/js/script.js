document.addEventListener('DOMContentLoaded', function() {
    const rfpFileInput = document.getElementById('rfpFile');
    const responseFileInput = document.getElementById('responseFile');
    const processBtn = document.getElementById('processBtn');
    const processingStatus = document.getElementById('processingStatus');
    const generateReportBtn = document.getElementById('generateReportBtn');
    const reportDiv = document.getElementById('report');
    const chatInput = document.getElementById('chat-input');
    const chatSendBtn = document.getElementById('chat-send');
    const chatMessages = document.getElementById('chat-messages');

    function updateFileLabel(input) {
        const label = input.nextElementSibling;
        label.textContent = input.files[0] ? input.files[0].name : 'Choose file';
    }

    rfpFileInput.addEventListener('change', () => updateFileLabel(rfpFileInput));
    responseFileInput.addEventListener('change', () => updateFileLabel(responseFileInput));

    processBtn.addEventListener('click', processDocuments);
    generateReportBtn.addEventListener('click', generateReport);
    chatSendBtn.addEventListener('click', sendChatMessage);
    chatInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            sendChatMessage();
        }
    });

    function processDocuments() {
        if (!rfpFileInput.files[0] || !responseFileInput.files[0]) {
            alert('Please upload both RFP and Response files.');
            return;
        }

        const formData = new FormData();
        formData.append('rfp', rfpFileInput.files[0]);
        formData.append('response', responseFileInput.files[0]);

        processingStatus.classList.remove('d-none');
        processingStatus.textContent = 'Processing documents...';
        processBtn.disabled = true;

        fetch('/api/process-documents/', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            processingStatus.textContent = data.message;
            processingStatus.classList.remove('alert-info');
            processingStatus.classList.add('alert-success');
        })
        .catch(error => {
            processingStatus.textContent = 'Error processing documents: ' + error.message;
            processingStatus.classList.remove('alert-info');
            processingStatus.classList.add('alert-danger');
        })
        .finally(() => {
            processBtn.disabled = false;
        });
    }

    function generateReport() {
        reportDiv.innerHTML = '<p>Generating report...</p>';

        fetch('/api/generate-report/', {
            method: 'POST'
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                throw new Error(data.error);
            }
            reportDiv.innerHTML = data.structured_report;
        })
        .catch(error => {
            reportDiv.innerHTML = '<p class="text-danger">Error generating report: ' + error.message + '</p>';
        });
    }

    function sendChatMessage() {
        const message = chatInput.value.trim();
        if (message) {
            addChatMessage('user', message);
            chatInput.value = '';

            fetch('/api/chat/', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    query: message
                })
            })
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    throw new Error(data.error);
                }
                addChatMessage('ai', data.response);
            })
            .catch(error => {
                addChatMessage('ai', 'Error: ' + error.message);
            });
        }
    }

    function addChatMessage(sender, message) {
        const messageDiv = document.createElement('div');
        messageDiv.classList.add('chat-message', sender === 'user' ? 'user-message' : 'ai-message');
        messageDiv.textContent = message;
        chatMessages.appendChild(messageDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
});
