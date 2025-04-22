from flask import Flask, render_template, request, jsonify
import os

app = Flask(__name__)

# Constants
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf'}

# Create upload folder
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_documents():
    if 'rfp' not in request.files or 'response' not in request.files:
        return jsonify({"error": "Both RFP and Response files are required"}), 400

    rfp_file = request.files['rfp']
    response_file = request.files['response']

    if not all(allowed_file(f.filename) for f in [rfp_file, response_file]):
        return jsonify({"error": "Only PDF files are allowed"}), 400

    try:
        # Save files
        rfp_path = os.path.join(UPLOAD_FOLDER, 'rfp.pdf')
        response_path = os.path.join(UPLOAD_FOLDER, 'response.pdf')
        
        rfp_file.save(rfp_path)
        response_file.save(response_path)

        return jsonify({
            "message": "Documents processed successfully"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/generate_report', methods=['POST'])
def generate_report():
    try:
        # Simple static report
        report = {
            "summary": "Analysis complete",
            "details": "Report generated successfully",
            "recommendations": "See findings below"
        }
        return jsonify({"report": report})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5001))
    app.run(debug=False, host='0.0.0.0', port=port) 