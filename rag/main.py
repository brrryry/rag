from flask import Flask, request, jsonify
from ragsec import RAGSec

from transformers import BertForSequenceClassification, BertTokenizer
from langchain.text_splitter import RecursiveCharacterTextSplitter


app = Flask(__name__)
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=1)
ragsec = RAGSec(
    detector_model=model,  # Replace with your actual model instance
    detector_model_path="./bert_binary_classifier.pth",  # Path to your model file
    tokenizer_path="bert-base-uncased",
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=192, chunk_overlap=50),
    data_file="./ragsec_data.json",
    device="cpu"
)

@app.route('/add', methods=['POST'])
def add_document():
    """
    Endpoint to add a new document to the RAG system.
    Expects JSON payload with 'doc' and optional 'file_name'.
    """
    data = request.json
    doc = data.get('doc')
    file_name = data.get('file_name', None)
    malicious_check = data.get('malicious_check', True)

    if not doc:
        return jsonify({"error": "Document content is required."}), 400

    try:
        doc_id = ragsec.add(doc, file_name, malicious_check)
        return jsonify({"doc_id": doc_id}), 201
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

@app.route('/query', methods=['POST'])
def query_documents():
    """
    Endpoint to query the RAG system.
    Expects JSON payload with 'query' and optional 'top_documents'.
    """
    data = request.json
    query = data.get('query')
    top_documents = data.get('top_documents', 5)

    if not query:
        return jsonify({"error": "Query is required."}), 400

    try:
        results = ragsec.query(query, top_documents=top_documents)
        return jsonify(results), 200
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

@app.route('/document/<doc_id>', methods=['GET'])
def get_document(doc_id):
    """
    Endpoint to retrieve a document by its ID.
    """
    try:
        doc = ragsec.get_document(doc_id)
        return jsonify(doc), 200
    except ValueError as e:
        return jsonify({"error": str(e)}), 404
    
@app.route('/document/<doc_id>', methods=['DELETE'])
def delete_document(doc_id):
    """
    Endpoint to delete a document by its ID.
    """
    try:
        ragsec.delete_document(doc_id)
        return jsonify({"message": "Document deleted successfully."}), 200
    except ValueError as e:
        return jsonify({"error": str(e)}), 404


if __name__ == '__main__':
    app.run(debug=True)