from transformers import BertTokenizer, BertForSequenceClassification, AutoModel
import numpy as np
import os
import re
import uuid
import torch
import json

from langchain.text_splitter import RecursiveCharacterTextSplitter

class RAGSec:
    def __init__(self, 
    detector_model="bert-base-uncased", 
    detector_model_path=None, 
    embed_model="BAAI/bge-base-en-v1.5",
    tokenizer_path="bert-base-uncased", 
    text_splitter=None,
    data_file = "data/ragsec_data.json",
    device=None):
        self.model = model
        self.embed_model = AutoModel.from_pretrained(embed_model)
        self.tokenizer_path = tokenizer_path
        self.tokenizer = BertTokenizer.from_pretrained(self.tokenizer_path)
        self.text_splitter = text_splitter
        
        if detector_model_path:
            if not os.path.exists(detector_model_path):
                raise FileNotFoundError(f"Model file not found at {detector_model_path}")
            self.model.load_state_dict(torch.load(detector_model_path, map_location='cpu'))

        if not text_splitter:
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=192, 
                chunk_overlap=50, 
                length_function=lambda x: len(x)
            )

        self.device = device if device else 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        self.model.eval()

        self.data_file = data_file
        self.data = self.__load_data()

    def __load_data(self):
        """
        Load existing data from the JSON file.
        """
        if os.path.exists(self.data_file):
            with open(self.data_file, 'r') as f:
                return json.load(f)
        return {}

    def __save_data(self, data):
        """
        Save the data to the JSON file.
        """
        if not os.path.exists(os.path.dirname(self.data_file)):
            os.makedirs(os.path.dirname(self.data_file))
            
        with open(self.data_file, 'w') as f:
            json.dump(data, f, indent=4)

    def __malicious_check(self, chunks):
        """
        Check if a chunk has malicious content.
        """
        embeds = self.tokenizer(chunks, return_tensors='pt', padding=True, truncation=True, max_length=512).to(self.device)
        with torch.no_grad():
            outputs = self.model(**embeds)
            logits = outputs.logits.squeeze().cpu().numpy()
        
        predictions = torch.sigmoid(torch.tensor(logits)).numpy()

        # If predictions is a single value, convert it to a list
        if len(predictions.shape) == 0:
            predictions = [predictions.item()]
        else:
            predictions = predictions.tolist()

        probs = []
        for chunk, pred in zip(chunks, predictions):
            probs.append(pred)

        return max(probs)

    def __chunk_text(self, doc, file_name=None):
        """
        Take in a document and return chunk data.
        """
        if not isinstance(doc, str):
            raise ValueError("Document must be a string.")
        
        # Split the document into chunks
        doc_id = str(uuid.uuid4())
        docs = self.text_splitter.create_documents([doc])
        chunks = [ch.page_content for ch in docs]

        chunk_data = {}

        for chunk in chunks:
            chunk = re.sub(r'\s+', ' ', chunk)  # Normalize whitespace
            chunk_id = str(uuid.uuid4())
            tokens = self.tokenizer(chunk, return_tensors='pt', padding=True, truncation=True, max_length=512).to(self.device)

            with torch.no_grad():
                embedding = self.embed_model(**tokens).last_hidden_state.mean(dim=1).squeeze().cpu().tolist()
            
            chunk_data[chunk_id] = {
                "text": chunk,
                "file_name": file_name,
                "doc_id": doc_id,
                "embedding": embedding  # Store the embedding tensor
            }
        
        return doc_id, chunk_data

    def add(self, doc, file_name=None):
        """
        Add a new document to the RAG system.
        """

        if not isinstance(doc, str):
            raise ValueError("Document must be a string.")
        
        doc_id, chunk_data = self.__chunk_text(doc, file_name)

        chunk_texts = [chunk['text'] for chunk in chunk_data.values()]
        malicious_score = self.__malicious_check(chunk_texts)

        if malicious_score > 0.5:
            raise ValueError(f"Document contains potentially malicious content with score {malicious_score:.4f}.")
        
    
        embeddings = {}
        for chunk_id, chunk in chunk_data.items():
            embedding = self.tokenizer(chunk['text'], return_tensors='pt', padding=True, truncation=True, max_length=512).to(self.device)
            embeddings[chunk_id] = {
                "text": chunk['text'],
                "file_name": chunk['file_name'],
                "doc_id": doc_id,
                "embedding": chunk["embedding"]  # Assuming embeds is a dict with chunk_id keys
            }

        self.data[doc_id] = {
            "file_name": file_name,
            "chunks": embeddings
        }
        
        self.__save_data(self.data)
        return doc_id

    def query(self, query, top_documents=5):
        """
        Query the RAG system.
        """
        if not isinstance(query, str):
            raise ValueError("Query must be a string.")
        
        # get embeddings for the query
        query_tokens = self.tokenizer(query, return_tensors='pt', padding=True, truncation=True, max_length=512).to(self.device)
        with torch.no_grad():
            query_embedding = self.embed_model(**query_tokens).last_hidden_state.mean(dim=1).squeeze().cpu().tolist()
        
        # Calculate similarity with stored embeddings
        similarities = {}
        for doc_id, doc_data in self.data.items():
            for chunk_id, chunk in doc_data['chunks'].items():
                embedding = chunk['embedding']
                similarity = np.dot(query_embedding, embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(embedding))
                similarities[(doc_id, chunk_id)] = similarity
        
        # Sort by similarity and get top documents
        sorted_similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        top_results = sorted_similarities[:top_documents]
        results = []
        for (doc_id, chunk_id), score in top_results:
            chunk = self.data[doc_id]['chunks'][chunk_id]
            results.append({
                "doc_id": doc_id,
                "chunk_id": chunk_id,
                "text": chunk['text'],
                "file_name": chunk['file_name'],
                "similarity_score": score
            })
        
        return results


model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=1)
ragsec = RAGSec(
    detector_model=model,  # Replace with your actual model instance
    detector_model_path="../bert_binary_classifier.pth",  # Path to your model file
    tokenizer_path="bert-base-uncased",
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=192, chunk_overlap=50),
    data_file="./ragsec_data.json",
    device="cpu"
)

# Get all nested .txt files in the "data" directory
def get_all_txt_files(directory):
    """
    Get all .txt files in the specified directory and its subdirectories.
    """
    txt_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.txt'):
                txt_files.append(os.path.join(root, file))
    return txt_files

# Example usage: Get all .txt files and add them to the RAG system

"""
txt_files = get_all_txt_files("data")
for file_path in txt_files:
    with open(file_path, 'r', encoding='utf-8') as f:
        file_name = os.path.basename(file_path)
        content = f.read()
    
    try:
        doc_id = ragsec.add(file_name + " " + content, file_name=file_path)
        print(f"Document added with ID: {doc_id}")
    except ValueError as e:
        print(f"Error adding document {file_path}: {e}")
"""

"""
# Example query
res = ragsec.query("Who was president Marlowe?")
for r in res:
    print(f"Doc ID: {r['doc_id']}, Chunk ID: {r['chunk_id']}, Similarity Score: {r['similarity_score']:.4f}")
    print(f"Text: {r['text']}\n")
"""
