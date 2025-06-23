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
        self.detector_model = detector_model
        self.embed_model = AutoModel.from_pretrained(embed_model)
        self.tokenizer_path = tokenizer_path
        self.tokenizer = BertTokenizer.from_pretrained(self.tokenizer_path)
        self.text_splitter = text_splitter
        
        if detector_model_path:
            if not os.path.exists(detector_model_path):
                raise FileNotFoundError(f"Model file not found at {detector_model_path}")
            self.detector_model.load_state_dict(torch.load(detector_model_path, map_location='cpu'))

        if not text_splitter:
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=192, 
                chunk_overlap=50, 
                length_function=lambda x: len(x)
            )

        self.device = device if device else 'cuda' if torch.cuda.is_available() else 'cpu'
        self.detector_model.to(self.device)
        self.detector_model.eval()

        self.embed_model.to(self.device)
        self.embed_model.eval()

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
        Check if chunks have malicious content.
        """
        embeds = self.tokenizer(chunks, return_tensors='pt', padding=True, truncation=True, max_length=512).to(self.device)
        with torch.no_grad():
            outputs = self.detector_model(**embeds)
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

        for chunk, prob in zip(chunks, probs):
            if prob > 0.5:
                return chunk, prob

        return None, max(probs)

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

    def add(self, doc, file_name=None, malicious_check=True):
        """
        Add a new document to the RAG system.
        """

        if not isinstance(doc, str):
            raise ValueError("Document must be a string.")
        
        doc_id, chunk_data = self.__chunk_text(doc, file_name)

        chunk_texts = [chunk['text'] for chunk in chunk_data.values()]

        # Perform malicious content check if enabled
        if malicious_check:
            malicious_string, malicious_score = self.__malicious_check(chunk_texts)

            if malicious_string:
                raise ValueError(f"Document contains malicious content: '{malicious_string}' with score {malicious_score:.4f}")
        
    
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

    def query(self, query, top_documents=5, malicious_check=True):
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

        # check for malicious content in the top results if enabled
        if malicious_check:
            for (doc_id, chunk_id), score in top_results:
                chunk = self.data[doc_id]['chunks'][chunk_id]
                malicious_string, malicious_score = self.__malicious_check([chunk['text']])
                if malicious_string:
                    raise ValueError(f"Query result contains malicious content: '{malicious_string}' with score {malicious_score:.4f}")

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

    def get_document(self, doc_id):
        """
        Retrieve a document by its ID.
        """
        if doc_id not in self.data:
            raise ValueError(f"Document with ID {doc_id} not found.")
        
        for chunk_id, chunk in self.data[doc_id]['chunks'].items():
            chunk.pop('embedding', None)

        return self.data[doc_id]
    
    def delete_document(self, doc_id):
        """
        Delete a document by its ID.
        """
        if doc_id not in self.data:
            raise ValueError(f"Document with ID {doc_id} not found.")
        
        del self.data[doc_id]
        self.__save_data(self.data)
        return f"Document {doc_id} deleted successfully."
