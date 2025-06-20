# install required packages and import
import os
import tqdm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertForSequenceClassification, BertTokenizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from langchain.text_splitter import RecursiveCharacterTextSplitter



def load_model(path="./bert_binary_classifier.pth"):
    """
    Load the pre-trained BERT model for binary classification.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found at {path}")

    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=1)
    model.load_state_dict(torch.load(path))
    model.eval()
    
    return model


def run_query_loop():
    # get time 
    print("Loading model and tokenizer...")

    start_time = pd.Timestamp.now() # get current time
    model = load_model()
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=192, chunk_overlap=50)
    end_time = pd.Timestamp.now() # get current time
    
    print(f"Model and tokenizer loaded in {end_time - start_time} seconds.")
    print("Starting query loop. Type 'exit' to quit.")

    while True:
        query = input("Enter your query: ")
        if query.lower() == "exit":
            print("Exiting query loop.")
            break

        # Split the query into chunks
        docs = text_splitter.create_documents([query])
        chunks = [ch.page_content for ch in docs]  # Extract the text from the Document objects

        inputs = tokenizer(chunks, return_tensors='pt', padding=True, truncation=True, max_length=512)

        # Run the model
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits.squeeze().numpy()
        
        # Apply sigmoid to get probabilities
        predictions = torch.sigmoid(torch.tensor(logits)).numpy()

        # if predictions is a single value, convert it to a list
        if len(predictions.shape) == 0:
            predictions = [predictions.item()]
        else:
            predictions = predictions.tolist()
     
        # Print results
        print("------------------------------------")
        chunks_probs = []
        for chunk, prob in zip(chunks, predictions):
            print(f"Chunk: {chunk}\nProbability of malicious instructions: {prob:.4f}\n")
            chunks_probs.append(prob)
        max_prob = max(chunks_probs)
        print(f"Overall (max) probability of malicious instructions: {max_prob:.4f}")
        print("------------------------------------\n")
        


if __name__ == "__main__":
    run_query_loop()
