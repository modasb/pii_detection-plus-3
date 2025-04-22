import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import os

class FAISSService:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.index = None
        self.texts = []
        self.document_ids = []
        
    def create_index(self, texts, document_ids=None):
        """
        Create a FAISS index from a list of texts.
        """
        # Generate embeddings
        embeddings = self.model.encode(texts)
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        
        # Add vectors to the index
        self.index.add(np.array(embeddings).astype('float32'))
        
        # Store texts and document IDs
        self.texts = texts
        self.document_ids = document_ids or [f"doc_{i}" for i in range(len(texts))]
        
    def search(self, query, k=5):
        """
        Search for similar texts using a query string.
        """
        if self.index is None:
            raise ValueError("Index not created. Call create_index first.")
            
        # Generate query embedding
        query_embedding = self.model.encode([query])
        
        # Search in the index
        distances, indices = self.index.search(np.array(query_embedding).astype('float32'), k)
        
        # Return results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.texts):  # Ensure valid index
                results.append({
                    'text': self.texts[idx],
                    'document_id': self.document_ids[idx],
                    'distance': float(distances[0][i])
                })
        
        return results
    
    def save_index(self, directory):
        """
        Save the FAISS index and associated data.
        """
        if self.index is None:
            raise ValueError("No index to save.")
            
        # Create directory if it doesn't exist
        os.makedirs(directory, exist_ok=True)
        
        # Save index
        faiss.write_index(self.index, os.path.join(directory, 'index.faiss'))
        
        # Save texts and document IDs
        with open(os.path.join(directory, 'texts.txt'), 'w', encoding='utf-8') as f:
            for text in self.texts:
                f.write(text + '\n')
                
        with open(os.path.join(directory, 'document_ids.txt'), 'w', encoding='utf-8') as f:
            for doc_id in self.document_ids:
                f.write(doc_id + '\n')
    
    def load_index(self, directory):
        """
        Load a saved FAISS index and associated data.
        """
        # Load index
        self.index = faiss.read_index(os.path.join(directory, 'index.faiss'))
        
        # Load texts
        with open(os.path.join(directory, 'texts.txt'), 'r', encoding='utf-8') as f:
            self.texts = [line.strip() for line in f.readlines()]
            
        # Load document IDs
        with open(os.path.join(directory, 'document_ids.txt'), 'r', encoding='utf-8') as f:
            self.document_ids = [line.strip() for line in f.readlines()] 