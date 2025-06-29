import logging
import json
from typing import List, Dict, Optional
from Bio import Entrez
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

logger = logging.getLogger(__name__)
Entrez.email = os.getenv("EMAIL_ID") # Replace or override with env var

class MedicalRetriever:
    def __init__(self, openai_api_key: str, pinecone_index=None, icd_mapper=None, knowledge_store=None):
        self.openai_client = OpenAI(api_key=openai_api_key)
        self.pinecone_index = pinecone_index
        self.icd_mapper = icd_mapper
        self.knowledge_store = knowledge_store

    def rag_lookup(self, symptoms: List[str], top_k: int = 5) -> List[Dict]:
        """Query Pinecone for similar diseases and map ICD-10 codes."""
        if not self.pinecone_index:
            return []

        query = "Symptoms: " + ", ".join(symptoms)
        try:
            emb_resp = self.openai_client.embeddings.create(
                model="text-embedding-ada-002", input=query
            )
            vec = emb_resp.data[0].embedding

            resp = self.pinecone_index.query(
                vector=vec,
                top_k=top_k,
                include_metadata=True
            )

            results = []
            for match in resp["matches"]:
                disease = match["metadata"]["disease"]
                icd10 = match["metadata"].get("icd10", "UNKNOWN")
                if icd10 in ("", "UNKNOWN") and self.icd_mapper:
                    icd10 = self.icd_mapper.get_codes(disease).get(disease, "Unknown")
                results.append({
                    "disease": disease,
                    "icd10": icd10,
                    "confidence": match["score"]
                })
            return results
        except Exception as e:
            logger.error(f"[Retriever] Pinecone RAG error: {e}")
            return []

    def fetch_pubmed_articles(self, query_terms: List[str], max_results: int = 10) -> List[str]:
        """Fetch abstracts from PubMed using Entrez."""
        query = " AND ".join(query_terms)
        try:
            handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results)
            record = Entrez.read(handle)
            ids = record['IdList']

            if not ids:
                return []

            handle = Entrez.efetch(db="pubmed", id=ids, rettype="abstract", retmode="text")
            articles = handle.read().split("\n\n\n")
            return [a.strip() for a in articles if a.strip()]
        except Exception as e:
            logger.error(f"[Retriever] PubMed error: {e}")
            return []

    def store_medical_knowledge(self, articles: List[str]):
        """Store chunks of articles in the knowledge base."""
        if not self.knowledge_store:
            return

        chunks = []
        metadatas = []
        ids = []

        for i, article in enumerate(articles):
            article_chunks = self._chunk_text(article)
            for j, chunk in enumerate(article_chunks):
                chunks.append(chunk)
                metadatas.append({"source": f"pubmed_{i}"})
                ids.append(f"doc_{i}_chunk_{j}")

        self.knowledge_store.add_texts(texts=chunks, metadatas=metadatas, ids=ids)

    def _chunk_text(self, text: str, chunk_size: int = 300) -> List[str]:
        """Break a long text into chunks for embedding."""
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = []
        current_length = 0

        for sentence in sentences:
            length = len(sentence.split())
            if current_length + length <= chunk_size:
                current_chunk.append(sentence)
                current_length += length
            else:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                current_chunk = [sentence]
                current_length = length

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks
