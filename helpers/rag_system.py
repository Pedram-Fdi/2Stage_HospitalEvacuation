import os
from typing import List
from langchain.schema import HumanMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
# Use the newer import for better compatibility
from langchain_openai import AzureOpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
import numpy as np

class RAGSystem:
    def __init__(self, llm, docs_folder="docs"):
        self.llm = llm
        self.docs_folder = docs_folder
        self.vector_store = None
        self.documents = []

        # Robust AzureOpenAIEmbeddings initialization
        try:
            # Try with stable API version first
            print("Attempting to initialize embeddings with stable API version...")
            self.embeddings_model = AzureOpenAIEmbeddings(
                azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
                azure_endpoint=os.getenv("AZURE_OPENAI_EMBEDDING_ENDPOINT"),
                api_key=os.getenv("AZURE_OPENAI_EMBEDDING_KEY"),
                api_version=os.getenv("AZURE_OPENAI_API_EMBEDDING_VERSION"),
                chunk_size=1000
            )
            print("✅ Embeddings initialized successfully with stable API version!")
        except Exception as e1:
            print(f"Failed with stable API version: {e1}")
            try:
                # Try with your current API version
                print("Attempting to initialize embeddings with current API version...")
                self.embeddings_model = AzureOpenAIEmbeddings(
                    azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
                    azure_endpoint=os.getenv("AZURE_OPENAI_EMBEDDING_ENDPOINT"),
                    api_key=os.getenv("AZURE_OPENAI_EMBEDDING_KEY"),
                    api_version=os.getenv("AZURE_OPENAI_API_EMBEDDING_VERSION"),
                    chunk_size=1000
                )
                print("✅ Embeddings initialized successfully with current API version!")
            except Exception as e2:
                print(f"Failed with current API version: {e2}")
                try:
                    # Try with minimal parameters
                    print("Attempting to initialize embeddings with minimal parameters...")
                    self.embeddings_model = AzureOpenAIEmbeddings(
                        deployment_name=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
                        openai_api_base=os.getenv("AZURE_OPENAI_ENDPOINT"),
                        openai_api_key=os.getenv("AZURE_OPENAI_KEY"),
                        openai_api_version="2023-12-01-preview",
                        chunk_size=1000
                    )
                    print("✅ Embeddings initialized successfully with legacy parameters!")
                except Exception as e3:
                    print(f"All embedding initialization attempts failed:")
                    print(f"  Stable API: {e1}")
                    print(f"  Current API: {e2}")
                    print(f"  Legacy params: {e3}")
                    print("Will use fallback text search instead.")
                    self.embeddings_model = None

        self.initialize_rag()
    
    def initialize_rag(self):
        """Initialize the RAG system by loading and indexing documents"""
        try:
            doc_files = ["overview_Rag.md", "introduction_Rag.md", "solution_Rag.md"]
            content = ""

            for file_name in doc_files:
                file_path = os.path.join(self.docs_folder, file_name)
                if os.path.exists(file_path):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content += f.read() + "\n\n"
                else:
                    print(f"Warning: {file_name} not found in {self.docs_folder}")

            if not content.strip():
                print("Warning: No valid documentation files were found. RAG system disabled.")
                return

            print(f"Document content length: {len(content)} characters")
            print(f"First 200 characters: {content[:200]}...")

            # Split the document into chunks with better markdown awareness
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=800,
                chunk_overlap=100,
                separators=[
                    "\n## ", "\n### ", "\n#### ",
                    "\n\n", "\n", ". ", "! ", "? ",
                    ", ", " ", ""
                ]
            )
            chunks = text_splitter.split_text(content)
            self.documents = chunks

            print(f"Document split into {len(chunks)} chunks")
            for i, chunk in enumerate(chunks[:3]):  # Preview first 3 chunks
                print(f"Chunk {i}: {chunk[:100]}...")

            docs = [Document(page_content=chunk) for chunk in chunks]

            if self.embeddings_model:
                try:
                    print("Testing embeddings with a simple query...")
                    test_embedding = self.embeddings_model.embed_query("test")
                    print(f"✅ Embeddings test successful! Vector dimension: {len(test_embedding)}")

                    self.vector_store = FAISS.from_documents(docs, self.embeddings_model)
                    print(f"✅ RAG system initialized with {len(docs)} document chunks using vector store.")
                except Exception as embedding_error:
                    print(f"❌ Failed to create FAISS index with embeddings: {embedding_error}")
                    print("Will use fallback text search instead.")
                    self.vector_store = None
            else:
                print("❌ Embeddings model is None, using fallback text search.")
                self.vector_store = None

        except Exception as e:
            print(f"Error initializing RAG system: {e}")
            self.vector_store = None


    def is_informational_query(self, user_message: str) -> bool:
        """Determine if the user message is asking for information rather than wanting to run the model"""
        informational_keywords = [
            "what", "how", "why", "when", "where", "which", "who",
            "tell me", "explain", "describe", "difference", "compare",
            "information", "info", "details", "help me understand",
            "stochastic model", "deterministic model", "2stage model", "average model",
            "mip solver", "alns", "scenarios", "clustering", "reinforcement learning",
            "can you", "could you", "would you", "do you know", "i want to know",
            "i need to understand", "clarify", "confused about"
        ]
        extraction_keywords = [
            "region", "city", "disaster", "affected", "evacuation", "hospital",
            "patients", "casualties", "emergency", "crisis", "toronto", "montreal",
            "solve", "run", "execute", "optimize", "plan", "need help with planning"
        ]
        user_lower = user_message.lower().strip()
        has_question_word = any(word in user_lower for word in informational_keywords[:7])
        has_info_request = any(phrase in user_lower for phrase in informational_keywords[7:])
        has_extraction_intent = any(word in user_lower for word in extraction_keywords)
        if len(user_message.split()) <= 5 and has_extraction_intent:
            return False
        if (has_question_word or has_info_request) and has_extraction_intent:
            if any(user_lower.startswith(word) for word in informational_keywords[:7]):
                return True
            if "tell me" in user_lower or "explain" in user_lower:
                return True
            return False
        return has_question_word or has_info_request

    def simple_text_search(self, query: str, top_k: int = 3) -> List[str]:
        """Fallback text search when vector search is not available"""
        if not self.documents:
            return []
        
        query_lower = query.lower()
        query_words = query_lower.split()
        
        # Score each document based on keyword matches
        scored_docs = []
        for doc in self.documents:
            doc_lower = doc.lower()
            score = 0
            
            # Exact phrase match gets high score
            if query_lower in doc_lower:
                score += 10
            
            # Individual word matches
            for word in query_words:
                if word in doc_lower:
                    score += doc_lower.count(word)
            
            if score > 0:
                scored_docs.append((score, doc))
        
        # Sort by score and return top results
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        return [doc for score, doc in scored_docs[:top_k]]

    def semantic_similarity_search(self, query: str, top_k: int = 3) -> List[str]:
        """Use FAISS vector search to retrieve most relevant document chunks"""
        if not self.vector_store:
            print("Vector search not available, using simple text search")
            return self.simple_text_search(query, top_k)
        
        print(f"Searching for: '{query}'")
        results = self.vector_store.similarity_search(query, k=top_k)
        retrieved_docs = [doc.page_content for doc in results]
        
        print(f"Retrieved {len(retrieved_docs)} documents:")
        for i, doc in enumerate(retrieved_docs):
            print(f"Doc {i}: {doc[:150]}...")
        
        return retrieved_docs

    def answer_informational_query(self, user_message: str) -> str:
        """Answer informational queries using RAG"""
        if not self.vector_store:
            return "I don't have enough information to answer that question. Please refer to the documentation or ask about running the optimization model."

        # Try multiple search approaches for better retrieval
        relevant_docs = []
        
        # 1. Direct search
        direct_results = self.semantic_similarity_search(user_message, top_k=3)
        relevant_docs.extend(direct_results)
        
        # 2. Search with expanded keywords for stochastic model
        if "stochastic" in user_message.lower():
            expanded_query = "stochastic model two-stage uncertainty scenarios"
            expanded_results = self.semantic_similarity_search(expanded_query, top_k=2)
            relevant_docs.extend(expanded_results)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_docs = []
        for doc in relevant_docs:
            if doc not in seen:
                seen.add(doc)
                unique_docs.append(doc)
        
        relevant_docs = unique_docs[:5]  # Limit to top 5 unique documents
        
        if not relevant_docs:
            return "I don't have enough information to answer that question. Please refer to the documentation or ask about running the optimization model."

        context = "\n\n".join(relevant_docs)
        print(f"Final context length: {len(context)} characters")

        prompt = f"""Based on the following documentation about disaster planning and optimization models, please answer the user's question. The documentation contains information about different model types including stochastic and deterministic models.

Documentation:
{context}

User Question: {user_message}

Please provide a helpful and accurate answer based on the information in the documentation. Be specific and detailed when explaining technical concepts."""

        try:
            response = self.llm([HumanMessage(content=prompt)])
            answer = response.content.strip()
            
            # Be less restrictive about "not enough information" responses
            if len(answer) < 50 or any(phrase in answer.lower() for phrase in [
                "don't have enough information", 
                "not enough information",
                "insufficient information",
                "cannot answer based on the documentation",
                "not mentioned in the documentation"
            ]):
                return f"Based on the available documentation: {answer}"
            
            return answer
        except Exception as e:
            print(f"Error in LLM call: {e}")
            return "I don't have enough information to answer that question. Please refer to the documentation or ask about running the optimization model."