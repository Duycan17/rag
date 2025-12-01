"""RAG Chain module using LangChain.

This module provides the RAG chain implementation for document-scoped
question answering using Gemini 2.5 Flash as the LLM.

Requirements: 6.1, 6.2, 6.3, 6.4
"""
from dataclasses import dataclass
from typing import Any, Dict, List
from uuid import UUID

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

from app.config import Settings
from app.db.vector_store import SupabaseVectorStore, SearchResult
from app.services.embedding_service import EmbeddingService


@dataclass
class RAGResponse:
    """Response from the RAG chain."""
    answer: str
    sources: List[Dict[str, Any]]
    has_context: bool


# Prompt template for context-aware responses
# Requirements: 6.3, 6.4
RAG_PROMPT_TEMPLATE = """You are a helpful assistant that answers questions based on the provided document context.

CONTEXT:
{context}

INSTRUCTIONS:
- Answer the question based ONLY on the provided context above.
- If the context does not contain enough information to answer the question, respond with: "I don't have enough information in the document to answer this question."
- Do not make up information or use knowledge outside of the provided context.
- Be concise and direct in your answers.
- If you quote from the context, indicate that you are doing so.

QUESTION: {question}

ANSWER:"""


class RAGChain:
    """RAG chain for document-scoped question answering.
    
    Uses LangChain's retrieval patterns with Gemini 2.5 Flash as the LLM.
    Implements document-scoped retrieval by filtering on document_id.
    
    Requirements:
    - 6.1: Use LangChain's retrieval chain patterns
    - 6.2: Filter results by document_id for document-scoped queries
    - 6.3: Use prompt template for context-aware responses
    - 6.4: Handle no-context case appropriately
    """
    
    NO_CONTEXT_RESPONSE = "I don't have enough information in the document to answer this question."
    # Minimum similarity threshold for relevant results
    MIN_SIMILARITY_THRESHOLD = 0.3
    
    def __init__(self, settings: Settings) -> None:
        """Initialize the RAG chain.
        
        Args:
            settings: Application settings containing API keys and retrieval config
        """
        self._settings = settings
        self._retrieval_k = settings.retrieval_k
        
        # Initialize LLM - Gemini 2.0 Flash
        self._llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=settings.gemini_api_key,
            temperature=0.1,  # Low temperature for factual responses
        )
        
        # Initialize embedding service for query embedding
        self._embedding_service = EmbeddingService(settings)
        
        # Initialize vector store for retrieval
        self._vector_store = SupabaseVectorStore(settings)
        
        # Create prompt template
        self._prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
        
        # Output parser
        self._output_parser = StrOutputParser()

    def retrieve(
        self,
        query: str,
        document_id: UUID
    ) -> List[SearchResult]:
        """Retrieve relevant chunks for a query from a specific document.
        
        Args:
            query: The user's question
            document_id: UUID of the document to search within
            
        Returns:
            List of SearchResult objects with relevant chunks
            
        Requirements: 6.2, 6.4
        """
        # Generate embedding for the query
        query_embedding = self._embedding_service.generate_embedding(query)
        
        # Search within the specific document
        results = self._vector_store.similarity_search(
            query_embedding=query_embedding,
            document_id=document_id,
            k=self._retrieval_k
        )
        
        # Filter out low-relevance results (Requirement 6.4)
        # This ensures we don't use irrelevant context
        relevant_results = [
            r for r in results
            if r.similarity >= self.MIN_SIMILARITY_THRESHOLD
        ]
        
        return relevant_results
    
    def _format_context(self, search_results: List[SearchResult]) -> str:
        """Format search results into context string for the prompt.
        
        Args:
            search_results: List of search results from retrieval
            
        Returns:
            Formatted context string
        """
        if not search_results:
            return ""
        
        context_parts = []
        for i, result in enumerate(search_results, 1):
            context_parts.append(f"[Source {i}]\n{result.content}")
        
        return "\n\n".join(context_parts)
    
    def _format_sources(
        self,
        search_results: List[SearchResult]
    ) -> List[Dict[str, Any]]:
        """Format search results into source references.
        
        Args:
            search_results: List of search results from retrieval
            
        Returns:
            List of source dictionaries with content and metadata
        """
        return [
            {
                "content": result.content,
                "metadata": result.metadata
            }
            for result in search_results
        ]
    
    def invoke(
        self,
        question: str,
        document_id: UUID
    ) -> RAGResponse:
        """Invoke the RAG chain to answer a question.
        
        Retrieves relevant context from the document and generates
        a response using Gemini 2.5 Flash.
        
        Args:
            question: The user's question
            document_id: UUID of the document to query
            
        Returns:
            RAGResponse with answer, sources, and context flag
            
        Requirements: 6.1, 6.2, 6.3, 6.4
        """
        # Retrieve relevant chunks
        search_results = self.retrieve(question, document_id)
        
        # Handle no-context case (Requirement 6.4)
        if not search_results:
            return RAGResponse(
                answer=self.NO_CONTEXT_RESPONSE,
                sources=[],
                has_context=False
            )
        
        # Format context for the prompt
        context = self._format_context(search_results)
        
        # Build the chain: prompt -> llm -> output parser
        chain = self._prompt | self._llm | self._output_parser
        
        # Invoke the chain
        answer = chain.invoke({
            "context": context,
            "question": question
        })
        
        # Format sources for response
        sources = self._format_sources(search_results)
        
        return RAGResponse(
            answer=answer,
            sources=sources,
            has_context=True
        )
