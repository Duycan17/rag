"""MCQ Chain module for generating multiple choice questions.

This module provides the MCQ chain implementation for generating
practice questions from document content using Gemini 2.5 Flash.

Requirements: 1.1, 2.1, 2.2, 2.3, 5.1, 5.3
"""
import json
import re
from dataclasses import dataclass
from typing import List
from uuid import UUID

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

from app.config import Settings
from app.db.vector_store import SupabaseVectorStore, SearchResult
from app.models.mcq_schemas import DifficultyLevel, MCQQuestion
from app.services.embedding_service import EmbeddingService


@dataclass
class MCQGenerationResult:
    """Result from MCQ generation chain."""
    questions: List[MCQQuestion]
    has_context: bool


class MCQGenerationError(Exception):
    """Exception raised when MCQ generation fails."""
    pass


# Prompt template for MCQ generation
# Requirements: 2.1, 2.2, 2.3
MCQ_PROMPT_TEMPLATE = """You are an expert educator creating multiple choice questions.

DOCUMENT CONTENT:
{context}

INSTRUCTIONS:
- Generate exactly {num_questions} multiple choice questions based on the document content above.
- Difficulty level: {difficulty}
  - easy: Basic recall and simple facts
  - medium: Understanding and application of concepts  
  - hard: Analysis, synthesis, or evaluation
- Each question must have exactly 4 options (A, B, C, D).
- Only ONE option should be correct.
- Distractors should be plausible but clearly incorrect.
- Include a brief explanation for why the correct answer is correct.

OUTPUT FORMAT (JSON):
{{
  "questions": [
    {{
      "question": "Question text here?",
      "options": ["Option A", "Option B", "Option C", "Option D"],
      "correct_answer_index": 0,
      "explanation": "Explanation of why this answer is correct."
    }}
  ]
}}

Generate the questions now:"""


class MCQChain:
    """MCQ chain for generating multiple choice questions from documents.
    
    Uses LangChain with Gemini 2.5 Flash to generate MCQs based on
    document content retrieved from the vector store.
    
    Requirements:
    - 1.1: Retrieve relevant document content and generate questions
    - 2.1: Generate easy questions testing basic recall
    - 2.2: Generate medium questions requiring understanding
    - 2.3: Generate hard questions requiring analysis
    - 5.1: Use existing vector store and embedding infrastructure
    - 5.3: Use Gemini 2.5 Flash for question generation
    """
    
    # Minimum similarity threshold for relevant results
    MIN_SIMILARITY_THRESHOLD = 0.3
    # Number of chunks to retrieve for context
    RETRIEVAL_K = 6
    
    def __init__(self, settings: Settings) -> None:
        """Initialize the MCQ chain.
        
        Args:
            settings: Application settings containing API keys
        """
        self._settings = settings
        
        # Initialize LLM - Gemini 2.0 Flash (Requirements: 5.3)
        self._llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=settings.gemini_api_key,
            temperature=0.7,  # Higher temperature for creative question generation
        )
        
        # Initialize embedding service for query embedding (Requirements: 5.1)
        self._embedding_service = EmbeddingService(settings)
        
        # Initialize vector store for retrieval (Requirements: 5.1)
        self._vector_store = SupabaseVectorStore(settings)
        
        # Create prompt template
        self._prompt = ChatPromptTemplate.from_template(MCQ_PROMPT_TEMPLATE)
        
        # Output parser
        self._output_parser = StrOutputParser()

    def retrieve(
        self,
        document_id: UUID,
        num_questions: int
    ) -> List[SearchResult]:
        """Retrieve relevant chunks from a document for MCQ generation.
        
        Args:
            document_id: UUID of the document to retrieve from
            num_questions: Number of questions to generate (affects retrieval count)
            
        Returns:
            List of SearchResult objects with relevant chunks
            
        Requirements: 5.1
        """
        # Generate a generic query embedding to retrieve diverse content
        query = "key concepts, facts, and important information"
        query_embedding = self._embedding_service.generate_embedding(query)
        
        # Retrieve more chunks for more questions
        k = max(self.RETRIEVAL_K, num_questions * 2)
        
        # Search within the specific document
        results = self._vector_store.similarity_search(
            query_embedding=query_embedding,
            document_id=document_id,
            k=k
        )
        
        # Filter out low-relevance results
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
            context_parts.append(f"[Section {i}]\n{result.content}")
        
        return "\n\n".join(context_parts)
    
    def _parse_llm_output(self, output: str) -> List[MCQQuestion]:
        """Parse LLM JSON output into MCQQuestion objects.
        
        Args:
            output: Raw LLM output string
            
        Returns:
            List of MCQQuestion objects
            
        Raises:
            MCQGenerationError: If parsing fails
        """
        try:
            # Try to extract JSON from the output
            # LLM might include markdown code blocks
            json_match = re.search(r'\{[\s\S]*\}', output)
            if not json_match:
                raise MCQGenerationError("No JSON found in LLM output")
            
            json_str = json_match.group()
            data = json.loads(json_str)
            
            if "questions" not in data:
                raise MCQGenerationError("Missing 'questions' key in LLM output")
            
            questions = []
            for q_data in data["questions"]:
                # Validate required fields
                if not all(k in q_data for k in ["question", "options", "correct_answer_index", "explanation"]):
                    continue  # Skip malformed questions
                
                # Validate options count
                if len(q_data["options"]) != 4:
                    continue  # Skip questions without exactly 4 options
                
                # Validate correct_answer_index
                if not (0 <= q_data["correct_answer_index"] <= 3):
                    continue  # Skip questions with invalid index
                
                questions.append(MCQQuestion(
                    question=q_data["question"],
                    options=q_data["options"],
                    correct_answer_index=q_data["correct_answer_index"],
                    explanation=q_data["explanation"]
                ))
            
            return questions
            
        except json.JSONDecodeError as e:
            raise MCQGenerationError(f"Failed to parse LLM JSON output: {e}")
        except Exception as e:
            raise MCQGenerationError(f"Error parsing MCQ output: {e}")
    
    def invoke(
        self,
        document_id: UUID,
        num_questions: int,
        difficulty: DifficultyLevel
    ) -> MCQGenerationResult:
        """Invoke the MCQ chain to generate questions.
        
        Retrieves relevant context from the document and generates
        MCQs using Gemini 2.5 Flash.
        
        Args:
            document_id: UUID of the document to generate questions from
            num_questions: Number of questions to generate
            difficulty: Difficulty level for the questions
            
        Returns:
            MCQGenerationResult with questions and context flag
            
        Raises:
            MCQGenerationError: If generation fails
            
        Requirements: 1.1, 2.1, 2.2, 2.3, 5.1, 5.3
        """
        # Retrieve relevant chunks (Requirements: 5.1)
        search_results = self.retrieve(document_id, num_questions)
        
        # Handle no-context case
        if not search_results:
            return MCQGenerationResult(
                questions=[],
                has_context=False
            )
        
        # Format context for the prompt
        context = self._format_context(search_results)
        
        # Build the chain: prompt -> llm -> output parser
        chain = self._prompt | self._llm | self._output_parser
        
        # Invoke the chain (Requirements: 5.3)
        try:
            output = chain.invoke({
                "context": context,
                "num_questions": num_questions,
                "difficulty": difficulty.value
            })
        except Exception as e:
            raise MCQGenerationError(f"LLM invocation failed: {e}")
        
        # Parse the output into MCQQuestion objects
        questions = self._parse_llm_output(output)
        
        return MCQGenerationResult(
            questions=questions,
            has_context=True
        )
