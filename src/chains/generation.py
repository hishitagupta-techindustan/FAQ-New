"""
Answer generation chain (OpenAI version)
"""
from typing import List, Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from loguru import logger

from ..config import settings, ANSWER_GENERATION_PROMPT, FOLLOWUP_QUESTIONS_PROMPT


class AnswerGenerator:
    """Generate answers from retrieved context"""

    def __init__(self):
        """Initialize answer generator"""
        self.llm = ChatOpenAI(
            model=settings.openai_model_large,  # e.g. gpt-4o
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            api_key=settings.openai_api_key
        )

        self.answer_prompt = ChatPromptTemplate.from_template(
            ANSWER_GENERATION_PROMPT
        )
        self.followup_prompt = ChatPromptTemplate.from_template(
            FOLLOWUP_QUESTIONS_PROMPT
        )

    def generate(
        self,
        query: str,
        context_docs: List[Dict[str, Any]],
        conversation_history: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """Generate an answer based on query, context, and conversation history"""
        # print(context_docs)
        
        context_text = self._format_context(context_docs)
        # print(context_text)
        logger.debug(f"Formatted context: {context_text[:200]}...")  # Fixed: use logger instead of print
        
        history_text = self._format_history(conversation_history or [])
        logger.debug(f"Formatted history: {history_text[:200]}...")  # Fixed: use logger instead of print

        try:
            messages = self.answer_prompt.format_messages(
                query=query,
                context=context_text,
                history=history_text or "No previous conversation"
            )

            response = self.llm.invoke(messages)
            answer = response.content.strip()

            sources = self._extract_sources(context_docs)

            logger.debug(f"Generated answer with {len(sources)} sources")

            return {
                "answer": answer,
                "sources": sources,
                "context_docs": context_docs
            }

        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return {
                "answer": "I encountered an error generating the answer.",
                "sources": [],
                "context_docs": []
            }

    def _format_context(self, context_docs: List[Dict[str, Any]]) -> str:
        """Format retrieved documents into a single context string"""
        if not context_docs:
            return "No relevant documents found."

        formatted_docs = []
        for i, doc in enumerate(context_docs, 1):
            # Use 'text' instead of 'content'
            content = doc.get("text", "")
            
            # Extract source from nested metadata
            metadata = doc.get("metadata", {})
            source = metadata.get("source", "Unknown source")
            
            # Optional: Include page number if available
            page = metadata.get("page", None)
            source_str = f"{source}" if page is None else f"{source} (page {page})"
            
            # Optional: Include rerank score for debugging
            rerank_score = doc.get("rerank_score")
            score_str = f" [score: {rerank_score:.3f}]" if rerank_score is not None else ""
            
            formatted_docs.append(
                f"Document {i}:\nSource: {source_str}{score_str}\nContent:\n{content}\n"
            )

        return "\n\n".join(formatted_docs)

    def _format_history(self, history: List[Dict]) -> str:
        """Format conversation history"""
        if not history:
            return ""

        formatted_history = []
        for message in history:
            role = message.get("role", "user")
            content = message.get("content", "")
            formatted_history.append(f"{role.capitalize()}: {content}")

        return "\n".join(formatted_history)

    def _extract_sources(self, context_docs: List[Dict[str, Any]]) -> List[str]:
        """Extract unique sources from context documents"""
        sources = []
        
        # ✅ Add this type check
        if not isinstance(context_docs, list):
            logger.error(f"context_docs is not a list, got {type(context_docs)}")
            return sources
        
        for doc in context_docs:
            # ✅ Add this check too
            if not isinstance(doc, dict):
                continue
            
            metadata = doc.get("metadata", {})
            
            if isinstance(metadata, dict):  # ✅ And this one
                source = metadata.get("source")
                if source and source not in sources:
                    sources.append(source)
        
        return sources

    def generate_followup_questions(
        self,
        query: str,
        answer: str,
        context_docs: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Generate follow-up questions based on the original query,
        generated answer, and retrieved context.
        """

        context_text = self._format_context(context_docs)

        try:
            messages = self.followup_prompt.format_messages(
                query=query,
                answer=answer,
                context=context_text
            )

            response = self.llm.invoke(messages)
            content = response.content.strip()

            # Fixed: More robust parsing of follow-up questions
            questions = []
            for line in content.split("\n"):
                line = line.strip()
                if not line:
                    continue
                # Remove common list markers
                for marker in ["- ", "* ", "• "]:
                    if line.startswith(marker):
                        line = line[len(marker):].strip()
                        break
                # Remove numbering like "1. ", "2. ", etc.
                if len(line) > 2 and line[0].isdigit() and line[1:3] in [". ", ") "]:
                    line = line[3:].strip()
                elif len(line) > 3 and line[:2].isdigit() and line[2:4] in [". ", ") "]:
                    line = line[4:].strip()
                
                if line:
                    questions.append(line)

            logger.debug(f"Generated {len(questions)} follow-up questions")

            return questions[:5]  # Limit to max 5 follow-ups

        except Exception as e:
            logger.error(f"Error generating follow-up questions: {e}")
            return []