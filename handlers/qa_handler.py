import os
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.callbacks import get_openai_callback
from langchain.docstore.document import Document
from config import TEXT_SPLITTER_CONFIG, CONVERSATION_CONFIG, GROQ_CONFIG
import tiktoken

class QAHandler:
    def __init__(self, groq_api_key):
        self.groq_api_key = groq_api_key
        self.setup_embeddings()
        self.vector_store_path = "vector_store"
        os.makedirs(self.vector_store_path, exist_ok=True)
        self.tokenizer = tiktoken.get_encoding("cl100k_base")  # Used by Mixtral

    def setup_embeddings(self):
        model_name = "sentence-transformers/all-mpnet-base-v2"
        model_kwargs = {'device': 'cpu'}
        encode_kwargs = {'normalize_embeddings': False}
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    def truncate_to_token_limit(self, text: str, limit: int) -> str:
        tokens = self.tokenizer.encode(text)
        if len(tokens) <= limit:
            return text
        return self.tokenizer.decode(tokens[:limit])

    def _prepare_conversation_documents(self, conversation_history, question, knowledge_base):
        if not conversation_history.strip():
            return []
        
        # Split conversation history into individual QA pairs
        qa_pairs = conversation_history.strip().split('\n\n')
        conv_docs = []
        
        for qa_pair in qa_pairs:
            if not qa_pair.strip():
                continue
            conv_doc = Document(
                page_content=qa_pair,
                metadata={"source": "conversation_history"}
            )
            conv_docs.append(conv_doc)
        
        # Get semantic similarity scores for conversation history
        if conv_docs:
            scores = knowledge_base.similarity_search_with_score(question, k=len(conv_docs))
            relevant_docs = [doc for doc, score in scores if score > 0.7]  # Only keep highly relevant history
            return relevant_docs[:2]  # Limit to 2 most relevant conversation pieces
        
        return []

    def process_text(self, text, conversation_history=""):
        try:
            if not text or not text.strip():
                raise ValueError("No text content provided")
                
            # Initialize text splitter with correct configuration
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=TEXT_SPLITTER_CONFIG["chunk_size"],
                chunk_overlap=TEXT_SPLITTER_CONFIG["chunk_overlap"],
                length_function=TEXT_SPLITTER_CONFIG["length_function"],
                separators=TEXT_SPLITTER_CONFIG["separators"]
            )
            
            # Process text in chunks
            chunks = text_splitter.split_text(text)
            
            # Convert chunks to documents with metadata
            docs = [
                Document(
                    page_content=chunk, 
                    metadata={
                        "source": "main_text",
                        "chunk_index": idx,
                    }
                ) for idx, chunk in enumerate(chunks)
            ]
            
            # Create vector store
            vector_store = FAISS.from_documents(docs, self.embeddings)
            
            # Add conversation history if present
            if conversation_history:
                conv_docs = self._prepare_conversation_documents(conversation_history)
                if conv_docs:
                    vector_store.add_documents(conv_docs)
            
            # Save to disk
            vector_store.save_local(self.vector_store_path)
            return vector_store
                
        except Exception as e:
            raise ValueError(f"Error processing text: {str(e)}")

    def get_answer(self, knowledge_base, question, conversation_history):
        try:
            # Calculate stricter token budget
            max_total = CONVERSATION_CONFIG["max_total_tokens"]
            question_tokens = self.count_tokens(question)
            system_prompt_tokens = self.count_tokens(CONVERSATION_CONFIG["system_template"])
            available_context_tokens = min(
                CONVERSATION_CONFIG["max_context_tokens"],
                max_total - question_tokens - system_prompt_tokens - GROQ_CONFIG["token_safety_margin"]
            )

            # Get top-k most relevant documents with scores
            relevant_docs = knowledge_base.similarity_search_with_score(
                question,
                k=GROQ_CONFIG["max_docs_per_query"]
            )

            # Sort by relevance score and take only the most relevant
            filtered_docs = sorted(
                [(doc, score) for doc, score in relevant_docs if score > 0.6],
                key=lambda x: x[1],
                reverse=True
            )

            # Build context within strict token limits
            current_tokens = 0
            final_docs = []
            context_parts = []

            # Add only the most relevant document content that fits
            for doc, score in filtered_docs:
                doc_text = doc.page_content
                doc_tokens = self.count_tokens(doc_text)
                
                # If document is too large, truncate it
                if doc_tokens > available_context_tokens // 2:
                    doc_text = self.truncate_to_token_limit(doc_text, available_context_tokens // 2)
                    doc_tokens = self.count_tokens(doc_text)
                
                if current_tokens + doc_tokens <= available_context_tokens:
                    context_parts.append(doc_text)
                    current_tokens += doc_tokens
                    final_docs.append(Document(page_content=doc_text, metadata=doc.metadata))
                else:
                    break

            context = "\n".join(context_parts)
            
            # Only include conversation history if there's room
            conv_history = ""
            if conversation_history and current_tokens < available_context_tokens:
                conv_history = self.truncate_to_token_limit(
                    conversation_history,
                    available_context_tokens - current_tokens
                )

            enhanced_question = CONVERSATION_CONFIG["system_template"].format(
                conversation_history=conv_history,
                context=context,
                question=question
            )

            llm = ChatGroq(
                model="mixtral-8x7b-32768",
                groq_api_key=self.groq_api_key,
                temperature=0.3,
                max_tokens=CONVERSATION_CONFIG["max_response_tokens"]
            )
            
            chain = load_qa_chain(llm, chain_type="stuff")
            
            with get_openai_callback() as cb:
                response = chain.run(input_documents=final_docs, question=enhanced_question)

            if not response or len(response.strip()) < 10:
                return ("I couldn't generate a meaningful answer based on the document content. "
                       "Please try rephrasing your question or ask about something else.")

            return response

        except Exception as e:
            return f"An error occurred while processing your question: {str(e)}"
