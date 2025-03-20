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

    def _prepare_summary_chunks(self, text):
        """Prepare text chunks for summarization with token management."""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=TEXT_SPLITTER_CONFIG["chunk_size"]*2,  # Larger chunks for summary
            chunk_overlap=TEXT_SPLITTER_CONFIG["chunk_overlap"],
            length_function=self.count_tokens,
            separators=["\n\n", "\n", ". ", "! ", "? "]
        )
        
        chunks = text_splitter.split_text(text)
        processed_chunks = []
        current_tokens = 0
        max_tokens = GROQ_CONFIG["max_context_ratio"] * CONVERSATION_CONFIG["max_total_tokens"]
        
        for chunk in chunks:
            chunk_tokens = self.count_tokens(chunk)
            if current_tokens + chunk_tokens <= max_tokens:
                processed_chunks.append(chunk)
                current_tokens += chunk_tokens
            else:
                break
                
        return processed_chunks

    def summarize_text(self, text):
        try:
            llm = ChatGroq(
                model="mixtral-8x7b-32768",
                groq_api_key=self.groq_api_key,
                temperature=0.3,
                max_tokens=CONVERSATION_CONFIG["max_response_tokens"]
            )
            
            # Process text in chunks
            chunks = self._prepare_summary_chunks(text)
            
            if not chunks:
                return "Error: Document is too large to process"
            
            # For single small documents
            if len(chunks) == 1 and self.count_tokens(chunks[0]) < 4000:
                prompt = """Please provide a well-formatted summary of the following document:
                {}""".format(chunks[0])
                response = llm.invoke(prompt)
                return response.content if hasattr(response, 'content') else str(response)
            
            # For larger documents, summarize in stages
            summaries = []
            for i, chunk in enumerate(chunks):
                prompt = """Summarize this section of the document:
                {}""".format(chunk)
                response = llm.invoke(prompt)
                summary = response.content if hasattr(response, 'content') else str(response)
                summaries.append(summary)
            
            # Final combined summary if needed
            if len(summaries) > 1:
                combined_text = "\n\n".join(summaries)
                if self.count_tokens(combined_text) > 3000:
                    combined_text = self.truncate_to_token_limit(combined_text, 3000)
                
                final_prompt = """Create a coherent final summary from these section summaries:
                {}""".format(combined_text)
                
                final_response = llm.invoke(final_prompt)
                return final_response.content if hasattr(final_response, 'content') else str(final_response)
            
            return summaries[0]
            
        except Exception as e:
            return f"Error generating summary: {str(e)}"

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
            # Calculate token budget
            max_total = CONVERSATION_CONFIG["max_total_tokens"]
            question_tokens = self.count_tokens(question)
            available_context_tokens = min(
                CONVERSATION_CONFIG["max_context_tokens"],
                max_total - question_tokens - GROQ_CONFIG["token_safety_margin"]
            )

            # Get conversation context first
            conv_docs = []
            if conversation_history:
                conv_docs = self._prepare_conversation_documents(conversation_history, question, knowledge_base)

            # Get and filter main documents
            relevant_docs = knowledge_base.similarity_search_with_score(
                question,
                k=6
            )
            filtered_docs = [doc for doc, score in relevant_docs if score > 0.6]

            # Build context within token limits
            current_tokens = 0
            final_docs = []
            context_parts = []

            # Add conversation context if available
            for doc in conv_docs[:1]:
                doc_tokens = self.count_tokens(doc.page_content)
                if current_tokens + doc_tokens <= available_context_tokens:
                    context_parts.insert(0, f"Previous relevant context:\n{doc.page_content}")
                    current_tokens += doc_tokens
                    final_docs.append(doc)

            # Add document context
            for doc in filtered_docs:
                doc_tokens = self.count_tokens(doc.page_content)
                if current_tokens + doc_tokens <= available_context_tokens:
                    context_parts.append(doc.page_content)
                    current_tokens += doc_tokens
                    final_docs.append(doc)
                else:
                    break

            context = "\n".join(context_parts)
            
            enhanced_question = CONVERSATION_CONFIG["system_template"].format(
                conversation_history="" if not conv_docs else conv_docs[0].page_content,
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
