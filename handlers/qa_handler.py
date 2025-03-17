import os
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.callbacks import get_openai_callback
from langchain.docstore.document import Document
from config import TEXT_SPLITTER_CONFIG, CONVERSATION_CONFIG

class QAHandler:
    def __init__(self, groq_api_key):
        self.groq_api_key = groq_api_key
        self.setup_embeddings()
        self.vector_store_path = "vector_store"
        os.makedirs(self.vector_store_path, exist_ok=True)

    def setup_embeddings(self):
        model_name = "sentence-transformers/all-mpnet-base-v2"
        model_kwargs = {'device': 'cpu'}
        encode_kwargs = {'normalize_embeddings': False}
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )

    def _prepare_conversation_documents(self, conversation_history):
        if not conversation_history.strip():
            return []
        
        # Convert conversation history into a document format
        conv_doc = Document(
            page_content=conversation_history,
            metadata={"source": "conversation_history"}
        )
        return [conv_doc]

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
            # First, update vector store with latest conversation context
            if conversation_history:
                conv_docs = self._prepare_conversation_documents(conversation_history)
                if conv_docs:
                    knowledge_base.add_documents(conv_docs)
            
            # Use MMR with source-weighted search
            docs = knowledge_base.max_marginal_relevance_search(
                question,
                k=6,  # Increased to account for conversation context
                fetch_k=12,
                lambda_mult=0.7
            )
            
            # Prioritize recent conversation context
            context_docs = [doc for doc in docs if doc.metadata.get("source") == "conversation_history"]
            text_docs = [doc for doc in docs if doc.metadata.get("source") == "main_text"]
            
            # Combine context, prioritizing conversation history
            context = ""
            if context_docs:
                context += "\nRecent conversation context:\n" + "\n".join(doc.page_content for doc in context_docs[:2])
            context += "\nDocument context:\n" + "\n".join(doc.page_content for doc in text_docs[:3])

            enhanced_question = CONVERSATION_CONFIG["system_template"].format(
                conversation_history=conversation_history,
                context=context,
                question=question
            )

            llm = ChatGroq(
                model="mixtral-8x7b-32768", 
                groq_api_key=self.groq_api_key,
                temperature=0.3  # Lower temperature for more focused answers
            )
            
            chain = load_qa_chain(llm, chain_type="stuff")
            
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=enhanced_question)

            if not response or len(response.strip()) < 10:  # Check for very short answers
                return ("I couldn't generate a meaningful answer based on the document content. "
                       "Please try rephrasing your question or ask about something else.")

            return response

        except Exception as e:
            return f"An error occurred while processing your question: {str(e)}"
