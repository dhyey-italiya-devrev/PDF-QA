from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import CharacterTextSplitter
# from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.callbacks import get_openai_callback
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from transformers import AutoModel
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import streamlit as st
from PIL import Image
import pytesseract
import docx2txt
import os
from dotenv import load_dotenv
from utils.conversation_manager import ConversationManager
from config import TEXT_SPLITTER_CONFIG, CONVERSATION_CONFIG
from utils.text_processor import process_pdf, process_docx, process_image
from handlers.qa_handler import QAHandler
from datetime import datetime
load_dotenv()

# OPENAI_KEY = os.getenv("OPENAI_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")

st.set_page_config(page_title="DocQuest", page_icon=":robot:")
st.title("DocQuest: Empowering Your Documents with AI")
#st.subheader('AI App Implemented By [Maximilien Kpizingui](https://kpizmax.hashnode.dev)')
'''
Say goodbye to manual PDF and DOCX and PNG files text searching and summarization! Our AI-powered app instantly extracts valuable insights from your PDF, DOCX and PNG file, saving you time and effort.With a user-friendly interface, upload your PDF, ask questions, and receive accurate answers in seconds.
Our intelligent algorithms analyze the text, understand context,provide precise answers, write a new content and summarization even for complex queries.Keep track of your queries with session history and easily clear it when needed.
Experience the power of AI to unlock information from PDFs,DOCXs and Image text files  with our secure and efficient AI App.
'''
st.image("post.jpg")

if 'session_history' not in st.session_state:
    st.session_state['session_history'] = []

if 'conversation_manager' not in st.session_state:
    st.session_state['conversation_manager'] = ConversationManager()

if 'qa_handler' not in st.session_state:
    st.session_state['qa_handler'] = QAHandler(groq_api_key)

if 'user_input' not in st.session_state:
    st.session_state.user_input = ""

if 'messages_queue' not in st.session_state:
    st.session_state.messages_queue = []

if 'form_submitted' not in st.session_state:
    st.session_state.form_submitted = False

def clear_input():
    st.session_state.user_input = ""

def main():
    load_dotenv()

    st.sidebar.title("DocQuest")
    st.sidebar.subheader("Menu")
    menu_options = {
        "Upload PDF": upload_pdf,
        "Upload Word": upload_word,
        "Upload Image": upload_image,
        "Session History": display_session_history,
        "Clear Session": clear_session
    }
    choice = st.sidebar.selectbox("Select an option", list(menu_options.keys()))

    if choice in menu_options:
        menu_options[choice]()  # Call the selected function

def display_document_summary():
    if 'document_summary' in st.session_state:
        with st.expander("Document Summary", expanded=True):
            st.markdown(st.session_state['document_summary'])

def upload_pdf():
    st.subheader("Upload your PDF document")
    pdf = st.file_uploader("Choose a PDF file", type="pdf")

    if pdf is not None:
        try:
            with st.spinner("Processing PDF..."):
                extracted_text = process_pdf(pdf)
                st.session_state['document_text'] = extracted_text
                
                # Initialize vector store during upload
                knowledge_base = st.session_state['qa_handler'].process_text(extracted_text)
                st.session_state['knowledge_base'] = knowledge_base
                
                # Generate and store summary
                summary = st.session_state['qa_handler'].summarize_text(extracted_text)
                st.session_state['document_summary'] = summary
                
                # Display summary and start chat
                display_document_summary()
                perform_question_answering(extracted_text)
        except ValueError as e:
            st.error(str(e))

def upload_word():
    st.subheader("Upload your Word document")
    document = st.file_uploader("Choose a Word document", type=["docx"])

    if document is not None:
        try:
            with st.spinner("Processing Word document..."):
                extracted_text = process_docx(document)
                st.session_state['document_text'] = extracted_text
                
                knowledge_base = st.session_state['qa_handler'].process_text(extracted_text)
                st.session_state['knowledge_base'] = knowledge_base
                
                # Generate and store summary
                summary = st.session_state['qa_handler'].summarize_text(extracted_text)
                st.session_state['document_summary'] = summary
                
                # Display summary and start chat
                display_document_summary()
                perform_question_answering(extracted_text)
        except ValueError as e:
            st.error(str(e))

def upload_image():
    st.subheader("Upload your image")
    image = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

    if image is not None:
        try:
            with st.spinner("Processing image..."):
                extracted_text = process_image(image)
                st.session_state['document_text'] = extracted_text
                
                knowledge_base = st.session_state['qa_handler'].process_text(extracted_text)
                st.session_state['knowledge_base'] = knowledge_base
                
                # Display image
                img = Image.open(image)
                st.image(img, caption='Uploaded Image', use_column_width=True)
                
                # Generate and store summary
                summary = st.session_state['qa_handler'].summarize_text(extracted_text)
                st.session_state['document_summary'] = summary
                
                # Display summary and start chat
                display_document_summary()
                perform_question_answering(extracted_text)
        except ValueError as e:
            st.error(str(e))

def init_chat_styles():
    st.markdown("""
        <style>
        .chat-message {
            padding: 1.5rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
            display: flex;
            flex-direction: column;
            max-width: 100%;
        }
        .user-message {
            background-color: #e6f3ff;
            border: 1px solid #b3d7ff;
            color: #000000;
        }
        .bot-message {
            background-color: #f0f2f6;
            border: 1px solid #d1d5db;
            color: #000000;
        }
        .message-timestamp {
            font-size: 0.8rem;
            color: #666666;
            margin-bottom: 0.5rem;
        }
        .message-content {
            margin: 0;
            color: #000000;
            white-space: pre-wrap;
            word-wrap: break-word;
            line-height: 1.5;
            font-size: 1rem;
        }
        </style>
    """, unsafe_allow_html=True)

def display_chat_message(content, is_user=True, timestamp=None):
    message_class = "user-message" if is_user else "bot-message"
    # Format the content by replacing newlines with HTML line breaks
    formatted_content = content.replace('\n', '<br>')
    st.markdown(f"""
        <div class="chat-message {message_class}">
            <div class="message-timestamp">{timestamp}</div>
            <div class="message-content">{formatted_content}</div>
        </div>
    """, unsafe_allow_html=True)

def process_question(question, text, chat_container):
    if not question.strip():
        return
        
    with chat_container:
        st.session_state['processing'] = True
        display_chat_message(question, is_user=True, timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        
        with st.spinner("Processing your question..."):
            if 'knowledge_base' not in st.session_state:
                display_chat_message("Please upload a document first.", is_user=False, 
                                   timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                return
            
            response = st.session_state['qa_handler'].get_answer(
                st.session_state['knowledge_base'],
                question,
                st.session_state['conversation_manager'].get_context_string()
            )
            
            display_chat_message(response, is_user=False, timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        
        st.session_state['conversation_manager'].add_interaction(question, response)
        st.session_state['processing'] = False

def perform_question_answering(text):
    init_chat_styles()
    
    if 'processing' not in st.session_state:
        st.session_state['processing'] = False
    
    # Create a persistent container for the chat
    chat_container = st.container()
    
    # Display existing chat history in the container
    with chat_container:
        for msg in st.session_state['conversation_manager'].get_chat_messages():
            display_chat_message(msg["question"], is_user=True, timestamp=msg["timestamp"])
            display_chat_message(msg["answer"], is_user=False, timestamp=msg["timestamp"])
    
    # Place the form below the chat container
    with st.form(key='question_form', clear_on_submit=True):
        user_question = st.text_input("Ask a question:", key="question_input")
        submitted = st.form_submit_button("Send")
        
        if submitted and user_question:
            process_question(user_question, text, chat_container)
            st.rerun()

def display_session_history():
    st.title("Session History")
    if len(st.session_state['session_history']) > 0:
        for question, answer in st.session_state['session_history']:
            st.write("Question:", question)
            st.write("Answer:", answer)
            st.write("----")
    else:
        st.write("No session history available.")


def clear_session():
    st.session_state['session_history'] = []
    st.session_state['conversation_manager'].clear_history()
    st.session_state.pop('uploaded_pdf', None)
    st.session_state.pop('uploaded_word', None)
    st.session_state.pop('uploaded_image', None)
    st.session_state.pop('document_summary', None)  # Clear summary
    if os.path.exists("vector_store"):
        import shutil
        shutil.rmtree("vector_store")
    st.session_state.pop('knowledge_base', None)
    st.rerun()


if __name__ == '__main__':
    main()
