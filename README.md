# PDF Question Answering

PDF Question Answering is an AI-powered application that allows users to extract answers from PDF files by asking questions. It leverages advanced natural language processing techniques and powerful libraries to provide quick and accurate responses.

## Features

- Upload PDF files: Users can easily upload their PDF files using the user-friendly interface.
- Extract text: The application extracts the text content from the uploaded PDF files using PyPDF2 library.
- Text processing and embeddings: The text is processed and split into chunks using langchain's CharacterTextSplitter, and semantic embeddings are generated using OpenAIEmbeddings.
- Building the knowledge base: The text chunks and embeddings are stored in FAISS, enabling efficient similarity search.
- User interaction and question answering: Users can input their questions, and the application utilizes the question answering chain and knowledge base to provide answers.
- Session history and clearing: The application keeps track of user queries and their answers, and the session history can be cleared with the "Clear" button.

## Technical Summary

### Architecture & Tech Stack Decisions

The application is built using a modern, scalable tech stack with carefully chosen components:

- **Streamlit**: Selected for rapid UI development and seamless state management. Its simple deployment options and interactive widgets made it ideal for building a user-friendly interface.

- **LangChain**: Powers our core Q&A functionality, providing a flexible framework for connecting various LLM components. We chose it for its robust document loading, text splitting, and chain management capabilities.

- **Groq & Mixtral-8x7b**: After evaluating various LLMs, we opted for Groq's hosting of Mixtral-8x7b for its excellent balance of performance and cost. It offers faster inference times than OpenAI while maintaining high-quality responses.

- **FAISS**: Implements our vector storage system, chosen for its efficiency in similarity search operations and ability to handle large document collections without significant performance degradation.

- **HuggingFace Embeddings**: We use all-mpnet-base-v2 for generating embeddings locally, reducing API costs and latency while maintaining high-quality semantic search capabilities.

### Technical Challenges & Solutions

1. **Memory Management**:
   - Challenge: Large documents caused memory issues during processing
   - Solution: Implemented streaming document processing and chunk-based text splitting with careful token management

2. **Context Management**:
   - Challenge: Maintaining conversation context while staying within token limits
   - Solution: Developed a sophisticated ConversationManager that prioritizes relevant historical context using similarity scores

3. **Response Quality**:
   - Challenge: Ensuring consistent, high-quality responses across different document types
   - Solution: Implemented a two-stage summarization process for large documents and enhanced prompting with structured templates

4. **Performance Optimization**:
   - Challenge: Slow response times with large documents
   - Solution: Added parallel processing for document chunks and optimized vector search parameters

### Future Improvements

With additional time, these enhancements would be valuable:

1. **Caching System**:
   - Implement Redis for caching frequent queries and document embeddings
   - Add document preprocessing cache to speed up repeated uploads

2. **Enhanced Document Processing**:
   - Support for more file formats (epub, markdown, etc.)
   - Better image text extraction with advanced OCR options
   - Table and chart data extraction

3. **Advanced Features**:
   - Multi-document comparison and cross-referencing
   - Document segmentation by topics
   - Custom training for domain-specific terminology

4. **UI/UX Improvements**:
   - Real-time answer streaming
   - Better visualization of document structure
   - Advanced search filters and sorting options

### Development Experience

This project has been an excellent opportunity to explore the intersection of document processing and LLMs. The main takeaway was the importance of careful system design when dealing with large language models - particularly around token management and context handling.

The most rewarding aspect was solving the real-time performance challenges while maintaining response quality. The project demonstrated the importance of balancing theoretical capabilities with practical limitations.

## Installation

1. Clone the repository:

git clone https://github.com/dhyey-italiya-devrev/PDF-QA.git


2. Install the required dependencies:

pip install -r requirements.txt


3. Set up the environment variables:

Create a `.env` file in the project root directory and add the necessary environment variables. Refer to the `.env.example` file for the required variables.

## Usage

1. Run the application:

streamlit run main.py


2. Upload your PDF file using the provided interface.
3. Ask a question about the PDF content in the text input field.
4. The application will display the answer based on the uploaded PDF file.

