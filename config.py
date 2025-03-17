TEXT_SPLITTER_CONFIG = {
    "chunk_size": 2000,
    "chunk_overlap": 400,
    "length_function": len,
    "separators": ["\n\n", "\n", ".", "!", "?", ",", " ", ""]
}

CONVERSATION_CONFIG = {
    "max_context_length": 5,
    "system_template": """You are a helpful AI assistant. Use the following conversation history and context to provide relevant answers.
Previous conversation:
{conversation_history}
Current context: {context}
Current question: {question}
"""
}
