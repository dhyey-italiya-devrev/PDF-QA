TEXT_SPLITTER_CONFIG = {
    "chunk_size": 500,  # Reduced chunk size
    "chunk_overlap": 100,  # Reduced overlap
    "length_function": len,
    "separators": ["\n\n", "\n", ". ", "! ", "? ", ", ", " ", ""]
}

CONVERSATION_CONFIG = {
    "max_context_length": 3,
    "system_template": """Answer based only on the following context. Be concise.
Previous interactions:
{conversation_history}
Context: {context}
Question: {question}
""",
    "max_total_tokens": 3000,  # Reduced from 4000
    "max_context_tokens": 2000,  # Reduced from 3000
    "max_response_tokens": 1000   
}

GROQ_CONFIG = {
    "max_tokens_per_minute": 4500,  # Leave buffer for rate limit
    "max_context_ratio": 0.7,  # Reduced from 0.8
    "token_safety_margin": 200,  # Increased buffer
    "max_docs_per_query": 3  # New: limit number of docs per query
}
