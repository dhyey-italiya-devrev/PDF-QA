from datetime import datetime

class ConversationManager:
    def __init__(self):
        self.conversation_history = []

    def add_interaction(self, question, answer):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.conversation_history.append({
            "question": question, 
            "answer": answer,
            "timestamp": timestamp
        })

    def get_context_string(self, max_context=5):
        context = ""
        recent_history = self.conversation_history[-max_context:] if self.conversation_history else []
        for interaction in recent_history:
            context += f"Q: {interaction['question']}\nA: {interaction['answer']}\n\n"
        return context

    def clear_history(self):
        self.conversation_history = []

    def get_chat_messages(self):
        return self.conversation_history
