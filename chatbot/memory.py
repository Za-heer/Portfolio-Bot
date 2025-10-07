# chatbot/memory.py
from typing import List, Dict

class ConversationMemory:
    def __init__(self, max_messages: int = 12):
        self.buffer: List[Dict[str, str]] = []
        self.max_messages = max_messages

    def add_user(self, text: str):
        self._add("user", text)

    def add_assistant(self, text: str):
        self._add("assistant", text)

    def _add(self, role: str, text: str):
        self.buffer.append({"role": role, "text": text})
        if len(self.buffer) > self.max_messages:
            self.buffer = self.buffer[-self.max_messages:]

    def formatted(self) -> str:
        # return as plain conversation string for prompt
        parts = []
        for m in self.buffer:
            if m["role"] == "user":
                parts.append(f"User: {m['text']}")
            else:
                parts.append(f"Assistant: {m['text']}")
        return "\n".join(parts)
