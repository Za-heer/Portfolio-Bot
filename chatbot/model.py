import os
import re
import requests
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.getenv("HF_API_TOKEN") or os.getenv("HF_TOKEN")
HF_LLM_MODEL = os.getenv("HF_LLM_MODEL", "HuggingFaceTB/SmolLM3-3B")

HEADERS = {
    "Authorization": f"Bearer {HF_TOKEN}",
    "Content-Type": "application/json"
} if HF_TOKEN else {}


def _parse_generation_response(resp):
    """Handle Hugging Face inference responses (list/dict/plain text)."""
    try:
        data = resp.json()
    except Exception:
        return resp.text

    # OpenAI-style response
    if isinstance(data, dict) and "choices" in data:
        text = data["choices"][0]["message"]["content"]
        # Clean out <think> or reasoning traces if model adds them
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        return text

    if isinstance(data, list) and len(data):
        if "generated_text" in data[0]:
            return data[0]["generated_text"]
        elif "text" in data[0]:
            return data[0]["text"]
        return str(data[0])

    if isinstance(data, dict):
        return data.get("generated_text") or data.get("text") or str(data)

    return str(data)


class LLMClient:
    def __init__(self, model_id: str = None):
        self.model = model_id or HF_LLM_MODEL
        if not HF_TOKEN:
            raise RuntimeError("❌ HF_API_TOKEN not set. Please add it in .env")
        self.api_url = "https://router.huggingface.co/v1/chat/completions"

    def generate(self, prompt: str, max_new_tokens: int = 300, temperature: float = 0.7) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": (
                    "You are Zaheer's AI portfolio assistant. "
                    "Answer briefly, professionally, and naturally. "
                    "Avoid reasoning traces or self-reflection. "
                    "Focus only on the question and context provided."
                )},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": 0.9
        }

        resp = requests.post(self.api_url, headers=HEADERS, json=payload, timeout=120)
        if resp.status_code != 200:
            raise RuntimeError(f"HF inference error [{resp.status_code}]: {resp.text}")

        return _parse_generation_response(resp)

    def chat_with_context(self, context: str, conversation: str, user_message: str) -> str:
        system_prompt = (
            "You are Zaheer's portfolio assistant. Use the CONTEXT below (skills, projects, education, contact, "
            "achievements, and experience) to answer briefly and politely. "
            "Do NOT include reasoning or explanation text — just answer directly. "
            "If the information is missing, say: 'I don't have that detail yet.'"
        )

        full_prompt = (
            f"{system_prompt}\n\n"
            f"CONTEXT:\n{context}\n\n"
            f"CONVERSATION HISTORY:\n{conversation}\n\n"
            f"User: {user_message}\nAssistant:"
        )

        return self.generate(full_prompt)
