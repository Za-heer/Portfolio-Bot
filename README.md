# 🤖 Portfolio Chatbot (RAG + FastAPI + Hugging Face)

An **AI-powered personal portfolio assistant** built with **FastAPI**, **Retrieval-Augmented Generation (RAG)**, and **Hugging Face Inference API**.  
This chatbot allows visitors to chat conversationally about Zaheer’s projects, skills, education, and experiences — directly from the portfolio website.

---

## 🚀 Features

- **Conversational Q&A** about portfolio content  
- **RAG-based context retrieval** using FAISS / embeddings  
- **Lightweight Hugging Face LLM** for natural responses  
- **FastAPI backend** with clean modular structure  
- **Ready for frontend integration (React / Next.js)**  
- **Persistent conversation memory** per session  

---

## 🧠 Tech Stack

| Component            | Technology                                              |
| -------------------- | ------------------------------------------------------- |
| Backend              | FastAPI                                                 |
| AI Model             | Hugging Face Inference API (`HuggingFaceTB/SmolLM3-3B`) |
| Embeddings           | `sentence-transformers/all-MiniLM-L6-v2`                |
| Retrieval            | FAISS vector search                                     |
| Environment          | Python 3.10 +                                           |
| Deployment           | Render / Hugging Face Spaces                            |
| Frontend Integration | React / Next.js (Vercel)                                |

---

## 📁 Project Structure

portfolio-chatbot/
├──chatbot/
│  ├──app.py # FastAPI main entry
│  ├── model.py # LLM client (Hugging Face Inference API)
│  ├── rag.py # RAG + FAISS embedding retrieval
│  └── memory.py # Conversation memory management
│
├── data/
│ └── portfolio.json # Portfolio data source
│
├── requirements.txt
├── run.py
└── README.md

---

## ⚙️ Setup & Run Locally

### 1️⃣ Clone the repo
git clone https://github.com/Za-heer/Portfolio-Bot.git
cd portfolio-chatbot

### 2️⃣ Create a virtual environment
python -m venv venv
source venv/bin/activate   # or venv\Scripts\activate on Windows

### 3️⃣ Install dependencies
pip install -r requirements.txt

### 4️⃣ Add .env file
Create a .env file in the root folder and add:
HF_API_TOKEN=your_huggingface_token
HF_LLM_MODEL=HuggingFaceTB/SmolLM3-3B
HF_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
RAG_INDEX_DIR=data/faiss_index

### 5️⃣ Build the FAISS index
python -m chat.rag
Then type any debug query to verify chunks are loaded:
Enter query (or 'exit'): projects

### 6️⃣ Run the FastAPI server
uvicorn chat.app:app --reload
Visit the API docs at:
👉 http://127.0.0.1:8000/docs


---
## 🌐 Deployment (Render / Hugging Face Spaces)
1. Push code to GitHub
2. Create a new Render Web Service
3. Set Build Command:
   pip install -r requirements.txt
4. Start Command:
   uvicorn chat.app:app --host 0.0.0.0 --port 10000
5. Add environment variables from .env in Render dashboard.
6. Deploy and copy the public API URL (e.g. https://portfolio-chatbot.onrender.com/chat).
   

---
## 💬 Frontend Integration (Next.js / React)
Once deployed, add this chatbot to your portfolio:

const res = await fetch("https://portfolio-chatbot.onrender.com/chat", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ message: "Tell me about Zaheer’s projects" }),
});
const data = await res.json();
console.log(data.response);

Or integrate the ready-made chat widget from the Chatbot.jsx component.


---
## 🧩 Example Conversation
User: “Tell me about your latest project.”
Bot: “My latest project is a Code Analyzer built using CodeBERT and Pylint. It analyzes .py and .ipynb files to detect code issues and provide quality insights.”


---
## 🧑‍💻 Author
### Zaheer Khan
🌐 Portfolio: https://zaheer-portfolio-786.vercel.app/
💼 GitHub: Za-heer
🧠 Focus: Data Science | AI | Web Development


---
## 🏷 License
This project is open-source under the MIT License.

## ⭐ If you like this project, don’t forget to give it a star!