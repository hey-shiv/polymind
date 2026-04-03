# PolyMind: Multi-Personality RAG System

A research-oriented multi-agent AI system that generates responses from multiple personas using retrieval-augmented generation (RAG) and multi-model routing.

## 🚀 Features
- Multi-personality reasoning (Elon Musk, Robert Greene, Steve Jobs)
- Persona-aware retrieval system
- Query expansion using Mini LLM
- Multi-model routing (Mistral + Phi)
- Controlled prompt engineering for behavior shaping

## 🧠 Architecture
User Query  
→ Query Expansion  
→ Persona-based Retrieval (FAISS)  
→ Prompt Builder  
→ Model Router  
→ LLMs (Mistral / Phi)  
→ Multi-Agent Output  

## 📁 Structure
- `rag/` → retrieval system  
- `pipeline/` → orchestration logic  
- `prompts/` → persona templates  
- `src/` → core modules  
- `ui/` → frontend (planned)  
- `notebooks/` → experiments  

## 🧪 Research Insight
This system demonstrates that combining:
- model-level diversity  
- persona conditioning  

leads to improved behavioral differentiation in LLM outputs.

## ⚙️ Setup
```bash
pip install -r requirements.txt
python app.py
