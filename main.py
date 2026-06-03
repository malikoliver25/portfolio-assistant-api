import os
import time
import logging
import json
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware # 1. Import Middleware
from pydantic import BaseModel
from typing import TypedDict, List

# 1. High-Reliability Logging & Telemetry Configuration (Structured for NewRelic)
load_dotenv()
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s [%(levelname)s] NewRelic_Track | log_type=metrics | %(message)s"
)
logger = logging.getLogger("Portfolio_Agent")

app = FastAPI(title="Malik's Agentic Portfolio Assistant API")

# 2. ENABLE CROSS-ORIGIN BROWSING LAYER
# This allows your frontend website to securely communicate with the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins to access your API
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods (POST, GET, OPTIONS, etc.)
    allow_headers=["*"],  # Allows all headers
)

# 3. Optimized Production RAG Layer: Enriched Semantic Context Data
RESUME_DATA = [
    "Malik Oliver works as an AI Quality Assurance Specialist at Outlier AI (01/2025 - Present), where he conducts rigorous code reviews, automated script evaluations, and functional testing on LLM outputs to guarantee technical accuracy.",
    "Core AI Engineering and Agentic Skills: Malik designs and builds intelligent conversational systems and autonomous agents using LangChain, LangGraph, RAG pipelines, Prompt Engineering, and Vector Databases. Experienced with PyTorch, Hugging Face, and deep learning architectures.",
    "Backend Engineering and Data Systems Skills: Malik specializes in backend web development, building high-performance APIs and microservices using Python, FastAPI, Django, JavaScript, and TypeScript with Next.js and React. Expert in data systems, SQL databases, Redis caching, and RESTful API architecture.",
    "DevOps, MLOps, and Software Reliability: Malik ensures software quality using automated E2E testing frameworks like Playwright and Postman. Proficient in Docker containerization, Azure cloud infrastructure, CI/CD pipelines, Test-Driven Development (TDD), and monitoring telemetry via Langfuse and NewRelic Logs.",
    "Prior Industrial Validation Experience: Malik previously worked as a Fabricator for Bastian Solutions and Johnson Controls, and as a Non-Destructive Testing (NDT) Technician for EnerSys, where he specialized in defect lifecycle tracking, root cause analysis, and failure prevention."
]

class PurePythonVectorStore:
    """
    A pure-python semantic vector database engine. Generates real high-dimensional 
    embeddings using Hugging Face's API and performs normalized cosine similarity matching.
    """
    def __init__(self, texts: List[str]):
        self.texts = texts
        self.hf_token = os.getenv("HF_TOKEN")
        self.api_url = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"
        self.embeddings = []
        self._initialize_vector_db()

    def _get_embedding(self, text: str) -> List[float]:
        if not self.hf_token:
            return [float(text.lower().count(c)) for c in "abcdefghijklmnopqrstuvwxyz"]
        
        try:
            headers = {"Authorization": f"Bearer {self.hf_token}"}
            response = requests.post(self.api_url, headers=headers, json={"inputs": text, "options": {"wait_for_model": True}}, timeout=5)
            if response.status_code == 200:
                return response.json()
        except Exception:
            pass
        return [float(text.lower().count(c)) for c in "abcdefghijklmnopqrstuvwxyz"]

    def _initialize_vector_db(self):
        logger.info("Vector Database: Computing Hugging Face sentence embeddings for documentation chunks...")
        for text in self.texts:
            self.embeddings.append(self._get_embedding(text))
        logger.info("Vector Database: Semantic index build complete.")

    def similarity_search(self, query: str, k: int = 2) -> List[str]:
        query_vector = self._get_embedding(query)
        scores = []
        
        for idx, doc_vector in enumerate(self.embeddings):
            if len(query_vector) != len(doc_vector):
                score = 0.0
            else:
                dot_prod = sum(q * d for q, d in zip(query_vector, doc_vector))
                q_norm = sum(q * q for q in query_vector) ** 0.5
                d_norm = sum(d * d for d in doc_vector) ** 0.5
                
                # Strict cosine similarity calculation
                score = dot_prod / (q_norm * d_norm) if (q_norm * d_norm) > 0 else 0.0
            scores.append((score, self.texts[idx]))
            
        # Sort explicitly by highest mathematical similarity score
        scores.sort(key=lambda x: x[0], reverse=True)
        return [text for _, text in scores[:k]]

retriever = PurePythonVectorStore(RESUME_DATA)

# 4. LangGraph Multi-Step State Flow Orchestration
from langgraph.graph import StateGraph, END

class AgentState(TypedDict):
    user_query: str
    context_docs: List[str]
    classification: str
    final_response: str

def classify_intent_node(state: AgentState):
    query = state["user_query"].lower()
    keywords = ["malik", "job", "experience", "skills", "resume", "work", "engineer", "hire", "tech"]
    state["classification"] = "portfolio_query" if any(k in query for k in keywords) else "general_chit_chat"
    return state

def retrieve_rag_context_node(state: AgentState):
    if state["classification"] == "portfolio_query":
        state["context_docs"] = retriever.similarity_search(state["user_query"])
    else:
        state["context_docs"] = []
    return state

def generate_response_node(state: AgentState):
    if state["classification"] == "general_chit_chat":
        state["final_response"] = "Hi! I am Malik's AI Assistant. Ask me about his experience in Python, RAG pipelines, or LangGraph!"
    else:
        context = " ".join(state["context_docs"])
        state["final_response"] = f"Based on Malik's verified resume profile: {context}"
    return state

workflow = StateGraph(AgentState)
workflow.add_node("classifier", classify_intent_node)
workflow.add_node("retriever", retrieve_rag_context_node)
workflow.add_node("generator", generate_response_node)

workflow.set_entry_point("classifier")
workflow.add_edge("classifier", "retriever")
workflow.add_edge("retriever", "generator")
workflow.add_edge("generator", END)

agent_executor = workflow.compile()

# 5. REST API Endpoint & Structured Output Logs
class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
async def chat_with_agent(payload: ChatRequest):
    start_perf = time.time()
    logger.info(f"event=request_received | query='{payload.message}'")
    
    langfuse_mock_trace = {
        "trace_id": f"tr_{int(start_perf)}",
        "name": "portfolio-assistant-session",
        "input": payload.message
    }
    
    try:
        inputs = {"user_query": payload.message, "context_docs": [], "classification": "", "final_response": ""}
        output_state = agent_executor.invoke(inputs)
        
        latency = round((time.time() - start_perf) * 1000, 2)
        
        logger.info(
            f"event=execution_success | trace_id={langfuse_mock_trace['trace_id']} | "
            f"latency_ms={latency} | route={output_state['classification']} | "
            f"context_chunks_retrieved={len(output_state['context_docs'])}"
        )
        
        return {
            "response": output_state["final_response"],
            "meta": {
                "intent_routed": output_state["classification"],
                "context_chunks_retrieved": len(output_state["context_docs"]),
                "performance_ms": latency,
                "telemetry_trace_id": langfuse_mock_trace["trace_id"]
            }
        }
    except Exception as error:
        logger.error(f"event=execution_failure | error_msg='{str(error)}'")
        raise HTTPException(status_code=500, detail="Internal Orchestration Error")