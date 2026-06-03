# Portfolio Assistant API

A high-performance, lightweight asynchronous orchestration microservice built with **Python** and **FastAPI**. This API acts as the core intelligence layer for an interactive developer portfolio dashboard, driving a contextual AI Portfolio Agent capable of answering complex inquiries regarding system architecture, core technical proficiencies, and software engineering methodologies.

---

## Tech Stack & Pillars

* **Framework:** FastAPI (Asynchronous, high-performance ASGI web framework)
* **AI Orchestration:** LangChain / LangGraph (Contextual routing, custom tool invocation, and state processing)
* **Language:** Python 3.11+
* **Deployment:** Production-ready containerized deployment environment

---

## Core Architecture Features

* **Asynchronous Execution:** Engineered around FastAPI's async capabilities to handle low-latency client-side polling and streaming requests under varied load conditions.
* **Deterministic Agent Routing:** Implements advanced system prompting and orchestration logic to keep agent responses tightly bounded to engineering, architecture, and professional experience frameworks.
* **Error & Fallback Handling:** Features robust validation middleware and custom routing rules to elegantly intercept anomalous inputs and guide users back into optimal conversational pathways.
* **CORS Security:** Configured with secure Cross-Origin Resource Sharing (CORS) headers to tightly regulate access from verified frontend clients.

---

## Directory Structure

```text
├── app/
│   ├── __init__.py
│   ├── main.py          # FastAPI application entry point & CORS config
│   ├── agent.py         # LangChain / LangGraph orchestration logic
│   └── prompts.py       # System instructions and prompt templates
├── requirements.txt     # Service dependencies
└── README.md
