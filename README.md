# MediLink: Multi-Agent Diagnostic Chatbot

## Introduction
MediLink is designed to bridge the gap between scattered web articles and reliable medical context for users seeking information about symptoms and possible diagnoses. It is **not** intended to replace healthcare professionals, but rather to provide accurate, context-aware insights by retrieving and reasoning over relevant content (web articles, PubMed abstracts, patient history). The system uses multiple specialized microservices (agents) to:

- **Chit-Chat Mode:** General conversational interactions.  
- **Symptom Diagnosis Mode:** Identify and standardize symptoms, retrieve candidate diseases, and generate a preliminary diagnosis.  
- **Patient History Mode:** Incorporate stored patient information to refine diagnostic accuracy.

A modular “agent router” inspects the user’s query, classifies intent, and triggers the appropriate service pipeline. Each microservice acts as a tool in the workflow, and interaction history is stored via a LangChain memory buffer to personalize follow-up questions and maintain context.

Below is an overview of each microservice and its responsibility.

---

## Service Overview

### 1. Auth-Service  
Handles user authentication:  
- **Endpoints:**  
  - `POST /auth/signup` → Register a new user (email/password).  
  - `POST /auth/login` → Validate credentials and issue JWT.  
  - `GET /auth/me` → Return user profile from token.  
  - `POST /auth/logout` → Revoke or blacklist token.  

### 2. User-Service  
Manages user profiles and preferences:  
- **Endpoints:**  
  - `GET /users/{id}` → Retrieve user details (demographics, settings).  
  - `PUT /users/{id}` → Update user profile (age, sex, weight, height).  

### 3. Symptom-Service  
Extracts and standardizes symptoms from free-text:  
- **Endpoints:**  
  - `POST /symptoms/extract` →  
    1. Accept `text_input` (e.g., “burning chest pain for two days”).  
    2. Use GPT-3.5 Turbo to identify symptom phrases.  
    3. Map phrases → SNOMED-CT codes (via SciSpacy).  
    4. Return a list of `{ text, snomed_id, umls_cui }`.  

### 4. RAG-Service  
Retrieval-Augmented Generation:  
- **Endpoints:**  
  - `POST /rag/query` →  
    1. Accept standardized symptom embeddings (or `symptom_texts`).  
    2. Encode with `all-mpnet-base-v2` to produce embeddings.  
    3. Query Pinecone (or a FAISS emulator) to retrieve top N candidate diseases.  
    4. Return a ranked list of `{ disease, score, icd10 }`.  

### 5. Diagnosis-Service  
Ensembles RAG results with LLM inference to propose a diagnosis:  
- **Endpoints:**  
  - `POST /diagnose` →  
    1. Accept `{ user_id, symptoms, rag_candidates }`.  
    2. Run MedAlpaca LoRA inference to score candidate diseases.  
    3. Ensemble RAG + LLM confidences to produce a final diagnosis.  
    4. If confidence ≥ threshold, return `{ diagnosis, icd10, confidence, explanation }`.  
    5. Otherwise, return `{ needs_follow_up, follow_up_questions }`.  

### 6. Reasoning-Service  
Fetches evidence from PubMed and generates a Chain-of-Thought explanation:  
- **Endpoints:**  
  - `POST /reason` →  
    1. Accept `{ user_id, demographics, symptoms, disease }`.  
    2. Use Entrez API to retrieve relevant PubMed abstracts.  
    3. Chunk abstracts, embed, and store in ChromaDB (or Pinecone).  
    4. Retrieve top evidence chunks.  
    5. Call GPT-4o to produce a coherent, doctor-style explanation.  
    6. Return `{ cot_explanation }`.  

### 7. Frontend (React + TypeScript)  
Provides a user-facing interface:  
- **Features:**  
  - **Authentication Flow:** Signup, login, logout.  
  - **Chat UI:**  
    1. Free-text input for symptoms/chit-chat.  
    2. Display extracted symptoms.  
    3. Show RAG retrieval and diagnosis results.  
    4. If follow-up is required, render question buttons.  
    5. Once confident, display a CoT explanation from the Reasoning-Service.  
  - **Profile Page:** Edit demographics and personal settings.  

---

## Tech Stack & Architecture

- **Backend Framework:** FastAPI (Python) for all microservices.  
- **Frontend Framework:** React with TypeScript.  
- **Databases:**  
  - **PostgreSQL** (Auth-Service & User-Service).  
  - **Pinecone** for RAG indexing (or FAISS local emulator in development).  
  - **ChromaDB** (or an alternative vector store) for PubMed evidence.  
- **Authentication:** JWT tokens (HTTP-Only cookies in production).  
- **LLMs & Embeddings:**  
  - **OpenAI GPT-3.5 Turbo / GPT-4o** for symptom extraction and CoT reasoning.  
  - **MedAlpaca LoRA** for diagnostic inference.  
  - **Sentence-Transformers (all-mpnet-base-v2)** for embeddings.  
- **Containerization:** Docker for all services; local orchestration via Docker Compose.  
- **Infrastructure as Code:** Terraform to provision AWS resources (VPC, RDS, ECS/Fargate, ECR, S3, CloudFront).  
- **CI/CD:** GitHub Actions to build, test, and push Docker images to ECR, then apply Terraform changes.  
- **Logging & Monitoring:**  
  - Structured JSON logs (stdout) → CloudWatch Logs.  
  - Health check endpoints (`GET /health`) for each service.  
- **Security Best Practices:**  
  - HTTPS enforced via ALB + ACM certificates.  
  - Environment secrets stored in AWS Secrets Manager.  
  - Rate limiting on Auth endpoints.  
  - Least-privilege IAM roles for ECS tasks.  

---

## Folder Structure
/docs/  
/infra/  
/services/  
  ├─ auth-service/  
  ├─ user-service/  
  ├─ symptom-service/  
  ├─ rag-service/  
  ├─ diag-service/  
  ├─ reasoning-service/  
  ├─ frontend/  
  └─ common-libs/  
/docker/  
/ci-cd/  
README.md  
