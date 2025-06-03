### Multi-agent-diagnosis-chatbot

## Introduction
# The project aims to bridge gap between search engine articles and misinformation without context on the web. Where as a patient/user when they want to search for specific symptoms and what could be the possible diseases based on that that, mimicing the diagnosis procedure of the doctor. It is not here to replace doctors , but just to make the informaton accurate and find relevant information from the chunk of articles over the web and have a diagnositic feature within the system. The chatbot works on mutliple modes, one is just chit-chat, one is symptom diagnosis and another is patient history, a modular agent is designed to identify the intend of the query from the prompt provided by user and classify accordingly. A langchain memory conversation buffer is used to store the interactions and provide personalized diagnosis and understand context much better. Working towards making the project from monolithic to microservices and host into production so real users can interact with the services. There some identified services which will act as tools within the system , will be triggered according to the query itself. Below are the services.

## Auth-Service 
# Reponsible for authentication of the user to access the services.

## User-Service
# Reponsible for the access to the specific user service according to plan.

## Symptom-service
# Responsible for identifying symptom (Work as a seperate service)

## Rag-service
# Responsible for retrieving the relevant information related to user query, symptoms , disease diagnosis and pubmed articles.

## Reasoning-service
# Responsible to reason the AI responses (Disease predicted , as well as mimicing thinking of doctor for better and accurate answers)

## Diagnosis-service
# Responsible for diagnosing the symptoms based on user query.
