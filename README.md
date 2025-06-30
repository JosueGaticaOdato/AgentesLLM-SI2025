# Agentes Inteligentes y LLM - Sistemas Inteligentes

## Integrantes

- Josue Gatica Odato
- Lucas Latessa

## Propuesta de trabajo

Construir un sistema de recuperación de información (RAG - Retrieval Augmented Generation) utilizando modelos de lenguaje generativo y embeddings para responder preguntas sobre jugadores y equipos en FIFA

## Herramientas utilizadas:

- HuggingFace Embeddings para vectorizar los datos (GPU si está disponible)
- Gemini (Google) como modelo de lenguaje para responder consultas
- LlamaIndex para realizar búsquedas semánticas sobre los datos
- Streamlit para la interfaz interactiva

## Notebooks

### Implementacion en local

- FIFA_jugadores
- FIFA_equipos

### Implementacion desplegadas en la nube con Streamlit

- Equipos_Streamlit
- Jugadores_Streamlit



- Consultas sobre **Equipos**:  
  [https://equipos-agentesllm-2025.streamlit.app/](https://equipos-agentesllm-2025.streamlit.app/)

- Consultas sobre **Jugadores**:  
  [https://jugadores-agentesllm-2025.streamlit.app/](https://jugadores-agentesllm-2025.streamlit.app/)
