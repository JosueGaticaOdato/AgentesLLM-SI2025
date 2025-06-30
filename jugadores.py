import os
import pandas as pd
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from langchain_google_genai import ChatGoogleGenerativeAI
import torch
import streamlit as st
from llama_index.core import load_index_from_storage, VectorStoreIndex, Document, Settings
from llama_index.core.storage import StorageContext
import zipfile
import gdown
os.environ["HUGGINGFACEHUB_API_TOKEN"] = st.secrets["HUGGINGFACEHUB_API_TOKEN"]
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
# os.environ["GOOGLE_API_KEY"] = "" #PONER API KEY DE GOOGLE

# Cargar los jugadores
# df = pd.read_csv("jugadores_filtrados.csv")
# df = pd.read_csv("jugadores_RealMadrid.csv")

# Convertir cada fila a un documento de texto para indexar
#documents = []
#for _, row in df.iterrows():
#    text = "\n".join([f"{col}: {row[col]}" for col in df.columns])
#    documents.append(Document(text=text))

# Usar embeddings con GPU si hay disponible
device = "cuda" if torch.cuda.is_available() else "cpu"
embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-mpnet-base-v2",#all-MiniLM-L12-v2/all-MiniLM-L6-v2/all-mpnet-base-v2
    device=device
)

# Cliente Gemini LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)

# Crear contexto de servicio
Settings.llm = llm
Settings.embed_model = embed_model

# Creando indice
#index = VectorStoreIndex.from_documents(documents)
#query_engine = index.as_query_engine(similarity_top_k=3)

# # Guardar el indice
# from google.colab import drive
# drive.mount('/content/drive')
# persist_dir = "/content/drive/MyDrive/indice_jugadores"
# index.storage_context.persist(persist_dir=persist_dir)

file_id = "1GHdyMXzoRHeIT4AwCREHkIwXZHwkybOg"  # file id
zip_filename = "indice_jugadores.zip"

if not os.path.exists("indice_jugadores"):
    # Descargar desde Google Drive
    url = f"https://drive.google.com/uc?id={file_id}"
    output = gdown.download(url, zip_filename, quiet=False)

    # Descomprimir
    with zipfile.ZipFile(zip_filename, "r") as zip_ref:
        os.makedirs("indice_jugadores", exist_ok=True)
        zip_ref.extractall("indice_jugadores")

#Cargar carpeta con indices para que tarde menos
@st.cache_resource
def cargar_query_engine():
    # Cargar el storage context desde el directorio
    storage_context = StorageContext.from_defaults(persist_dir="indice_jugadores")
    # Cargar el indice desde el contexto
    index = load_index_from_storage(storage_context)
    # Crear el query engine
    return index.as_query_engine(similarity_top_k=3)

query_engine = cargar_query_engine()

# Interfaz de usuario
st.set_page_config(
    page_title="Consultas FIFA 23 - Jugadores",
    page_icon="⚽",
    layout="centered"
)

st.title("Consultas FIFA 23 - Jugadorees")
st.markdown("### 🧠 Ejemplos de preguntas que podes hacer:")

ejemplos = [
    "¿Cual es la valoracion de Lionel Messi?",
    "¿Cuanto vale en euros Kevin De Bruyne?",
    "¿Cuantos años tiene Luka Modric?",
    "¿En que posicion juega Dibu Martinez?",
    "¿Cual es la velocidad de Kylian Mbappe?",
    "¿Cual es el salario de Karim Benzema?",
    "¿Cual es la resistencia y fuerza de Neymar Jr?",
]

# Mostrar los ejemplos con botones
for i, ejemplo in enumerate(ejemplos):
    col1, col2 = st.columns([0.8, 0.2])
    with col1:
        st.markdown(f"- {ejemplo}")
    with col2:
        if st.button("Consultar", key=f"btn_{i}"):
            st.session_state["consulta"] = ejemplo

# Campo de entrada manual
consulta = st.text_input("Ingresa tu consulta:", value=st.session_state.get("consulta", ""))

# Ejecutar si hay una consulta
if consulta:
    with st.spinner("⏳ Procesando tu consulta..."):
        consulta_modificada = consulta.strip() + ". Responde en español."
        respuesta_llm = llm.invoke(consulta_modificada)
        respuesta_rag = query_engine.query(consulta_modificada)

    st.success("✅ Consulta completada")

    st.markdown("### 💬 Respuesta sin RAG:")
    st.write(respuesta_llm.content)

    st.markdown("### 🔍 Respuesta con RAG:")
    st.write(str(respuesta_rag))
    print(consulta_modificada, str(respuesta_rag))
