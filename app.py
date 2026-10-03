import streamlit as st
import os
import chromadb
import pandas as pd
from google import genai
from dotenv import load_dotenv

# Carga las variables del archivo
load_dotenv()

# Configuración general de la página: título de la pestaña del navegador e icono.
st.set_page_config(page_title="Asistente de Correo", page_icon="📧")

# CSS personalizado para aumentar el tamaño de letra del chat y del campo de escritura.
st.markdown("""
    <style>
    .stChatMessage p {
        font-size: 18px;
    }
    .stChatInput textarea {
        font-size: 18px;
    }
    </style>
""", unsafe_allow_html=True)

# Diccionario: traduce el nombre que el usuario escribe en el login
# al nombre de la carpeta ChromaDB correspondiente.Para añadir futuros empleados 
# es el sitio que hay que ajustar.
EMPLEADOS = {
    "Sally Beck": "chroma_beck_A",
    "Mark Taylor": "chroma_taylor_A",
}

MODEL_NAME = "gemini-3.1-flash-lite"   # Modelo LLM usado para generar respuestas
UMBRAL_DISTANCIA = 0.6                  # Distancia máxima para considerar un correo "relevante"

if "empleado_actual" not in st.session_state:
    st.session_state.empleado_actual = None   # None = nadie ha hecho login todavía

if "mensajes" not in st.session_state:
    st.session_state.mensajes = []            # Historial de la conversación


def agente_generador_respuesta(client_gemini, pregunta, resultado_chroma, nombre_empleado, umbral_distancia=UMBRAL_DISTANCIA):
    """
    Recibe el resultado de una consulta a ChromaDB (5 correos candidatos) y genera
    una respuesta con Gemini usando solo los correos que superan el filtro de relevancia.
    Devuelve una tupla: (texto de la respuesta, lista de fuentes usadas).
    """
    documentos = resultado_chroma["documents"][0]   # Texto de los 5 correos recuperados
    distancias = resultado_chroma["distances"][0]   # Distancia semántica de cada uno (0 = idéntico, más alto = menos relacionado)
    ids = resultado_chroma["ids"][0]                 # Identificador de cada documento en ChromaDB
    metadatas = resultado_chroma["metadatas"][0]     # Asunto, remitente, fecha, etc. de cada uno

    documentos_relevantes = []
    fuentes = []
    # zip() recorre las 4 listas a la vez, elemento a elemento
    for doc, dist, doc_id, meta in zip(documentos, distancias, ids, metadatas):
        if dist <= umbral_distancia:
            documentos_relevantes.append(doc)
            fuentes.append({
                "id": doc_id,
                "asunto": meta.get("subject", ""),      # .get(clave, valor_si_no_existe) evita errores si falta el campo
                "de": meta.get("from_addr", ""),
                "fecha": meta.get("fecha_primer_mensaje", ""),
                "distancia": round(dist, 4),
            })

    # Si ningún correo es lo bastante relevante, no llamamos al LLM (ahorra coste y evita alucinaciones)
    if not documentos_relevantes:
        return "No information available in the provided emails.", []

    # Une todos los correos relevantes en un único bloque de texto, separados por "---"
    contexto = "\n\n---\n\n".join(documentos_relevantes)
    n_correos = len(documentos_relevantes)

    prompt = f"""Eres el asistente personal de correo electrónico de {nombre_empleado}. {nombre_empleado} es quien te está haciendo la pregunta directamente.

Respóndele en segunda persona ("you", "your"), como lo haría un asistente en una conversación normal.

Recibirás {n_correos} correos recuperados de una base de datos vectorial por su similitud semántica con la pregunta. Estos correos no forman necesariamente un mismo hilo ni están relacionados entre sí: trátalos como fuentes independientes.

Responde la pregunta ÚNICAMENTE usando la información de los correos proporcionados.

CORREOS:
{contexto}

PREGUNTA: {pregunta}

Responde de forma directa, en el idioma de los correos.
"""

    respuesta = client_gemini.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config={"temperature": 0.2} 
    )

    return respuesta.text.strip(), fuentes


def mostrar_fuentes(fuentes):
    """Muestra una tabla desplegable con los correos usados como fuente de una respuesta."""
    with st.expander("📎 Fuentes consultadas"):
        df_fuentes = pd.DataFrame(fuentes)
        df_fuentes = df_fuentes.rename(columns={
            "id": "ID correo",
            "asunto": "Asunto",
            "de": "De",
            "fecha": "Fecha primer mensaje",
            "distancia": "Distancia",
        })
        st.dataframe(df_fuentes, use_container_width=True, hide_index=True)


# --- Pantalla de LOGIN ---
# Si nadie ha entrado todavía (empleado_actual es None), mostramos el formulario de acceso.
if st.session_state.empleado_actual is None:
    st.title("📧 Asistente de Correo Electrónico Empresa Enron")
    st.write("Introduce tu nombre para acceder a tu bandeja de entrada.")

    nombre_input = st.text_input("Nombre completo")

    if st.button("Entrar"):
        nombre_normalizado = nombre_input.strip().title()
        if nombre_normalizado in EMPLEADOS:
            st.session_state.empleado_actual = nombre_normalizado
            st.rerun()   # Fuerza a Streamlit a re-ejecutar el script ya con la sesión iniciada
        else:
            st.error("No se ha reconocido ese nombre. Empleados disponibles: " + ", ".join(EMPLEADOS.keys()))

# --- Pantalla del CHAT (una vez ha hecho login) ---
else:
    nombre_empleado = st.session_state.empleado_actual
    carpeta_coleccion = EMPLEADOS[nombre_empleado]

    # Barra lateral con el nombre de la sesión activa y el botón de salir
    with st.sidebar:
        st.write(f"Sesión: **{nombre_empleado}**")
        if st.button("Cerrar sesión"):
            st.session_state.empleado_actual = None
            st.session_state.mensajes = [] 
            st.rerun()

    st.title(f"📧 Bandeja de {nombre_empleado}")

    # Abre la colección ChromaDB persistida en disco correspondiente a este empleado.
    cliente_chroma = chromadb.PersistentClient(path=carpeta_coleccion)
    collection = cliente_chroma.get_collection(name=cliente_chroma.list_collections()[0].name)

    # Cliente de la API de Gemini, autenticado con la clave leída del .env
    client_gemini = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

    for mensaje in st.session_state.mensajes:
        with st.chat_message(mensaje["role"]): 
            st.write(mensaje["content"])
            if mensaje.get("fuentes"):
                mostrar_fuentes(mensaje["fuentes"])

    # Caja de texto fija en la parte inferior de la pantalla
    pregunta = st.chat_input("Escribe tu pregunta sobre tus correos...")

    if pregunta:   # Solo entra aquí si el usuario ha escrito y enviado algo
        st.session_state.mensajes.append({"role": "user", "content": pregunta, "fuentes": None})
        with st.chat_message("user"):
            st.write(pregunta)

        # Retriever: busca los 5 correos más parecidos semánticamente a la pregunta (Configuración A.1, k=5)
        resultado = collection.query(query_texts=[pregunta], n_results=5)

        # Generador: construye la respuesta a partir de esos correos
        respuesta, fuentes = agente_generador_respuesta(client_gemini, pregunta, resultado, nombre_empleado)

        st.session_state.mensajes.append({"role": "assistant", "content": respuesta, "fuentes": fuentes})
        with st.chat_message("assistant"):
            st.write(respuesta)
            if fuentes:
                mostrar_fuentes(fuentes)