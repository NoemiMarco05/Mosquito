
import streamlit as st

# Personalizzazioni CSS

st.markdown(
    """
    <style>
    .stApp {
        background-color: #515e65;
        color: #e7edf1;
    }
    </style>
    """,
    unsafe_allow_html=True)

chiave = st.secrets["superkey"]

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_community.chat_models import ChatOpenAI

st.header("Mosquito Alemasat Help")

from PIL import Image
logo = Image.open("Logo_Scritta.png")
st.image(logo, width=250)

import streamlit as st
from PIL import Image

# Sidebar a sinistra
st.sidebar.title("Mosquito Base e Mosquito Light")

# Descrizione sotto il titolo
st.sidebar.write("Hai bisogno di aiuto per attivare il tuo allarme Mosquito Alemasat? Il nostro chatbot fornisce assistenza esclusiva per i sistemi Mosquito Base e Mosquito Light.")

# Prima immagine
sidebar_image_1 = Image.open("mosquitobase.png")
st.sidebar.image(sidebar_image_1, width=300)

# Seconda immagine
sidebar_image_2 = Image.open("mosquitolight.png")  
st.sidebar.image(sidebar_image_2, width=300)


# st.image(logo, use_column_width=True)

# with st.sidebar:
#  st.title("Carica i tuoi documenti")
#  file = st.file_uploader("Carica il tuo file", type="pdf")
file = "Mosquito_manuale.pdf"

from PyPDF2 import PdfReader

if file is not None:
    testo_letto = PdfReader(file)

    testo = ""
    for pagina in testo_letto.pages:
        testo = testo + pagina.extract_text()
        # st.write(testo)

    # Usiamo il text splitter di Langchain
    testo_spezzato = RecursiveCharacterTextSplitter(
        separators="\n",
        chunk_size=1000, # Numero di caratteri per chunk
        chunk_overlap=150,
        length_function=len
        )

    pezzi = testo_spezzato.split_text(testo)
    # st.write(pezzi)

    # Generazione embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=chiave)

    # Vector store - FAISS (by Facebook)
    vector_store = FAISS.from_texts(pezzi, embeddings)

# --------------------------------------------------
# Gestione prompt
# --------------------------------------------------

    def invia():
      st.session_state.domanda_inviata = st.session_state.domanda
      # salva il contenuto di input in domanda_inviata
      st.session_state.domanda = ""
      # reset dopo invio

    st.text_input("Chiedi al chatbot:", key="domanda", on_change=invia)
    # key="domanda": assegna a st.session_state ciò che scriviamo (domanda)
    # Ogni volta che l’utente modifica il campo e preme Invio,
    # la funzione invia() viene chiamata.

    domanda = st.session_state.get("domanda_inviata", "")
    # Recupera il valore salvato in "domanda_inviata".
    # Se "domanda_inviata" non è ancora stato definito (es. al primo avvio dell'app),
    # allora il valore predefinito sarà "" (secondo argomento dell'istruzione)

# --------------------------------------------------

    if domanda:
      # st.write("Sto cercando le informazioni che mi hai richiesto...")
      rilevanti = vector_store.similarity_search(domanda)

      # Definiamo l'LLM
      llm = ChatOpenAI(
          openai_api_key = chiave,
          temperature = 1.0,
          max_tokens = 1000,
          model_name = "gpt-3.5-turbo-0125")
      # https://platform.openai.com/docs/models/compare

      # Output
      # Chain: prendi la domanda, individua i frammenti rilevanti,
      # passali all'LLM, genera la risposta
      chain = load_qa_chain(llm, chain_type="stuff")
      risposta = chain.run(input_documents = rilevanti, question = domanda)
      st.write(risposta)
