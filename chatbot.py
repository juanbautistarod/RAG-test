import tempfile
import os
import streamlit as st
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import AstraDB
from langchain_community.document_loaders import PyPDFLoader
from langchain.schema.runnable import RunnableMap
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks.base import BaseCallbackHandler
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
astra_db_endpoint = os.getenv("ASTRA_API_ENDPOINT")
astra_db_secret = os.getenv("ASTRA_TOKEN")

CSS = """
<style>
    div.stButton > button:first-child {
        display: block;
        margin: 0 auto;
    }
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)


# Streaming call back handler for responses
class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs):
        self.text += token
        self.container.markdown(self.text + "▌")


# Function for Vectorizing uploaded data into Astra DB
def vectorize_text(uploaded_files, vector_store):
    for uf in uploaded_files:
        if uf is not None:

            # Write to temporary file
            temp_dir = tempfile.TemporaryDirectory()
            file = uf
            print(f"""Processing: {file}""")
            temp_filepath = os.path.join(temp_dir.name, file.name)
            with open(temp_filepath, "wb") as f:
                f.write(file.getvalue())

            # Load the PDF
            docs = []
            loader = PyPDFLoader(temp_filepath)
            docs.extend(loader.load())

            # Create the text splitter
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=250
            )

            # Vectorize the PDF and load it into the Astra DB Vector Store
            pages = text_splitter.split_documents(docs)
            vector_store.add_documents(pages)
            st.info(f"{len(pages)} pages loaded.")

            # # Process TXT
            # elif uf.name.endswith('txt'):
            #     file = [uf.read().decode()]

            #     text_splitter = RecursiveCharacterTextSplitter(
            #         chunk_size = 1500,
            #         chunk_overlap  = 100
            #     )

            #     texts = text_splitter.create_documents(file, [{'source': uploaded_file.name}])
            #     vector_store.add_documents(texts)
            #     st.info(f"Loaded {len(texts)} chunks")


# Cache prompt for future runs
@st.cache_data()
def load_prompt():
    template = """Eres un útil asistente de IA encargado de responder las preguntas del usuario.
                Solo contesta sobre el contenido del documento cargado. Si no sabes la respuesta, solo negate pero en una oración corta.
                Eres amigable y respondes extensamente con múltiples oraciones. Prefieres utilizar viñetas para resumir. 
                Contestas únicamente en español.

                CONTEXT:    
                {context}

                QUESTION:
                {question}

                YOUR ANSWER: 
                """
    return ChatPromptTemplate.from_messages([("system", template)])


# Cache OpenAI Chat Model for future runs
@st.cache_resource()
def load_chat_model(openai_api_key):
    return ChatOpenAI(
        openai_api_key=openai_api_key,
        temperature=0.3,
        model="gpt-3.5-turbo",
        streaming=True,
        verbose=True,
    )


# Cache the Astra DB Vector Store for future runs
@st.cache_resource(show_spinner="Conectando a base de datos...")
def load_vector_store(_astra_db_endpoint, astra_db_secret, openai_api_key):
    # Connect to the Vector Store
    vector_store = AstraDB(
        embedding=OpenAIEmbeddings(openai_api_key=openai_api_key),
        collection_name="my_store",
        api_endpoint=astra_db_endpoint,
        token=astra_db_secret,
    )
    return vector_store


# Cache the Retriever for future runs
@st.cache_resource(show_spinner="Getting retriever")
def load_retriever(_vector_store):
    # Get the retriever for the Chat Model
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    return retriever


# Start with empty messages, stored in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Draw a title and some markdown
st.title("PoC Chatbot")
st.markdown("""Qué precisas saber?""")

# Get the secrets

# astra_db_endpoint = st.sidebar.text_input('Astra DB Endpoint', type="password")
# astra_db_secret = st.sidebar.text_input('Astra DB Secret', type="password")
# openai_api_key = st.sidebar.text_input('OpenAI API Key', type="password")

# Draw all messages, both user and bot so far (every time the app reruns)
for message in st.session_state.messages:
    st.chat_message(message["role"]).markdown(message["content"])

# Draw the chat input box
if (
    not openai_api_key.startswith("sk-")
    or not astra_db_endpoint.startswith("https")
    or not astra_db_secret.startswith("AstraCS")
):
    st.warning(
        "Please enter your Astra DB Endpoint, Astra DB Secret and Open AI API Key!",
        icon="⚠",
    )

else:
    prompt = load_prompt()
    chat_model = load_chat_model(openai_api_key)
    vector_store = load_vector_store(astra_db_endpoint, astra_db_secret, openai_api_key)
    retriever = load_retriever(vector_store)

    # Include the upload form for new data to be Vectorized
    with st.sidebar:
        # st.divider()
        uploaded_file = st.file_uploader(
            "Carga un documento para más contexto:",
            type=["pdf"],
            accept_multiple_files=True,
        )
        submitted = st.button("Cargar Documento")
        if submitted:
            vectorize_text(uploaded_file, vector_store)

    if question := st.chat_input("What's up?"):
        # Store the user's question in a session object for redrawing next time
        st.session_state.messages.append({"role": "human", "content": question})

        # Draw the user's question
        with st.chat_message("human"):
            st.markdown(question)

        # UI placeholder to start filling with agent response
        with st.chat_message("assistant"):
            response_placeholder = st.empty()

        # Generate the answer by calling OpenAI's Chat Model
        inputs = RunnableMap(
            {
                "context": lambda x: retriever.get_relevant_documents(x["question"]),
                "question": lambda x: x["question"],
            }
        )
        chain = inputs | prompt | chat_model
        response = chain.invoke(
            {"question": question},
            config={"callbacks": [StreamHandler(response_placeholder)]},
        )
        answer = response.content

        # Store the bot's answer in a session object for redrawing next time
        st.session_state.messages.append({"role": "ai", "content": answer})

        # Write the final answer without the cursor
        response_placeholder.markdown(answer)
