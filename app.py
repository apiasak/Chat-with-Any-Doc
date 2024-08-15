import streamlit as st
import os
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader, Docx2txtLoader, UnstructuredExcelLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from tempfile import NamedTemporaryFile
from langchain.schema import Document

# Available models
MODELS = {
    "GPT-4o-Mini": "gpt-4o-mini",
    "GPT-4o": "gpt-4o",
    "GPT-4o-2024-08-06": "gpt-4o-2024-08-06"
}

# Streamlit app
st.title("☕️ Chat with your document ")

# Sidebar for configurations
st.sidebar.header("Configuration")

# API Key input
api_key = st.sidebar.text_input("Enter your OpenAI API Key", type="password")

# Model selection
selected_model = st.sidebar.selectbox("Select AI Model", list(MODELS.keys()))

# Clear chat button
if st.sidebar.button("Clear Chat"):
    st.session_state.messages = []
    st.session_state.chain = None
    st.rerun()

# Initialize ChatOpenAI and embeddings
@st.cache_resource(show_spinner=False)
def get_openai_model(api_key, model_name):
    return ChatOpenAI(temperature=0, model_name=model_name, openai_api_key=api_key)

@st.cache_resource(show_spinner=False)
def get_embeddings(api_key):
    return OpenAIEmbeddings(openai_api_key=api_key)

# Function to get file type and loader
def get_file_loader(file):
    file_extension = os.path.splitext(file.name)[1].lower()
    if file_extension == '.pdf':
        return PyPDFLoader
    elif file_extension == '.txt':
        return TextLoader
    elif file_extension == '.csv':
        return CSVLoader
    elif file_extension in ['.docx', '.doc']:
        return Docx2txtLoader
    elif file_extension in ['.xlsx', '.xls']:
        return UnstructuredExcelLoader
    else:
        raise ValueError(f"Unsupported file type: {file_extension}")

# Main app logic
if api_key:
    chat = get_openai_model(api_key, MODELS[selected_model])
    embeddings = get_embeddings(api_key)

    # File uploader
    uploaded_file = st.file_uploader("Choose a file", type=["pdf", "txt", "csv", "docx", "doc", "xlsx", "xls"])

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chain" not in st.session_state:
        st.session_state.chain = None

    # Create chain from uploaded file
    if uploaded_file is not None:
        try:
            with NamedTemporaryFile(delete=False, suffix=f'.{uploaded_file.name.split(".")[-1]}') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name

            # Get appropriate loader
            loader_class = get_file_loader(uploaded_file)
            loader = loader_class(tmp_file_path)
            documents = loader.load()

            if not documents:
                raise ValueError("No content could be extracted from the file.")

            text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
            texts = text_splitter.split_documents(documents)

            if not texts:
                raise ValueError("No text could be extracted from the document.")

            # Create vector store and chain
            db = FAISS.from_documents(texts, embeddings)
            st.session_state.chain = ConversationalRetrievalChain.from_llm(
                llm=chat, 
                retriever=db.as_retriever(),
                return_source_documents=True
            )

            os.unlink(tmp_file_path)  # Remove temporary file

            st.success(f"{uploaded_file.name} uploaded and processed successfully!")
        except Exception as e:
            st.error(f"An error occurred while processing the file: {str(e)}")
            st.session_state.chain = None

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User input
    if prompt := st.chat_input("Ask about your document"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        if st.session_state.chain:
            try:
                with st.chat_message("assistant"):
                    response = st.session_state.chain({"question": prompt, "chat_history": []})
                    st.markdown(response['answer'])
                    st.session_state.messages.append({"role": "assistant", "content": response['answer']})
            except Exception as e:
                st.error(f"An error occurred while generating the response: {str(e)}")
        else:
            st.warning("Please upload a document first.")

    if not uploaded_file:
        st.info("Please upload a document to start chatting.")
else:
    st.warning("Please enter your OpenAI API key in the sidebar to start using the app.")