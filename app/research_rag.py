import os
import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import SemanticChunker
from langchain.embeddings import CohereEmbeddings
from langchain.vectorstores import Milvus
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from transformers import LlamaForCausalLM, LlamaTokenizer, pipeline
import torch

@st.cache_resource
def load_llm():
    model_name = "meta-llama/Llama-2-7b-chat-hf"
    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    model = LlamaForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
    
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=512,
        temperature=0.7,
        top_p=0.95,
        repetition_penalty=1.15
    )
    
    llm = HuggingFacePipeline(pipeline=pipe)
    return llm

# Initialize Cohere embeddings
@st.cache_resource
def load_embeddings():
    return CohereEmbeddings(cohere_api_key=os.getenv("COHERE_API_KEY"))

def process_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    embeddings = load_embeddings()
    text_splitter = SemanticChunker(embeddings)
    texts = text_splitter.split_documents(documents)
    return texts

@st.cache_resource
def create_vector_store(texts):
    embeddings = load_embeddings()
    vector_store = Milvus.from_documents(
        texts,
        embeddings,
        connection_args={"host": "localhost", "port": "19530"},
        collection_name="pdf_store"
    )
    return vector_store

def setup_retrieval_qa(vector_store):
    llm = load_llm()
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    return qa_chain

# Streamlit UI
st.title("PDF Q&A Chat")

# Sidebar for PDF upload
st.sidebar.header("Upload PDFs")
uploaded_files = st.sidebar.file_uploader("Choose PDF files", accept_multiple_files=True, type="pdf")

if uploaded_files:
    all_texts = []
    for uploaded_file in uploaded_files:
        with st.spinner(f"Processing {uploaded_file.name}..."):
            pdf_path = f"temp_{uploaded_file.name}"
            with open(pdf_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            texts = process_pdf(pdf_path)
            all_texts.extend(texts)
            os.remove(pdf_path)
    
    with st.spinner("Creating vector store..."):
        vector_store = create_vector_store(all_texts)
        qa_chain = setup_retrieval_qa(vector_store)
    
    st.success("PDFs processed and ready for questions!")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("What would you like to know about the PDFs?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate assistant response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            with st.spinner("Thinking..."):
                response = qa_chain({"query": prompt})
                full_response = response['result']
            message_placeholder.markdown(full_response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response})

else:
    st.warning("Please upload PDF files to start the Q&A process.")

# Add a button to clear chat history
if st.button("Clear Chat History"):
    st.session_state.messages = []
    st.experimental_rerun()
