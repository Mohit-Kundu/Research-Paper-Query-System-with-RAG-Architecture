# Research Paper Query System with RAG Architecture

This project implements a Question-Answering system for PDF documents using LLAMA-2, Cohere embeddings, and Milvus vector database. It includes a Streamlit-based chat interface and uses ARES (Automatic Response Evaluation System) to evaluate the quality of generated responses.

## Features

- PDF document processing and semantic chunking
- Question-Answering using LLAMA-2 language model
- Vector storage and retrieval using Milvus
- Streamlit-based chat interface
- ARES evaluation for response quality assessment
- Docker containerization for easy deployment

## Prerequisites

- Docker and Docker Compose
- Cohere API key

## Project Structure
project_root/
│
├── app/
│ ├── pdf_qa_app.py
│ └── requirements.txt
│
├── Dockerfile
└── docker-compose.yml

## Setup and Installation

1. Clone the repository:

2. Set up your Cohere API key as an environment variable:

3. Build and run the Docker containers:
docker-compose up --build


4. Access the application at `http://localhost:8501`

## Usage

1. Upload PDF files using the sidebar.
2. Wait for the PDFs to be processed and indexed.
3. Ask questions about the content of the PDFs in the chat interface.
4. View the generated answers and their ARES evaluations.

## Components

- **LLAMA-2**: Large Language Model for generating responses and evaluations.
- **Cohere Embeddings**: Used for creating semantic embeddings of text.
- **Milvus**: Vector database for efficient storage and retrieval of document chunks.
- **Streamlit**: Web application framework for the user interface.
- **ARES**: Automatic Response Evaluation System for assessing response quality.

## Docker Services

- `pdf_qa_app`: The main Streamlit application.
- `standalone`: Milvus standalone service.
- `etcd`: Distributed key-value store used by Milvus.
- `minio`: Object storage used by Milvus.

## Customization

- Adjust the LLAMA-2 model parameters in `pdf_qa_app.py` to fine-tune performance.
- Modify the ARES evaluation criteria in the `evaluate_response` function.
- Update the Streamlit UI in `pdf_qa_app.py` to change the application's appearance.