# RAG BASED QNA BOT WITH GROQ

## 🚀 Overview
This project implements a Retrieval-Augmented Generation (RAG) based QnA Bot using Groq's open-source LLM, FAISS for vector storage, and SBERT embeddings. The bot allows users to upload documents (PDF, TXT, DOCX), retrieve relevant information, and generate AI-powered answers.

## Screenshots

![Screenshot 2025-03-11 175159](https://github.com/user-attachments/assets/dc72d729-0b2d-4785-a628-9a89721460ca)

![Screenshot 2025-03-11 175226](https://github.com/user-attachments/assets/920fa5b9-b9fb-4fa5-8e4a-332dc7d29d25)

## 🔹 Features
📂 Document Upload: Supports PDF, TXT, and DOCX file formats.
🔍 Efficient Search: Uses FAISS to retrieve relevant document chunks.
🧠 AI-Powered Answers: Uses Groq's LLM for accurate and context-aware responses.
🏆 SBERT Embeddings: Provides fast and efficient document vectorization.
🎨 Interactive UI: Built with Streamlit for a user-friendly experience.

## 🛠 Tech Stack
- LangChain: For integrating LLMs and retrieval-based processing.
- Groq API: Open-source LLM model for generating responses.
- Deepseek-r1: Open source LLM model used to generate text outputs.
- FAISS: Fast retrieval of document chunks.
- Sentence-BERT (SBERT): Embedding model for vector storage.
- Streamlit: UI framework for an interactive chatbot experience.
- Python: Core programming language.

  

## 🎯 How It Works
1️⃣ Upload a DocumentSelect a PDF, TXT, or DOCX file to upload.
The document is processed and stored in FAISS.
2️⃣ Ask a QuestionEnter a question related to the uploaded document.
The bot retrieves relevant chunks using FAISS.
3️⃣ Get AI-Powered AnswersGroq's LLM generates a well-structured response based on the retrieved context.
