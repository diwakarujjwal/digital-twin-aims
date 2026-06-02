# Digital Twin - Marvin Minsky

Conversational RAG system utilizing the books and audio interviews of Marvin Minsky and his historical research on cognitive science and artificial intelligence (such as the Society of Mind and The Emotion Machine theories). The system uses local models for query analysis and intent routing on Apple Silicon, combined with Gemini Flash for persona-based response synthesis.

## Core Features

* **First-Person Persona**: Speaks in the first person as Marvin Minsky. It analyzes and critiques topics in Minsky's style, avoiding references to being an AI model, digital twin, or database.
* **Double-Router Pipeline**:
  * Stage 1: Fast-routes casual greetings (routed as GENERAL) or post-2016 modern machine learning questions (routed as OUT_OF_DOMAIN) using keyword pre-filtering.
  * Stage 2: Rewrites legacy domain queries (routed as DOMAIN) using context history for vector search retrieval.
* **Hybrid RAG Retriever**: Combines semantic vector search (local BAAI/bge-large-en-v1.5 embeddings) and keyword-based search (BM25) over indexed materials stored in a local Chroma database. Candidates are reranked using a local Cross-Encoder (BAAI/bge-reranker-large).
* **Synthesis Engine**: Uses gemini-2.5-flash with conversation history context to construct responses. Responses are limited to 2-3 paragraphs and include clean document citations.
* **Streamlit Interface**: A web UI designed with a clean academic dark theme, typewriter streaming, show thinking execution logs, and short-term conversation context management.

## File Reference

* **main_model.py**: The computational engine containing query expansion, intent classification routing, retrieval, and response synthesis logic.
* **app.py**: The Streamlit user interface, handling layout, visual styles, text streaming, and state tracking.
* **utils/data_chunker.py**: Parses books and audio transcripts using semantic splitting and recursive token splitters, then writes them to the local Chroma vector database.
* **utils/pdf_text_extractor.py**: Converts PDF books into clean Markdown files, filtering page artifacts.
* **utils/audio_transcrber.py**: Transcribes raw interview audio using Whisper with MPS GPU acceleration.

## Setup and Running

1. **Virtual Environment**:
   Set up the local environment and install requirements:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Environment Configuration**:
   Configure the Gemini API key in a `.env` file at the root:
   ```env
   GEMINI_API_KEY=your_api_key_here
   ```

3. **Database Ingestion (Optional)**:
   To parse and index raw source documents:
   ```bash
   python utils/pdf_text_extractor.py
   python utils/audio_transcrber.py
   python utils/data_chunker.py
   ```

4. **Running the Web UI**:
   Start the Streamlit application:
   ```bash
   streamlit run app.py
   ```
