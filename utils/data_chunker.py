import os
import re
import sys
import uuid
import torch
from pathlib import Path
from tqdm import tqdm
import chromadb
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

def clean_audio_title(filename: str) -> str:
    stem = Path(filename).stem
    pattern = re.compile(r"Marvin\s+Mins?[ky]{2}\s*[\-–—]\s*(.*?)\s*(?:\(\d+[\u29f8/]\d+\))?$", re.IGNORECASE)
    match = pattern.search(stem)
    if match:
        raw_title = match.group(1).strip()
        return raw_title.strip("'\"“”‘’＂").replace("_", " ")
    else:
        raw_title = stem.strip()
        return raw_title.strip("'\"“”‘’＂").replace("_", " ").capitalize()

def clean_general_title(name: str) -> str:
    return name.replace("_", " ").strip().capitalize()

def read_file_content(path: Path) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except UnicodeDecodeError:
        with open(path, "r", encoding="latin-1") as f:
            return f.read().strip()

def split_markdown_into_chapters(text: str) -> list[tuple[str, str]]:
    lines = text.split("\n")
    chapters = []
    current_chapter_title = "Introduction"
    current_chapter_lines = []
    
    header_pattern = re.compile(r"^#{1,3}\s+(.+)$")
    
    for line in lines:
        match = header_pattern.match(line)
        if match:
            if current_chapter_lines:
                chapters.append((current_chapter_title, "\n".join(current_chapter_lines)))
                current_chapter_lines = []
            current_chapter_title = match.group(1).strip().strip("*_#")
        else:
            current_chapter_lines.append(line)
            
    if current_chapter_lines:
        chapters.append((current_chapter_title, "\n".join(current_chapter_lines)))
        
    return chapters

def main():
    base_dir = Path("./data/books_markdown")
    audio_dir = Path("./data/audio_transcript")
    
    if not base_dir.exists() and not audio_dir.exists():
        print(f"Error: Missing data directories in {os.getcwd()}")
        sys.exit(1)
        
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Initializing embeddings on device: {device}")
    
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-large-en-v1.5",
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True}
    )
    
    semantic_splitter = SemanticChunker(
        embeddings,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=85
    )
    
    child_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=256,
        chunk_overlap=40
    )
    
    chroma_client = chromadb.PersistentClient(path="./data/chroma_db")
    collection = chroma_client.get_or_create_collection(name="digital-twin-db")
    
    documents_to_process = []
    
    if audio_dir.exists():
        for f in sorted(list(audio_dir.glob("*.md"))):
            try:
                content = read_file_content(f)
                if content:
                    documents_to_process.append({
                        "text": content,
                        "source_title": clean_audio_title(f.name),
                        "source_type": "transcript",
                        "source_chapter": ""
                    })
            except Exception as e:
                print(f"Error reading transcript {f.name}: {e}")
                
    if base_dir.exists():
        for txt_file in sorted(list(base_dir.glob("*.txt"))):
            try:
                content = read_file_content(txt_file)
                if content:
                    documents_to_process.append({
                        "text": content,
                        "source_title": clean_general_title(txt_file.stem),
                        "source_type": "article",
                        "source_chapter": ""
                    })
            except Exception as e:
                print(f"Error reading standalone article {txt_file.name}: {e}")
                
        book_folders = ["perceptrons", "society_of_mind", "emotion_machine"]
        for subfolder in sorted(list(base_dir.iterdir())):
            if subfolder.is_dir():
                md_file = subfolder / f"{subfolder.name}.md"
                if md_file.exists():
                    try:
                        content = read_file_content(md_file)
                        if not content:
                            continue
                            
                        title = clean_general_title(subfolder.name)
                        
                        if subfolder.name in book_folders:
                            chapters = split_markdown_into_chapters(content)
                            for ch_title, ch_text in chapters:
                                if ch_text.strip():
                                    documents_to_process.append({
                                        "text": ch_text.strip(),
                                        "source_title": title,
                                        "source_type": "book",
                                        "source_chapter": ch_title
                                    })
                        else:
                            documents_to_process.append({
                                        "text": content,
                                        "source_title": title,
                                        "source_type": "book",
                                        "source_chapter": ""
                            })
                    except Exception as e:
                        print(f"Error reading paper/book {md_file.name}: {e}")
                        
    if not documents_to_process:
        print("No documents found to chunk.")
        sys.exit(1)
        
    print(f"Loaded {len(documents_to_process)} document parts for processing.")
    
    ids_batch = []
    embeddings_batch = []
    metadatas_batch = []
    docs_batch = []
    
    batch_size = 64
    
    def flush_batch():
        if ids_batch:
            collection.add(
                ids=ids_batch,
                embeddings=embeddings_batch,
                metadatas=metadatas_batch,
                documents=docs_batch
            )
            ids_batch.clear()
            embeddings_batch.clear()
            metadatas_batch.clear()
            docs_batch.clear()
            
    pbar = tqdm(documents_to_process, desc="Processing documents")
    for doc in pbar:
        try:
            desc = doc["source_title"]
            if doc["source_chapter"]:
                desc += f" - {doc['source_chapter']}"
            pbar.set_description(f"Processing: {desc[:40]}")
            
            parent_chunks = semantic_splitter.split_text(doc["text"])
            
            for parent_idx, parent_chunk in enumerate(parent_chunks):
                if not parent_chunk.strip():
                    continue
                    
                child_chunks = child_splitter.split_text(parent_chunk)
                
                for child_idx, child_chunk in enumerate(child_chunks):
                    if not child_chunk.strip():
                        continue
                        
                    chunk_id = f"{doc['source_title']}_p{parent_idx}_c{child_idx}_{str(uuid.uuid4())[:8]}"
                    
                    metadata = {
                        "chunk_id": chunk_id,
                        "source_title": doc["source_title"],
                        "source_type": doc["source_type"],
                        "parent_context": parent_chunk
                    }
                    if doc["source_chapter"]:
                        metadata["source_chapter"] = doc["source_chapter"]
                        
                    ids_batch.append(chunk_id)
                    metadatas_batch.append(metadata)
                    docs_batch.append(child_chunk)
                    
                    if len(ids_batch) >= batch_size:
                        embeddings_batch.extend(embeddings.embed_documents(docs_batch))
                        flush_batch()
                        
        except Exception as e:
            print(f"Error chunking document {doc['source_title']}: {e}")
            continue
            
    if ids_batch:
        embeddings_batch.extend(embeddings.embed_documents(docs_batch))
        flush_batch()
        
    print("Database ingestion completed successfully.")

if __name__ == "__main__":
    main()
