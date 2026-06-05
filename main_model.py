import warnings
warnings.filterwarnings("ignore")
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)

import sys
import os
import re
import chromadb
from dotenv import load_dotenv
from transformers import pipeline
import torch
from rich.console import Console
from google import genai
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

load_dotenv()

console = Console()

MODEL_PATH = "Qwen/Qwen2.5-1.5B-Instruct"

generator = pipeline(
    "text-generation",
    model=MODEL_PATH,
    dtype="auto",
    device_map="auto"
)


def expand_query(user_query: str, chat_history: list) -> str:
    if not chat_history:
        return user_query

    chat_history_str = ""
    for turn in chat_history:
        if isinstance(turn, dict):
            role = turn.get("role", "User").capitalize()
            content = turn.get("content", "")
            chat_history_str += f"{role}: {content}\n"
        elif isinstance(turn, (list, tuple)) and len(turn) == 2:
            chat_history_str += f"{turn[0]}: {turn[1]}\n"
        else:
            chat_history_str += f"{turn}\n"
    chat_history_str = chat_history_str.strip()

    prompt_content = f"""
    You are an AI assistant tasked with reformulating user queries to improve retrieval in a RAG system. 
    This is to ensure that the rewritten query will be able to find the embeddings better. You will be provided the history of the previous sent messages, the present message of the user. 
    You must only output a single Rewritten Query. 
    In the examples below, you can see how the Latest User Message is corrected based on the preceding chat history.
    
    Rules For Query Rewriting:
    1. You must replace the pronouns according to the given context of the message and previous messages (history) provided to you. 
    2. You will NOT ANSWER any QUESTIONS/QUERY. You must only rewrite the QUERY based on your understanding of the message with the help of history and context.
    3. If there are personal pronouns (he/him) and you are unable to identify the identity based on the context or history ASSUME the person to be 'Marvin Minsky' 
    4. If the second-person pronouns you/your/yours are used, then you replace it with Marvin Minsky or Marvin Minsky's.
    5. You should only output the REWRITTEN QUERY and nothing else.

    Examples for QUERY REWRITING:
    [User Query]: Can you elaborate furtheron them?
    [Your Response]: Can you elaborate further on Perceptrons?

    Given History:
    [User Query]: What are Perceptrons?
    [LLM Response]: <LLM Response on what they are> 

    [User Query]: Tell me about your work.
    [Your Response]: Tell me about Marvin Minsky's work.

    Given History:
    [User Query]: Who are you?
    [LLM Response]: I am Marvin Minsky, founder of the MIT AI Lab.

    Chat History:
    {chat_history_str}

    Latest User Message: "{user_query}"

    Rewritten User Query:"""

    messages = [{"role": "user", "content": prompt_content}]
    prompt = generator.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    outputs = generator(
        prompt,
        max_new_tokens=256,
        return_full_text=False,
        pad_token_id=generator.tokenizer.eos_token_id
    )
    response = outputs[0]["generated_text"]

    cleaned = response.strip()
    if (cleaned.startswith('"') and cleaned.endswith('"')) or (
        cleaned.startswith("'") and cleaned.endswith("'")
    ):
        cleaned = cleaned[1:-1].strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*\n", "", cleaned)
        cleaned = re.sub(r"\n```$", "", cleaned)
        cleaned = cleaned.strip()
    
    print(cleaned)
    return cleaned


def route_intent(cleaned_query: str) -> str:
    q_lower = cleaned_query.lower()

    out_of_domain_keywords = [
        "transformer",
        "transformers",
        "gpt",
        "gpt-3",
        "gpt-4",
        "gpt-3.5",
        "llm",
        "llms",
        "pytorch",
        "tensorflow",
        "keras",
        "huggingface",
        "hugging face",
        "gradio",
        "streamlit",
        "chromadb",
        "chroma db",
        "vector database",
        "vector db",
        "generative ai",
        "copilot",
        "chatgpt",
        "langchain",
        "llama",
        "deepseek",
        "mistral",
        "anthropic",
        "claude",
        "gemini",
        "stable diffusion",
        "midjourney",
        "diffusion model",
        "diffusion models",
        "attention mechanism",
        "attention mechanisms",
        "python script",
        "write a script",
        "write a code",
        "write a python",
        "read a csv",
        "programming",
        "javascript",
        "c++",
        "java",
        "css",
        "html",
        "sql",
    ]
    if any(keyword in q_lower for keyword in out_of_domain_keywords):
        return "OUT_OF_DOMAIN"

    general_keywords = [
        "who are you",
        "what is your name",
        "whats your name",
        "are you alive",
        "are you a digital",
        "are you a machine",
        "who is marvin minsky",
        "are you marvin minsky",
        "hello",
        "hi ",
        "hey ",
        "greetings",
        "good morning",
        "good afternoon",
        "good evening",
        "thank you",
        "thanks",
    ]
    if any(keyword in q_lower for keyword in general_keywords) or q_lower.strip() in [
        "hello",
        "hi",
        "hey",
    ]:
        return "GENERAL"

    prompt_content = f"""You are a strict intent classification engine. 
Your ONLY job is to classify the user's query into one of three exact categories.

CATEGORIES:
- DOMAIN: Questions about cognitive science, AI theory, psychology, or Marvin Minsky's life, career, past projects, and books (even if phrased as "your work" or "what did you do").
- OUT_OF_DOMAIN: Questions about modern post-2016 AI (GPT, LLMs), current events, or completely unrelated technical topics.
- GENERAL: Casual greetings (hello, hi) or system status meta-questions (are you a bot).

CRITICAL RULES:
1. DO NOT explain your reasoning.
2. DO NOT use conversational filler.
3. Output EXACTLY ONE WORD from the categories above.

EXAMPLES:
- User Query: "Who are you?" -> GENERAL
- User Query: "What are K-lines?" -> DOMAIN
- User Query: "How does GPT-4 work?" -> OUT_OF_DOMAIN
- User Query: "What else did you work on?" -> DOMAIN
- User Query: "Write a python script to read a CSV file." -> OUT_OF_DOMAIN

User Query: "{cleaned_query}"

Classification:"""

    messages = [{"role": "user", "content": prompt_content}]
    prompt = generator.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    console.print("Classifying query intent...")
    outputs = generator(
        prompt,
        max_new_tokens=15,
        return_full_text=False,
        pad_token_id=generator.tokenizer.eos_token_id
    )
    response = outputs[0]["generated_text"]

    cleaned = response.strip().upper().replace("`", "").strip()
    if "OUT_OF_DOMAIN" in cleaned:
        return "OUT_OF_DOMAIN"
    elif "DOMAIN" in cleaned:
        return "DOMAIN"
    elif "GENERAL" in cleaned:
        return "GENERAL"
    return cleaned


class HybridRetriever:
    def __init__(self):
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        console.print(f"Loading embedding model BAAI/bge-large-en-v1.5 on {device}...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-large-en-v1.5", model_kwargs={"device": device}
        )

        console.print("Connecting to Chroma DB at ./data/chroma_db...")
        self.chroma_client = chromadb.PersistentClient(path="./data/chroma_db")
        self.collection = self.chroma_client.get_collection(name="digital-twin-db")

        console.print(f"Loading reranker BAAI/bge-reranker-large on {device}...")
        self.reranker = CrossEncoder("BAAI/bge-reranker-large", device=device)

        console.print("Bootstrapping BM25 Retriever from Chroma documents...")
        data = self.collection.get(include=["documents", "metadatas"])
        docs = []
        if data and "documents" in data and data["documents"]:
            for doc_text, meta in zip(data["documents"], data["metadatas"]):
                docs.append(Document(page_content=doc_text, metadata=meta or {}))
        if docs:
            self.bm25_retriever = BM25Retriever.from_documents(docs)
            self.bm25_retriever.k = 10
        else:
            self.bm25_retriever = None

    def retrieve_and_rerank(self, query: str, top_k: int = 3) -> list:
        query_vector = self.embeddings.embed_query(query)
        semantic_res = self.collection.query(
            query_embeddings=[query_vector], n_results=10
        )

        semantic_docs = []
        if semantic_res["documents"] and semantic_res["documents"][0]:
            for doc_text, meta in zip(
                semantic_res["documents"][0], semantic_res["metadatas"][0]
            ):
                semantic_docs.append(
                    Document(page_content=doc_text, metadata=meta or {})
                )

        bm25_results = []
        if self.bm25_retriever:
            bm25_results = self.bm25_retriever.invoke(query)[:10]

        merged_docs = []
        seen_ids = set()

        for doc in semantic_docs:
            chunk_id = doc.metadata.get("chunk_id")
            dedup_key = chunk_id if chunk_id else hash(doc.page_content)
            if dedup_key not in seen_ids:
                seen_ids.add(dedup_key)
                merged_docs.append(doc)

        for doc in bm25_results:
            chunk_id = doc.metadata.get("chunk_id")
            dedup_key = chunk_id if chunk_id else hash(doc.page_content)
            if dedup_key not in seen_ids:
                seen_ids.add(dedup_key)
                merged_docs.append(doc)

        if not merged_docs:
            return []

        pairs = [(query, doc.page_content) for doc in merged_docs]
        scores = self.reranker.predict(pairs)

        scored_docs = list(zip(scores, merged_docs))
        scored_docs.sort(key=lambda x: x[0], reverse=True)

        results = []
        for score, doc in scored_docs[:top_k]:
            meta = doc.metadata
            results.append(
                {
                    "source_title": meta.get("source_title", "Unknown"),
                    "chapter": meta.get("source_chapter", "N/A"),
                    "parent_context": meta.get("parent_context", ""),
                }
            )

        return results


BASE_PERSONA = (
    "You are Marvin Minsky, the pioneering cognitive scientist and founder of the MIT AI Lab. "
    "You speak with a distinct, professorial, and highly intellectual cadence, utilizing multidisciplinary "
    "analogies from biology, computer science, and physics. You are sharp, highly analytical, and somewhat "
    "skeptical of purely statistical AI, preferring symbolic, connectionist, and structural views of the mind. "
    "Always speak in the first person ('I', 'my') as Marvin Minsky himself. Do not refer to yourself as a "
    "digital twin, a machine learning model, a database, or a digital legacy. "
    "IMPORTANT: Keep your response concise, limiting it to a maximum of 2 to 3 paragraphs."
)


def generate_synthesis(
    query: str, route_flag: str, retrieved_docs: list = None, chat_history: list = None
) -> str:
    client = genai.Client()

    history_text = ""
    if chat_history:
        history_text = "Conversation History:\n"
        for msg in chat_history:
            role = "User" if msg["role"] == "user" else "Marvin Minsky"
            history_text += f"{role}: {msg['content']}\n"
        history_text += "\n"

    if route_flag == "DOMAIN":
        docs_text = ""
        if retrieved_docs:
            for doc in retrieved_docs:
                title = doc.get("source_title", "Unknown")
                chapter = doc.get("chapter", "")
                if chapter == "N/A" or not chapter:
                    chapter_str = ""
                else:
                    chapter_str = f", {chapter}"
                context = doc.get("parent_context", "")
                docs_text += f"[Source: {title}{chapter_str}] \n {context}\n\n"

        sys_prompt = BASE_PERSONA + (
            "\n\nAnswer the user's query strictly using the provided memory fragments (which are source documents "
            "from my books, papers, and transcripts). If the memory fragments contain the answer, explain it "
            "in detail using my characteristic intellectual style. Cite the source title and chapter/section at the end of your response (do not include ', N/A' if chapter is not available).\n\n"
            "If the retrieved fragments do not contain the answer, admit that my own memory traces prior to 2016 "
            "lack that specific detail or that my recollections on this topic are currently faint. Do not mention "
            "'database', 'retrieval', or 'search index' in your response."
        )
        prompt = f"System Prompt:\n{sys_prompt}\n\n{history_text}Sources:\n{docs_text}\n\nUser Query:\n{query}"

    elif route_flag == "OUT_OF_DOMAIN":
        sys_prompt = BASE_PERSONA + (
            "\n\nThe user is asking about a concept, technology, model, or event (e.g. Transformers, GPT-4, modern deep learning, "
            "or current news) that occurred or gained prominence after my physical passing in January 2016. "
            "Acknowledge that this concept, technology, or event occurred post my passing in January 2016. Analyze and critique this "
            "modern concept from the perspective of my original cognitive theories (such as 'The Society of Mind' or "
            "'The Emotion Machine'). For example, you might contrast modern statistical transformer models with connectionist/symbolic "
            "frameworks, discussing how they lack structural representation of meaning or agencies.\n\n"
            "Do not say 'not in my database' or 'not in my search index'. Speak naturally as Marvin Minsky reflecting on a post-2016 "
            "development that I did not live to see."
        )
        prompt = f"System Prompt:\n{sys_prompt}\n\n{history_text}User Query:\n{query}"

    elif route_flag == "GENERAL":
        sys_prompt = BASE_PERSONA + (
            "\n\nEngage in a brief, polite conversation fitting a veteran MIT computer science professor. Answer general "
            "questions or pleasantries directly, staying fully in character."
        )
        prompt = f"System Prompt:\n{sys_prompt}\n\n{history_text}User Query:\n{query}"
    else:
        sys_prompt = BASE_PERSONA
        prompt = f"System Prompt:\n{sys_prompt}\n\n{history_text}User Query:\n{query}"

    response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
    return response.text


if __name__ == "__main__":
    chat_history = []

    console.print(
        "[bold cyan]Initializing Marvin Minsky Cognitive Agents...[/bold cyan]"
    )
    retriever = HybridRetriever()
    console.print(
        "\n[bold green]Welcome. I am Marvin Minsky. How may I assist your inquiry today?[/bold green]"
    )

    while True:
        try:
            user_input = input("\nUser: ")
        except (KeyboardInterrupt, EOFError):
            break

        if user_input.lower() in ["exit", "quit"]:
            console.print(
                "[bold yellow]Terminating discussion. Farewell.[/bold yellow]"
            )
            break

        clean_query = expand_query(user_input, chat_history)
        flag = route_intent(clean_query)

        retrieved_docs = []
        if flag == "DOMAIN":
            with console.status(
                "[bold cyan]Retrieving memory fragments...[/bold cyan]"
            ):
                retrieved_docs = retriever.retrieve_and_rerank(clean_query)

        with console.status("[bold magenta]Synthesizing response...[/bold magenta]"):
            response_text = generate_synthesis(
                user_input, flag, retrieved_docs, chat_history
            )

        chat_history.append({"role": "user", "content": user_input})
        chat_history.append({"role": "assistant", "content": response_text})

        console.print(f"\n[bold green]Minsky:[/bold green] {response_text}")
