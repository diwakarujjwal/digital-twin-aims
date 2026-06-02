import streamlit as st
import mlx.core as mx
import time

# 0. Initialize thread-local Metal streams for MLX to prevent stream errors in threads
try:
    mx.default_stream(mx.gpu)
except RuntimeError:
    mx.set_default_stream(mx.new_thread_local_stream(mx.gpu))

from main_model import HybridRetriever, expand_query, route_intent, generate_synthesis

st.set_page_config(
    page_title="Marvin Minsky",
    page_icon="🧠",
    layout="wide"
)

# Custom Minsky-style styling (Academic MIT AI Lab theme)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Fira+Code:wght@400;500&family=Lora:ital,wght@0,400;0,600;1,400&family=Inter:wght@300;400;600&display=swap');
    
    /* General body font */
    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Headers */
    .minsky-title {
        font-family: 'Lora', serif;
        font-weight: 600;
        font-size: 2.5rem;
        color: #f3f4f6;
        margin-bottom: 5px;
    }
    .minsky-subtitle {
        font-family: 'Fira Code', monospace;
        color: #10b981;
        font-size: 0.95rem;
        margin-bottom: 25px;
        border-bottom: 1px solid #2d3748;
        padding-bottom: 15px;
    }
    
    /* Target chat message containers by role using the stable :has and data-testid */
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
        background-color: #1a2333 !important;
        border-left: 4px solid #3b82f6 !important;
        border-radius: 8px;
        padding: 15px !important;
        margin-bottom: 12px !important;
    }
    
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
        background-color: #0f141c !important;
        border-left: 4px solid #10b981 !important;
        border-radius: 8px;
        padding: 15px !important;
        margin-bottom: 12px !important;
    }
    
    /* Custom status block styling */
    div[data-testid="stStatusWidget"] {
        border: 1px solid #30363d;
        border-radius: 8px;
        background-color: #161b22;
    }
    
    /* Make st.expander details float over content in header columns, but not in chat messages */
    [data-testid="stColumn"] [data-testid="stExpander"] {
        position: relative;
        border: none !important;
        background: transparent !important;
    }
    [data-testid="stColumn"] [data-testid="stExpander"] > details {
        border: none !important;
    }
    [data-testid="stColumn"] [data-testid="stExpander"] > details > div {
        position: absolute !important;
        z-index: 999999 !important;
        background-color: #0d1117 !important;
        border: 1px solid #30363d !important;
        border-radius: 8px !important;
        box-shadow: 0px 8px 24px rgba(0, 0, 0, 0.8) !important;
        width: 350px !important;
        max-height: 400px;
        overflow-y: auto;
        right: 0 !important;
        top: 45px !important; /* positions details below the header button to prevent blocking click events */
        padding: 15px !important;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to generate condensed context (User query as is, Assistant responses limited to first 2 sentences)
def get_condensed_history(history: list) -> list:
    condensed = []
    for msg in history:
        role = msg["role"]
        content = msg["content"]
        if role == "assistant":
            # Extract first 2 sentences
            sentences = content.split(".")
            sentences = [s.strip() for s in sentences if s.strip()]
            first_few = ". ".join(sentences[:2]).strip()
            if first_few and not first_few.endswith("."):
                first_few += "."
            condensed.append({"role": role, "content": first_few})
        else:
            condensed.append({"role": role, "content": content})
    return condensed


@st.cache_resource(show_spinner=False)
def load_retrieval_engine():
    return HybridRetriever()

retriever = load_retrieval_engine()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
col1, col2 = st.columns([0.65, 0.35])
with col1:
    st.markdown('<div class="minsky-title">Marvin Minsky — Digital Twin</div>', unsafe_allow_html=True)
    st.markdown('<div class="minsky-subtitle">A conversational reflection on the Society of Mind.</div>', unsafe_allow_html=True)
with col2:
    st.markdown("<div style='height: 25px;'></div>", unsafe_allow_html=True)
    sub_col1, sub_col2 = st.columns([0.55, 0.45])
    with sub_col1:
        with st.expander("Context", expanded=False):
            condensed = get_condensed_history(st.session_state.chat_history)
            if condensed:
                for msg in condensed:
                    role_label = msg["role"].upper()
                    st.markdown(f"**{role_label}:** {msg['content']}")
            else:
                st.write("*Context is empty.*")
    with sub_col2:
        if st.button("Reset Memory", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

for message in st.session_state.chat_history:
    with st.chat_message(message["role"], avatar="🧠" if message["role"] == "assistant" else "👤"):
        if message["role"] == "assistant" and "thinking_logs" in message:
            with st.expander("Show thinking", expanded=False):
                for log in message["thinking_logs"]:
                    st.markdown(log)
        st.markdown(message["content"])

def stream_text(text: str):
    for word in text.split(" "):
        yield word + " "
        time.sleep(0.015)

if user_input := st.chat_input("Talk to Marvin Minsky..."):
    history_before_append = get_condensed_history(list(st.session_state.chat_history))
    
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_input)
        
    with st.chat_message("assistant", avatar="🧠"):
        thinking_logs = []
        
        with st.status("Engaging Cognitive Agencies...", expanded=True) as status:
            log_1 = "**Determining domain alignment in the Society of Mind...**"
            st.write(log_1)
            thinking_logs.append(log_1)
            
            raw_flag = route_intent(user_input)
            
            if raw_flag in ['OUT_OF_DOMAIN', 'GENERAL']:
                flag = raw_flag
                clean_query = user_input
                if flag == 'GENERAL':
                    log_2 = "→ *Routing inquiry to conversational pleasantry agency.*"
                else:
                    log_2 = "→ *Identified concept as outside legacy knowledge horizon.*"
                st.write(log_2)
                thinking_logs.append(log_2)
                
                retrieved_docs = []
            else:
                # STAGE 2: Post-Rewrite Router
                log_2 = "**Reconstructing context from short-term memory traces...**"
                st.write(log_2)
                thinking_logs.append(log_2)
                
                clean_query = expand_query(user_input, history_before_append)
                
                log_3 = f"→ *Formulated thought context:* `{clean_query}`"
                st.write(log_3)
                thinking_logs.append(log_3)
                
                log_4 = "**Confirming domain alignment on context-expanded query...**"
                st.write(log_4)
                thinking_logs.append(log_4)
                
                flag = route_intent(clean_query)
                if flag == 'DOMAIN':
                    log_5 = "→ *Inquiry aligned with legacy theoretical paradigms.*"
                    st.write(log_5)
                    thinking_logs.append(log_5)
                    
                    log_6 = "**Arousing associated K-lines to recall past states...**"
                    st.write(log_6)
                    thinking_logs.append(log_6)
                    
                    retrieved_docs = retriever.retrieve_and_rerank(clean_query)
                    if retrieved_docs:
                        log_7 = "**Recalled Memory Contexts:**"
                        st.write(log_7)
                        thinking_logs.append(log_7)
                        for doc in retrieved_docs:
                            title = doc.get('source_title', 'Unknown')
                            chapter = doc.get('chapter', '')
                            # Filter out N/A in status logs
                            if chapter == 'N/A' or not chapter:
                                log_doc = f"- *{title}*"
                            else:
                                log_doc = f"- *{title}* — Chapter: *{chapter}*"
                            st.write(log_doc)
                            thinking_logs.append(log_doc)
                    else:
                        log_7 = "*No memory segments recovered for the current active agents.*"
                        st.write(log_7)
                        thinking_logs.append(log_7)
                elif flag == 'OUT_OF_DOMAIN':
                    log_5 = "→ *Identified concept as outside legacy knowledge horizon.*"
                    st.write(log_5)
                    thinking_logs.append(log_5)
                    retrieved_docs = []
                else:
                    log_5 = "→ *Routing inquiry to conversational pleasantry agency.*"
                    st.write(log_5)
                    thinking_logs.append(log_5)
                    retrieved_docs = []
            
            log_synthesis = "**Reflecting on memory synthesis...**"
            st.write(log_synthesis)
            thinking_logs.append(log_synthesis)
            
            response_text = generate_synthesis(user_input, flag, retrieved_docs, history_before_append)
            
            status.update(label="Cognitive Agencies Synthesis Complete", state="complete", expanded=False)
            
        st.write_stream(stream_text(response_text))
        
        st.session_state.chat_history.append({
            "role": "assistant", 
            "content": response_text,
            "thinking_logs": thinking_logs
        })
        st.rerun()
