import streamlit as st
import google.generativeai as genai
import time

st.set_page_config(page_title="AI Data Extractor", page_icon="📋", layout="wide")

st.title("📋 Unstructured Text to Structured Data Transformer")
st.write(
    "A Generative AI application that takes natural, disorganized paragraphs and extracts "
    "structured schemas (Action Items, JSON, or Key-Value pairs) using high-speed streaming."
)

api_key = st.secrets.get("GEMINI_API_KEY", "")

# Cache the model initialization so Streamlit doesn't reconnect on every run
@st.cache_resource
def get_model(key):
    genai.configure(api_key=key.strip())
    return genai.GenerativeModel(
        model_name="gemini-3.8-flash",
        generation_config={
            "temperature": 0.1,
            "max_output_tokens": 800  # Caps length so responses finish 2x faster
        }
    )

col1, col2 = st.columns(2)

with col1:
    st.subheader("Input Text")
    raw_text = st.text_area(
        "Paste messy text, meeting notes, or transcripts:",
        height=260,
        placeholder="Paste your unorganized paragraph here..."
    )
    
    format_choice = st.selectbox(
        "Choose output structure:",
        ("Action Items & Summary", "Structured JSON", "Key-Value Entities")
    )
    
    extract_btn = st.button("Extract Structured Data", type="primary", use_container_width=True)

with col2:
    st.subheader("Structured Output")
    
    if extract_btn:
        if not api_key:
            st.error("API Key not found! Please check GEMINI_API_KEY in Streamlit Cloud Secrets.")
        elif not raw_text.strip():
            st.warning("Please paste some text to extract.")
        else:
            # Concise prompts generate output much faster
            if format_choice == "Action Items & Summary":
                prompt = f"""Extract concise structured data from this text. No filler text.
Format:
1. Executive Summary (1-2 sentences)
2. Decisions: (bullet list or 'None')
3. Action Items Table: | Task | Assignee | Deadline |

Text:
\"\"\"{raw_text}\"\"\""""
            elif format_choice == "Structured JSON":
                prompt = f"""Extract entities, dates, people, decisions, and tasks into raw, compact JSON only. No markdown formatting.

Text:
\"\"\"{raw_text}\"\"\""""
            elif format_choice == "Key-Value Entities":
                prompt = f"""Extract named entities, quantities, dates, and topics in a concise Key-Value list.

Text:
\"\"\"{raw_text}\"\"\""""

            model = get_model(api_key)
            
            # Fast execution with quick 4-second backoff
            max_retries = 2
            for attempt in range(max_retries):
                try:
                    with st.spinner("⚡ Extracting..."):
                        response = model.generate_content(prompt, stream=True)
                        
                        def stream_chunks():
                            for chunk in response:
                                if chunk.text:
                                    yield chunk.text

                        st.write_stream(stream_chunks)
                        break

                except Exception as e:
                    err_msg = str(e)
                    if "429" in err_msg or "quota" in err_msg.lower():
                        if attempt < max_retries - 1:
                            st.info("API busy. Retrying in 4 seconds...")
                            time.sleep(4)
                        else:
                            st.warning("Free-tier quota limit reached (5 requests/min). Please wait 20 seconds before your next query.")
                    else:
                        st.error(f"Error: {err_msg}")
                        break
