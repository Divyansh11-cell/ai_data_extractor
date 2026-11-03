import streamlit as st
import google.generativeai as genai
import time

st.set_page_config(page_title="AI Data Extractor", page_icon="📋", layout="wide")

st.title("📋 Unstructured Text to Structured Data Transformer")
st.write(
    "A Generative AI application that takes natural, disorganized paragraphs and extracts "
    "structured schemas (Action Items, JSON, or Key-Value pairs) using low-temperature extraction."
)

api_key = st.secrets.get("GEMINI_API_KEY", "")

if api_key:
    genai.configure(api_key=api_key.strip())
    model = genai.GenerativeModel(
        model_name="gemini-3.8-flash",
        generation_config={"temperature": 0.1}
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
            if format_choice == "Action Items & Summary":
                prompt = f"""
You are an expert Data Extraction Engine. Extract structured data from this text.
Do not hallucinate or extrapolate facts. Be concise.

Output format:
1. Executive Summary (2 sentences max)
2. Decisions Made (bulleted list, or 'None' if not applicable)
3. Action Items Table: | Task | Assignee (Who) | Deadline |

Source Paragraph:
\"\"\"{raw_text}\"\"\"
"""
            elif format_choice == "Structured JSON":
                prompt = f"""
Extract all key entities, dates, people, decisions, and tasks from the text into a clean, valid JSON object.
Return ONLY valid JSON without markdown code fences or conversational text.

Source Paragraph:
\"\"\"{raw_text}\"\"\"
"""
            elif format_choice == "Key-Value Entities":
                prompt = f"""
Identify and list all named entities, numeric quantities, locations, dates, and topics in a clean Key-Value Markdown list format.

Source Paragraph:
\"\"\"{raw_text}\"\"\"
"""

            # Automatic retry logic for 429 rate limit
            max_retries = 3
            success = False

            for attempt in range(max_retries):
                try:
                    with st.spinner("Processing text with Gemini..."):
                        response = model.generate_content(prompt, stream=True)
                        
                        def stream_chunks():
                            for chunk in response:
                                if chunk.text:
                                    yield chunk.text

                        st.write_stream(stream_chunks)
                        success = True
                        break

                except Exception as e:
                    err_msg = str(e)
                    if "429" in err_msg or "quota" in err_msg.lower():
                        if attempt < max_retries - 1:
                            st.info("Free tier rate limit reached. Waiting a few seconds before retrying...")
                            time.sleep(12)  # Wait before retrying
                        else:
                            st.warning("Google Free Tier limit reached (5 requests/min). Please wait about 30 seconds and try again.")
                    else:
                        st.error(f"Error: {err_msg}")
                        break
