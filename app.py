import streamlit as st
import google.generativeai as genai

st.set_page_config(page_title="AI Data Extractor", page_icon="📋", layout="wide")

st.title("📋 Unstructured Text to Structured Data Transformer")
st.write(
    "A Generative AI application that takes natural, disorganized paragraphs and extracts "
    "structured schemas (Action Items, JSON, or Key-Value pairs) using low-temperature extraction."
)

with st.sidebar:
    st.header("🔑 Configuration")
    api_key = st.text_input("Enter Gemini API Key", type="password", placeholder="AIza...")
    st.markdown("[Get a free Gemini API Key](https://aistudio.google.com/)")

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
            st.error("Please enter your Gemini API Key in the sidebar.")
        elif not raw_text.strip():
            st.warning("Please paste some text to extract.")
        else:
            if format_choice == "Action Items & Summary":
                prompt = f"""
You are an expert Data Extraction Engine. Extract structured data from this text.
Do not hallucinate or extrapolate facts.

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

            with st.spinner("Processing text with Gemini..."):
                try:
                    genai.configure(api_key=api_key.strip())
                    
                    # Target the updated 3.8-flash model
                    model = genai.GenerativeModel(
                        model_name="gemini-3.8-flash",
                        generation_config={"temperature": 0.1}
                    )
                    response = model.generate_content(prompt)
                    
                    if format_choice == "Structured JSON":
                        st.code(response.text, language="json")
                    else:
                        st.markdown(response.text)
                except Exception as e:
                    st.error(f"Error: {str(e)}")
