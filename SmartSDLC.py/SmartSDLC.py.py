import streamlit as st
from transformers import (
    pipeline, AutoTokenizer, AutoModelForCausalLM,
    AutoModelForSeq2SeqLM
)
import fitz  # PyMuPDF
import torch

# Set Streamlit config
st.set_page_config(page_title="SmartSDLC AI Tools", layout="centered")

# Load GPT-2 (Chatbot & Summarizer)
@st.cache_resource
def load_gpt2():
    device = 0 if torch.cuda.is_available() else -1
    return pipeline("text-generation", model="gpt2", device=device)

gpt2 = load_gpt2()

# Load DeepSeek Code Generator
@st.cache_resource
def load_codegen():
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-1.3b-base")
    model = AutoModelForCausalLM.from_pretrained("deepseek-ai/deepseek-coder-1.3b-base").to(
        "cuda" if torch.cuda.is_available() else "cpu")
    return tokenizer, model

code_tokenizer, code_model = load_codegen()

# Load CodeT5 Bug Fixer
@st.cache_resource
def load_fixer_model():
    tokenizer = AutoTokenizer.from_pretrained("alexjercan/codet5-base-buggy-code-repair")
    model = AutoModelForSeq2SeqLM.from_pretrained("alexjercan/codet5-base-buggy-code-repair").to(
        "cuda" if torch.cuda.is_available() else "cpu")
    return tokenizer, model

fix_tokenizer, fix_model = load_fixer_model()

# Sidebar
st.sidebar.title("SmartSDLC AI Tools")
mode = st.sidebar.radio("Choose Mode", [
    "💬 Chatbot",
    "📄 Summarize PDF",
    "💻 Multilingual Code Generator",
    "🛠️ Code Bug Fixer"
])

# --- Mode 1: Chatbot ---
if mode == "💬 Chatbot":
    st.title("💬 AI Chatbot Assistance")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_input = st.text_input("Ask a question:")
    col1, col2 = st.columns([1, 1])

    with col1:
        if st.button("Ask") and user_input:
            with st.spinner("Thinking..."):
                response = gpt2(user_input, max_length=100, do_sample=True)[0]["generated_text"]
            st.session_state.chat_history.append(("You", user_input))
            st.session_state.chat_history.append(("AI", response))

    with col2:
        if st.button("Clear Chat"):
            st.session_state.chat_history = []

    for sender, message in st.session_state.chat_history:
        st.markdown(f"**{sender}:** {message}")

# --- Mode 2: PDF Summarization ---
elif mode == "📄 Summarize PDF":
    st.title("📄 PDF Summarizer")
    pdf_file = st.file_uploader("Upload a PDF file", type="pdf")

    if pdf_file:
        with st.spinner("Extracting text from PDF..."):
            doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
            text = "".join(page.get_text() for page in doc)

        st.success("Text extracted successfully!")

        if st.button("Summarize"):
            with st.spinner("Summarizing..."):
                summary = gpt2(f"Summarize this:\n{text}", max_length=300, do_sample=True)[0]["generated_text"]
            st.subheader("Summary")
            st.write(summary)

# --- Mode 3: Code Generator ---
elif mode == "💻 Multilingual Code Generator":
    st.title("💻 Multilingual Code Generator")

    language = st.selectbox("Choose Language", ["Python", "Java", "C"])
    task_description = st.text_area("Describe what you want the code to do:")

    if st.button("Generate Code") and task_description:
        with st.spinner("Generating code..."):
            prefix = {
                "Python": "# Python program\n",
                "Java": "// Java program\npublic class Main {\n    public static void main(String[] args) {\n",
                "C": "// C program\n#include <stdio.h>\nint main() {\n"
            }
            suffix = {
                "Python": "\n# End of program",
                "Java": "\n    }\n}",
                "C": "\n    return 0;\n}"
            }

            prompt = prefix[language] + "# Task: " + task_description + suffix[language]
            inputs = code_tokenizer(prompt, return_tensors="pt").to(code_model.device)

            outputs = code_model.generate(
                **inputs,
                max_length=256,
                temperature=0.7,
                do_sample=False,
                num_return_sequences=1,
                pad_token_id=code_tokenizer.eos_token_id
            )
            code = code_tokenizer.decode(outputs[0], skip_special_tokens=True)

        st.subheader(f"Generated {language} Code")
        st.code(code, language=language.lower())

# --- Mode 4: Bug Fixer ---
elif mode == "🛠️ Code Bug Fixer":
    st.title("🛠️ Code Bug Fixer")

    language = st.selectbox("Select Code Language", ["Python", "Java", "C"])
    user_code = st.text_area("Paste your buggy code here:")

    if st.button("Fix Code") and user_code.strip():
        with st.spinner("Fixing your code..."):
            prompt = f"Fix the bugs in this {language} code:\n{user_code}\nFixed code:\n"

            inputs = fix_tokenizer(prompt, return_tensors="pt", truncation=True, padding=True).to(fix_model.device)

            outputs = fix_model.generate(
                **inputs,
                max_length=512,
                temperature=0.5,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                pad_token_id=fix_tokenizer.eos_token_id
            )

            decoded = fix_tokenizer.decode(outputs[0], skip_special_tokens=True)
            fixed_code = decoded.replace(prompt, "").strip()

        st.subheader("🔧 Fixed Code")
        st.code(fixed_code, language=language.lower())
