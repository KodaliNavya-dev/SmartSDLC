# SmartSDLC
🧠 SmartSDLC AI Tools

An intelligent developer assistant built with **Streamlit**, powered by **Hugging Face models**, designed to support every phase of the Software Development Life Cycle (SDLC).

> 💻 Runs locally with GPU support (CUDA + RTX 3050 ready)  
> ✅ No internet required once models are downloaded  
> 🆓 100% free to use – no API keys or subscriptions needed

---

🔧 Features

💬 AI Chatbot
Ask any technical or SDLC-related question. Powered by `DialoGPT-small`, running fully offline once downloaded.

📄 PDF Summarizer
Upload a PDF (e.g., project report or documentation), and get a clean summary extracted using `PyMuPDF` and summarized with a transformer model.

💻 Multilingual Code Generator
Generate Python, Java, or C code from simple English task descriptions using the `deepseek-ai/deepseek-coder` model.

🛠️ Code Bug Fixer
Paste buggy code and get it auto-repaired using the `alexjercan/codet5-base-buggy-code-repair` model. Supports multiple languages.

🚀 Installation

1. Clone the repo:
``bash
git clone https://github.com/your-username/smartsdlc-ai-tools.git
cd smartsdlc-ai-tools

2. Create virtual environment and install dependencies:
bash
python -m venv venv
venv\Scripts\activate  # On Windows
pip install -r requirements.txt

3. Download models (one-time setup):
In Python, run this once to cache the models:

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM

AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")

AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-1.3b-base")
AutoModelForCausalLM.from_pretrained("deepseek-ai/deepseek-coder-1.3b-base")

AutoTokenizer.from_pretrained("alexjercan/codet5-base-buggy-code-repair")
AutoModelForSeq2SeqLM.from_pretrained("alexjercan/codet5-base-buggy-code-repair")
This step downloads all models to your Hugging Face cache (usually located at ~/.cache/huggingface/).

4. Run the app
bash
streamlit run app.py
