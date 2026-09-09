# Running the Kivi Phonetic Memory System

This guide outlines the exact steps to install, run, and test the system locally.

## Prerequisites
* Python 3.11+
* A valid Groq API Key (the system uses `groq/compound-mini` for lightning-fast inference).

## 1. Installation

1. Navigate to the project root directory.
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   
   # Windows:
   venv\Scripts\activate
   # macOS/Linux:
   source venv/bin/activate
   ```
3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Set your Groq API key:
   ```bash
   # Windows (PowerShell):
   $env:GROQ_API_KEY="your_api_key_here"
   
   # macOS/Linux:
   export GROQ_API_KEY="your_api_key_here"
   ```
   *(Note: The system contains a fallback hardcoded Groq key for immediate testing if you prefer to skip this step).*

## 2. Start the Backend API

The core system runs as a FastAPI service. 

Run the following command from the project root:
```bash
uvicorn app.main:app --reload --port 8080
```
*The database (`kivi.db`) will automatically initialize itself on startup if it doesn't exist.*

## 3. Start the Interactive UI (Streamlit)

Open a **new terminal window**, activate your virtual environment again, and run the Streamlit frontend:

```bash
streamlit run app_ui.py
```
This will automatically open the interactive web interface in your default browser at `http://localhost:8501`.

## 4. Testing the Edge Cases

You can use the Streamlit UI to test the exact scenarios discussed in the architecture overview.

1. **Populate the Dictionary:**
   Go to the **Learning Engine** -> **Explicit Correction** tab and add:
   * Canonical Word: `Kivi` | Context: `Kivi is an AI company.`
   * Canonical Word: `Aaditya Kshatriya` | Context: `Aaditya works on backend.` | ASR Token: `aditya shatriya`

2. **Test Semantic Disambiguation (Context Bleed Prevention):**
   Go to **Process ASR** and type:
   > *"I spoke with aditya shatriya earlier this morning, and aditya mentioned that the kiwi authentication service is failing in production. After the meeting, I ate a kiwi fruit."*
   
   The system will output:
   > *"I spoke with Aaditya Kshatriya earlier this morning, and Aaditya mentioned that the Kivi authentication service is failing in production. After the meeting, I ate a kiwi fruit."*

3. **Inspect the Results:**
   Go to **Intervention Logs** to see exactly how the LLM Judge evaluated the context and arrived at its True/False decisions in milliseconds.

## 5. Resetting the System

To start fresh and clear all learned vocabulary, navigate to the **User Dictionary** tab in the UI and click the **🚨 Nuke Database** button in the top right corner. This will wipe the SQLite database clean.
