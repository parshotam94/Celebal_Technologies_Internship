```markdown
# 📚 Document Question Answering System (RAG)

An intelligent, production-ready Retrieval-Augmented Generation (RAG) pipeline built to answer questions grounded in your private, custom documents. This system bypasses the limitations of an LLM's static training data by ingesting PDFs, converting them into semantic text chunks, indexing them into a vector store, and sourcing exact context to answer user queries using the latest **Google Gemini Models**.

---

## 🛠️ System Architecture

The workflow splits cleanly into an ingestion stage (Backend) and a user consumption stage (Frontend):

1. **Document Ingestion:** Raw PDFs are parsed using LangChain utility loaders.
2. **Text Chunking:** Text chunks are split into granular semantic segments (1000 characters with a 200-character sliding overlap) to preserve context continuity.
3. **Embedding Strategy:** Text segments are passed to the `gemini-embedding-2-preview` model in throttled batches to prevent API rate exhaustion.
4. **Vector Store:** Vector arrays are indexed locally in a highly performant **ChromaDB** configuration.
5. **Grounded Generation:** Queries map matching indices back to text context blocks, compiling an explicit prompt layout for the **Gemini 3.5 Flash** reasoning model.

---

## 🧰 Tech Stack

* **Backend Framework:** FastAPI (Python)
* **Frontend UI:** Streamlit
* **Orchestration:** LangChain Expression Language (LCEL)
* **Vector Database:** ChromaDB
* **AI Engine:** Google Gemini API (`gemini-3.5-flash` & `gemini-embedding-2-preview`)

---

## 📁 Repository Structure

```text
QA_SYSTEM/
│
├── app.py              # FastAPI Backend API & Core LCEL Pipeline
├── main.py             # Streamlit Interactive User Interface
├── .env                # Local Private Environment Configuration (Ignored by Git)
├── .gitignore          # Rules defining untracked environment extensions
├── requirements.txt    # Project runtime library dependencies
└── chroma_db/          # Persistent local database workspace (Ignored by Git)

```

---

## 🚀 Setup & Installation

### 1. Clone & Navigate

```bash
git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
cd QA_SYSTEM

```

### 2. Configure Environment Secrets

Create a `.env` file in the root folder to house your private API keys safely:

```env
GEMINI_API_KEY=AIzaSyYourActualGoogleAIStudioKeyHere

```

### 3. Install Core Requirements

Ensure your workspace is fully aligned with modern LangChain distributions:

```bash
pip install -r requirements.txt

```

*If a `requirements.txt` file is not present yet, execute:*

```bash
pip install langchain langchain-community langchain-chroma langchain-google-genai pydantic fastapi uvicorn python-multipart python-dotenv streamlit requests

```

---

## 🏃‍♂️ Running the Application

To run the full stack, you will need to open two separate terminal instances.

### Step A: Start the FastAPI Engine

Initialize the background ingestion and processing engine:

```bash
uvicorn app:app --reload --port 8000

```

The API docs will become interactively accessible at `http://localhost:8000/docs`.

### Step B: Launch the Streamlit Interface

In your second terminal window, initiate the UI layer:

```bash
streamlit run main.py

```

The client dashboard will automatically deploy at `http://localhost:8501`.

---

## 💡 Key Features Implemented

* **Rate-Limit Protection:** Built-in loop-throttling inside the upload endpoint splits file payloads into batch arrays of 20 documents, allowing seamless parsing on standard Google AI Studio free tier limits.
* **Pure LCEL Design:** Uses modern LangChain Expression Language piping (`|`), preventing legacy `ModuleNotFoundError` dependency updates from breaking production lifecycles.
* **Total Sovereignty:** Complete decoupling between your local machine environment setups and the workspace project folder configuration.

```

```
