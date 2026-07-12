import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import time
from dotenv import load_dotenv
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise ValueError("CRITICAL ERROR: 'GOOGLE_API_KEY' not found in your local .env file.")

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

app = FastAPI(title="Document Question Answering System")

#db
CHROMA_DB_DIR = "./chroma_db"
os.makedirs(CHROMA_DB_DIR, exist_ok=True)

#embeddings
embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2-preview", 
    google_api_key=API_KEY
)
vector_store = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embeddings)


class QueryRequest(BaseModel):
    question: str


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Ingests a PDF, chunks it, and stores it in ChromaDB using rate-limiting protection."""
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        loader = PyPDFLoader(temp_path)
        docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )
        chunks = text_splitter.split_documents(docs)

        BATCH_SIZE = 20
        for i in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[i : i + BATCH_SIZE]
            vector_store.add_documents(batch)

            if i + BATCH_SIZE < len(chunks):
                time.sleep(3.0)

        return {
            "message": f"Successfully processed {file.filename} safely under free tier limits.",
            "chunks_created": len(chunks),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def format_docs(docs):
    """Converts retrieved segments into a plain string context."""
    return "\n\n".join(doc.page_content for doc in docs)


@app.post("/query")
async def query_document(request: QueryRequest):
    """Retrieves context and generates an answer using the explicitly authenticated LLM."""
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-3.5-flash", 
            temperature=0, 
            google_api_key=API_KEY
        )

        retriever = vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": 4}
        )

        system_prompt = (
            "You are an assistant for question-answering tasks.\n"
            "Use the following pieces of retrieved context to answer the question.\n"
            "If you don't know the answer, say that you don't know.\n\n"
            "Context:\n{context}\n\n"
            "Question: {question}\n\n"
            "Answer:"
        )
        prompt = ChatPromptTemplate.from_template(system_prompt)

        retrieved_docs = retriever.invoke(request.question)
        formatted_context = format_docs(retrieved_docs)

        rag_chain = prompt | llm | StrOutputParser()

        answer = rag_chain.invoke({
            "context": formatted_context,
            "question": request.question
        })

        return {
            "answer": answer,
            "source_documents": [
                {"page": doc.metadata.get("page", 0), "text": doc.page_content[:200]}
                for doc in retrieved_docs
            ],
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
