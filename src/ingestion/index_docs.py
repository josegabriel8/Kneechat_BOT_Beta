# src/ingestion/index_docs.py
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document

def extract_text_from_pdf(path: str) -> str:
    reader = PdfReader(path)
    text = []
    for page in reader.pages:
        text.append(page.extract_text() or "")
    return "\n".join(text)

def main():
    load_dotenv()  # carga OPENAI_API_KEY desde .env

    # 1. Rutas a tus PDFs
    pdf_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data")
    pdf_files = [
        os.path.join(pdf_dir, fname)
        for fname in os.listdir(pdf_dir)
        if fname.lower().endswith(".pdf")
    ]

    # 2. Extraer y concatenar textos
    docs = []
    for pdf_path in pdf_files:
        print(f"Extrayendo texto de {os.path.basename(pdf_path)}...")
        raw_text = extract_text_from_pdf(pdf_path)
        docs.append(Document(page_content=raw_text, metadata={"source": os.path.basename(pdf_path)}))

    # 3. Dividir en chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    texts = splitter.split_documents(docs)
    print(f"Total de chunks generados: {len(texts)}")

    # 4. Crear embeddings y almacenar en Chroma
    print("Generando embeddings y almacenando en Chroma...")
    embeddings = OpenAIEmbeddings()
    vectordb = Chroma.from_documents(
        documents=texts,
        embedding=embeddings,
        persist_directory=os.path.join(os.path.dirname(__file__), "..", "..", "data", "chroma_db")
    )
    vectordb.persist()
    print("Indexación completada.")

    # 5. Pequeña prueba de recuperación
    query = "¿Cómo preparar al paciente antes de la cirugía?"
    docs_returned = vectordb.similarity_search(query, k=3)
    print("\n=== Fragmentos recuperados para prueba ===")
    for i, doc in enumerate(docs_returned, 1):
        print(f"\n--- Fragmento {i} (fuente: {doc.metadata['source']}) ---")
        print(doc.page_content[:500].strip(), "...")
    
if __name__ == "__main__":
    main()
