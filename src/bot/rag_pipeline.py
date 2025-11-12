import os
import json
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

def extract_text_from_pdf(path: str) -> str:
    """Extract text from a PDF file."""
    reader = PdfReader(path)
    text = []
    for page in reader.pages:
        text.append(page.extract_text() or "")
    return "\n".join(text)

def extract_text_from_json(path: str) -> str:
    """Extract text from a JSON file."""
    try:
        with open(path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            
        # If it's a list of objects with title and content
        if isinstance(data, list):
            text_parts = []
            for item in data:
                if isinstance(item, dict):
                    # Extract title and content if they exist
                    if 'title' in item:
                        text_parts.append(f"## {item['title']}")
                    if 'content' in item:
                        text_parts.append(item['content'])
                    # Handle other possible fields
                    for key, value in item.items():
                        if key not in ['title', 'content'] and isinstance(value, str):
                            text_parts.append(f"{key}: {value}")
            return "\n\n".join(text_parts)
        
        # If it's a single object, extract all text values
        elif isinstance(data, dict):
            text_parts = []
            for key, value in data.items():
                if isinstance(value, str):
                    text_parts.append(f"{key}: {value}")
                elif isinstance(value, (list, dict)):
                    text_parts.append(f"{key}: {str(value)}")
            return "\n\n".join(text_parts)
        
        # If it's just a string or other type
        else:
            return str(data)
            
    except Exception as e:
        print(f"Error reading JSON file {path}: {e}")
        return ""

def ingest_documents(data_dir: str):
    """Ingest documents (PDFs and JSON files) from a directory and its subdirectories with priority handling."""
    # Priority files (Spanish, most important) - can be PDF or JSON
    priority_files = [
        "Main Knowledge.pdf"
    ]
    
    priority_docs = []
    secondary_docs = []
    
    # Process files in the root directory
    supported_extensions = ['.pdf', '.json']
    data_files = [
        os.path.join(data_dir, fname)
        for fname in os.listdir(data_dir)
        if any(fname.lower().endswith(ext) for ext in supported_extensions)
    ]
    
    for file_path in data_files:
        filename = os.path.basename(file_path)
        print(f"Processing {filename}...")
        try:
            # Determine file type and extract text accordingly
            if filename.lower().endswith('.pdf'):
                raw_text = extract_text_from_pdf(file_path)
            elif filename.lower().endswith('.json'):
                raw_text = extract_text_from_json(file_path)
            else:
                print(f"  -> Skipping unsupported file type: {filename}")
                continue
                
            if not raw_text.strip():
                print(f"  -> Warning: No text extracted from {filename}")
                continue
                
            doc = Document(page_content=raw_text, metadata={"source": filename})
            
            # Check if this is a priority document
            if filename in priority_files:
                priority_docs.append(doc)
                print(f"  -> Added to PRIORITY sources")
            else:
                secondary_docs.append(doc)
                print(f"  -> Added to secondary sources")
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    # Process files in subdirectories (these are secondary by default)
    for subdir in os.listdir(data_dir):
        subdir_path = os.path.join(data_dir, subdir)
        if os.path.isdir(subdir_path):
            print(f"Processing subdirectory: {subdir}")
            sub_data_files = [
                os.path.join(subdir_path, fname)
                for fname in os.listdir(subdir_path)
                if any(fname.lower().endswith(ext) for ext in supported_extensions)
            ]
            
            for file_path in sub_data_files:
                filename = os.path.basename(file_path)
                print(f"Processing {filename}...")
                try:
                    # Determine file type and extract text accordingly
                    if filename.lower().endswith('.pdf'):
                        raw_text = extract_text_from_pdf(file_path)
                    elif filename.lower().endswith('.json'):
                        raw_text = extract_text_from_json(file_path)
                    else:
                        print(f"  -> Skipping unsupported file type: {filename}")
                        continue
                        
                    if not raw_text.strip():
                        print(f"  -> Warning: No text extracted from {filename}")
                        continue
                        
                    doc = Document(page_content=raw_text, metadata={"source": f"{subdir}/{filename}"})
                    secondary_docs.append(doc)
                    print(f"  -> Added to secondary sources")
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
    
    print(f"\nSUMMARY:")
    print(f"Priority documents: {len(priority_docs)}")
    print(f"Secondary documents: {len(secondary_docs)}")
    print(f"Supported file types: {', '.join(supported_extensions)}")
    
    return priority_docs, secondary_docs

def deduplicate_chunks(texts, embeddings, similarity_threshold=0.95):
    """Remove duplicate chunks based on exact text match or embedding similarity > threshold."""
    if not texts:
        return []
    
    unique_texts = []
    seen_content = set()
    
    print(f"Deduplicating {len(texts)} chunks...")
    
    # First pass: remove exact duplicates
    for doc in texts:
        content = doc.page_content.strip()
        if content and content not in seen_content:
            seen_content.add(content)
            unique_texts.append(doc)
    
    print(f"After exact dedup: {len(unique_texts)} chunks")
    
    # Second pass: remove near-duplicates using embeddings
    if len(unique_texts) > 1:
        # Create embeddings for all unique texts
        texts_content = [doc.page_content for doc in unique_texts]
        embedded = embeddings.embed_documents(texts_content)
        
        # Compare embeddings and keep only dissimilar ones
        final_texts = [unique_texts[0]]  # Always keep first
        final_embeddings = [embedded[0]]
        
        for i in range(1, len(unique_texts)):
            current_emb = embedded[i]
            is_duplicate = False
            
            # Check similarity with all kept embeddings
            for kept_emb in final_embeddings:
                # Cosine similarity
                similarity = sum(a * b for a, b in zip(current_emb, kept_emb))
                if similarity > similarity_threshold:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                final_texts.append(unique_texts[i])
                final_embeddings.append(current_emb)
        
        print(f"After similarity dedup: {len(final_texts)} chunks")
        return final_texts
    
    return unique_texts

def create_faiss_indexes(priority_docs, secondary_docs):
    """Create separate FAISS indexes for priority and secondary documents."""
    # Larger chunks as requested (~1000 characters)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=300,
        # separators=["\n\n", "\n", " ", ""]
        separators=["\n\n", "\n", ". ", " "]
    )

    # Use LaBSE for multilingual embeddings (works well with Spanish and English)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/LaBSE",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True, 'batch_size': 32}
    )

    # Create priority index (no chunk limit)
    priority_texts = splitter.split_documents(priority_docs) if priority_docs else []
    print(f"Priority chunks created (before dedup): {len(priority_texts)}")
    
    # Deduplicate priority chunks
    if priority_texts:
        priority_texts = deduplicate_chunks(priority_texts, embeddings)
    
    # Create secondary index
    secondary_texts = splitter.split_documents(secondary_docs) if secondary_docs else []
    print(f"Secondary chunks created (before dedup): {len(secondary_texts)}")
    
    # Deduplicate secondary chunks
    if secondary_texts:
        secondary_texts = deduplicate_chunks(secondary_texts, embeddings)
    
    # Create separate vector databases
    priority_vectordb = FAISS.from_documents(priority_texts, embeddings) if priority_texts else None
    secondary_vectordb = FAISS.from_documents(secondary_texts, embeddings) if secondary_texts else None
    
    return priority_vectordb, secondary_vectordb

class HierarchicalRetriever:
    """A retriever that searches priority documents first, optionally falls back to secondary documents."""
    
    def __init__(self, priority_vectordb, secondary_vectordb, use_secondary=False, similarity_threshold=0.7):
        self.priority_vectordb = priority_vectordb
        self.secondary_vectordb = secondary_vectordb
        self.use_secondary = use_secondary
        self.similarity_threshold = similarity_threshold
    
    def get_relevant_documents(self, query, k=8):
        """Retrieve documents with hierarchical search strategy."""
        all_results = []
        
        # Always search in priority documents first
        if self.priority_vectordb:
            priority_results = self.priority_vectordb.similarity_search(query, k=k)
            print(f"Found {len(priority_results)} results in PRIORITY sources")
            all_results.extend(priority_results)
            
            # If not using secondary or we have enough results, return
            if not self.use_secondary or len(all_results) >= k:
                return all_results[:k]
        
        # If use_secondary is True and we need more results, search secondary documents
        remaining_k = k - len(all_results)
        if remaining_k > 0 and self.use_secondary and self.secondary_vectordb:
            print(f"Searching SECONDARY sources for {remaining_k} additional results...")
            secondary_results = self.secondary_vectordb.similarity_search(query, k=remaining_k)
            all_results.extend(secondary_results)
        
        return all_results[:k]

def main():
    load_dotenv()

    # Step 1: Ingest documents with priority handling
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data"))
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"The data directory does not exist: {data_dir}")
    priority_docs, secondary_docs = ingest_documents(data_dir)

    # Step 2: Create hierarchical FAISS indexes
    print("Creating hierarchical FAISS indexes...")
    priority_vectordb, secondary_vectordb = create_faiss_indexes(priority_docs, secondary_docs)

    # Step 3: Create hierarchical retriever
    retriever = HierarchicalRetriever(priority_vectordb, secondary_vectordb)

    # Step 4: Test retrieval
    query = "¿Cómo preparar al paciente antes de la cirugía?"
    print(f"\nQuery: {query}")
    results = retriever.get_relevant_documents(query, k=8)

    print("\n=== Retrieved Chunks ===")
    for i, result in enumerate(results, 1):
        print(f"\n--- Chunk {i} (source: {result.metadata['source']}) ---")
        print(result.page_content[:500].strip(), "...")

def get_retriever(use_secondary=False):
    """
    Create and return a hierarchical retriever from the FAISS indexes.
    
    Args:
        use_secondary (bool): Whether to use secondary sources. Default is False (only priority sources).
    """
    load_dotenv()

    # Try to load existing indexes first
    priority_vectordb, secondary_vectordb = load_existing_indexes()
    
    if priority_vectordb:
        print("Using existing FAISS indexes...")
        retriever = HierarchicalRetriever(priority_vectordb, secondary_vectordb, use_secondary=use_secondary)
        return retriever
    
    print("Creating new FAISS indexes...")
    # Step 1: Ingest documents with priority handling
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data"))
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"The data directory does not exist: {data_dir}")
    priority_docs, secondary_docs = ingest_documents(data_dir)

    # Step 2: Create hierarchical FAISS indexes
    print("Creating hierarchical FAISS indexes...")
    priority_vectordb, secondary_vectordb = create_faiss_indexes(priority_docs, secondary_docs)

    # Step 3: Create and return hierarchical retriever
    retriever = HierarchicalRetriever(priority_vectordb, secondary_vectordb, use_secondary=use_secondary)
    
    # Persist the indexes for reuse
    if priority_vectordb:
        priority_vectordb.save_local("faiss_index_priority")
    if secondary_vectordb:
        secondary_vectordb.save_local("faiss_index_secondary")

    return retriever

def get_simple_retriever():
    """
    Create and return a simple retriever that works with standard LangChain chains.
    Uses ONLY priority documents (Main Knowledge.pdf) by default.
    """
    load_dotenv()

    # Try to load existing index first
    _, _, simple_vectordb = load_existing_indexes()
    
    if simple_vectordb:
        print("Using existing simple FAISS index (priority sources only)...")
        return simple_vectordb.as_retriever(search_kwargs={"k": 8})

    print("Creating new simple FAISS index (priority sources only)...")
    # Step 1: Ingest documents with priority handling
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data"))
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"The data directory does not exist: {data_dir}")
    priority_docs, secondary_docs = ingest_documents(data_dir)

    # Step 2: Use ONLY priority documents (not secondary)
    all_docs = priority_docs
    print(f"Using {len(priority_docs)} priority documents, ignoring {len(secondary_docs)} secondary documents")
    
    # Step 3: Create single FAISS index
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=300,
        separators=["\n\n", "\n", " ", ""]
    )
    texts = splitter.split_documents(all_docs)
    print(f"Total chunks created from priority docs (before dedup): {len(texts)}")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/LaBSE",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True, 'batch_size': 32}
    )
    
    # Deduplicate chunks
    texts = deduplicate_chunks(texts, embeddings)
    
    vectordb = FAISS.from_documents(texts, embeddings)
    
    # Persist the index for reuse
    vectordb.save_local("faiss_index")

    # Return a standard retriever compatible with LangChain (reduced k for token limits)
    return vectordb.as_retriever(search_kwargs={"k": 8})

def load_existing_indexes():
    """Load existing FAISS indexes if they exist."""
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/LaBSE",
            model_kwargs={'device': 'cpu'},  
            encode_kwargs={'normalize_embeddings': True, 'batch_size': 32}
        )
        
        # Try to load priority index
        priority_vectordb = None
        if os.path.exists("faiss_index_priority"):
            print("Loading existing priority index...")
            priority_vectordb = FAISS.load_local("faiss_index_priority", embeddings, allow_dangerous_deserialization=True)
        
        # Try to load secondary index
        secondary_vectordb = None
        if os.path.exists("faiss_index_secondary"):
            print("Loading existing secondary index...")
            secondary_vectordb = FAISS.load_local("faiss_index_secondary", embeddings, allow_dangerous_deserialization=True)
        
        # Try to load simple index
        simple_vectordb = None
        if os.path.exists("faiss_index"):
            print("Loading existing simple index...")
            simple_vectordb = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        
        return priority_vectordb, secondary_vectordb, simple_vectordb
    except Exception as e:
        print(f"Error loading existing indexes: {e}")
        return None, None, None

def inspect_faiss_index(index_path="faiss_index", show_content_preview=False):
    """
    Inspect the contents of a FAISS index to see which sources are included.
    
    Args:
        index_path (str): Path to the FAISS index directory
        show_content_preview (bool): Whether to show a preview of document content
    """
    try:
        if not os.path.exists(index_path):
            print(f"❌ Index directory '{index_path}' not found!")
            return
        
        # Load the embeddings model (required for loading FAISS)
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/LaBSE",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True, 'batch_size': 32}
        )
        
        print(f"🔍 Inspecting FAISS index: {index_path}")
        print("=" * 50)
        
        # Load the FAISS index
        vectordb = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
        
        # Get the document store (contains metadata and content)
        docstore = vectordb.docstore
        
        # Get all document IDs
        doc_ids = list(docstore._dict.keys()) if hasattr(docstore, '_dict') else []
        
        print(f"📊 Total chunks in index: {len(doc_ids)}")
        print(f"📊 Vector dimension: {vectordb.index.d}")
        print(f"📊 Number of vectors: {vectordb.index.ntotal}")
        
        # Collect source information
        sources = {}
        for doc_id in doc_ids:
            doc = docstore.search(doc_id)
            if doc and hasattr(doc, 'metadata') and 'source' in doc.metadata:
                source = doc.metadata['source']
                if source not in sources:
                    sources[source] = []
                sources[source].append({
                    'id': doc_id,
                    'content_length': len(doc.page_content) if hasattr(doc, 'page_content') else 0,
                    'content_preview': doc.page_content[:200] + "..." if hasattr(doc, 'page_content') and show_content_preview else None
                })
        
        print(f"\n📁 Sources found: {len(sources)}")
        print("-" * 30)
        
        # Display sources and their chunk counts
        for source, chunks in sources.items():
            print(f"📄 {source}")
            print(f"   └── Chunks: {len(chunks)}")
            
            if show_content_preview and chunks:
                # Show preview of first chunk
                first_chunk = chunks[0]
                if first_chunk['content_preview']:
                    print(f"   └── Preview: {first_chunk['content_preview']}")
            print()
        
        print("📋 SUMMARY:")
        print(f"   Total chunks: {len(doc_ids)}")
        
        return sources
        
    except Exception as e:
        print(f"❌ Error inspecting index: {e}")
        return None

def compare_indexes():
    """Compare all available FAISS indexes."""
    print("🔍 COMPARING ALL FAISS INDEXES")
    print("=" * 60)
    
    indexes_to_check = [
        ("faiss_index", "Simple Combined Index"),
        ("faiss_index_priority", "Priority Documents Index"),
        ("faiss_index_secondary", "Secondary Documents Index")
    ]
    
    for index_path, description in indexes_to_check:
        print(f"\n📁 {description}")
        print("-" * 40)
        sources = inspect_faiss_index(index_path, show_content_preview=False)
        if not sources:
            print(f"   ❌ Index not found or empty\n")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "inspect":
        # Run inspection mode
        if len(sys.argv) > 2:
            # Inspect specific index
            index_path = sys.argv[2]
            inspect_faiss_index(index_path, show_content_preview=True)
        else:
            # Compare all indexes
            compare_indexes()
    else:
        # Run normal main function
        main()
