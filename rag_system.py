import os
from dotenv import load_dotenv
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex, Document, ServiceContext
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding
import google.generativeai as genai
import logging
import PyPDF2

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configure Google API Key
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
if not GOOGLE_API_KEY:
    raise ValueError("Please set GOOGLE_API_KEY in your .env file")

genai.configure(api_key=GOOGLE_API_KEY)

# Configure Gemini model and embeddings
llm = Gemini(model="models/gemini-pro", 
             max_tokens=2048,
             temperature=0.5)
embed_model = GeminiEmbedding()
service_context = ServiceContext.from_defaults(
    llm=llm,
    embed_model=embed_model,
    chunk_size=1024,
    chunk_overlap=20
)
Settings.llm = llm
Settings.embed_model = embed_model
Settings.service_context = service_context

def clean_text(text):
    """Clean text by removing problematic characters."""
    if not text:
        return ""
    # Remove emojis and other problematic unicode characters
    text = text.encode('ascii', 'ignore').decode('ascii')
    # Remove multiple spaces and newlines
    text = ' '.join(text.split())
    return text.strip()

def read_pdf(file_path):
    """Read PDF file and return its text content."""
    try:
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page_num, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text()
                if page_text:
                    cleaned_text = clean_text(page_text)
                    text += f"Page {page_num + 1}:\n{cleaned_text}\n\n"
            
            # Log the first 500 characters of the extracted text for debugging
            logger.info(f"Extracted text sample from {os.path.basename(file_path)} (first 500 chars):")
            logger.info(text[:500])
            
            if not text.strip():
                logger.warning(f"No text content extracted from {file_path}")
            else:
                logger.info(f"Successfully extracted {len(text)} characters from {file_path}")
            
            return text
    except Exception as e:
        logger.error(f"Error reading PDF {file_path}: {str(e)}")
        return ""

def build_knowledge_base(directory_path='data/my_knowledge_base/'):
    """Build the vector store index from documents in the specified directory."""
    try:
        logger.info(f"Loading documents from {directory_path}")
        documents = []
        
        # Manually process each file
        for filename in os.listdir(directory_path):
            if filename.lower().endswith('.pdf'):
                file_path = os.path.join(directory_path, filename)
                logger.info(f"Processing {filename}")
                
                # Read PDF content
                text = read_pdf(file_path)
                if text:
                    # Create Document object with metadata
                    doc = Document(
                        text=text,
                        metadata={
                            "file_name": filename,
                            "file_path": file_path,
                            "type": "pdf"
                        }
                    )
                    documents.append(doc)
                    logger.info(f"Successfully processed {filename} ({len(text)} characters)")
                else:
                    logger.warning(f"No text extracted from {filename}")
        
        logger.info(f"Loaded {len(documents)} documents")
        
        if not documents:
            logger.error("No valid documents found in the directory")
            return None
        
        # Create vector store index with custom service context
        logger.info("Creating vector store index...")
        index = VectorStoreIndex.from_documents(
            documents,
            service_context=service_context,
            show_progress=True
        )
        logger.info("Index created successfully")
        return index
    except Exception as e:
        logger.error(f"Error building knowledge base: {str(e)}")
        return None

def setup_query_engine(index):
    """Set up the query engine with the specified parameters."""
    if index:
        return index.as_query_engine(
            service_context=service_context,
            similarity_top_k=3,
            response_mode="compact",
            verbose=True
        )
    return None

def rag(query, query_engine):
    """Process a query using the RAG system."""
    try:
        logger.info(f"Processing query: {query}")
        
        # Log that we're about to query
        logger.info("Querying the index...")
        
        # Get response
        response = query_engine.query(query)
        
        # Log the response for debugging
        logger.info(f"Raw response: {str(response)}")
        
        # If response is empty or None, return a helpful message
        if not response or str(response).strip() == "":
            return "I apologize, but I couldn't find relevant information to answer your question. Please try rephrasing your question or ask something else about the Java documents."
        
        return str(response)
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        return f"I encountered an error while processing your query: {str(e)}"

def main():
    print("Initializing RAG system...")
    
    # Create data directory if it doesn't exist
    os.makedirs('data/my_knowledge_base/', exist_ok=True)
    
    print("Building knowledge base...")
    index = build_knowledge_base()
    
    if index:
        query_engine = setup_query_engine(index)
        print("\nRAG system is ready! You can start asking questions about your documents.")
        print("Type 'quit' to exit.")
        
        while True:
            try:
                query = input("\nEnter your query: ").strip()
                if not query:
                    print("Please enter a valid query.")
                    continue
                    
                if query.lower() == 'quit':
                    break
                    
                print("\nGenerating response...")
                response = rag(query, query_engine)
                print("\nResponse:", response)
                
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                logger.error(f"Error in main loop: {str(e)}")
                print(f"\nAn error occurred: {str(e)}")
                print("Please try again or type 'quit' to exit.")
    else:
        print("Failed to build knowledge base. Please check your data directory and files.")

if __name__ == "__main__":
    main()
