from pydantic_settings import BaseSettings
from .constants import MAX_FILE_SIZE, MAX_TOTAL_SIZE, ALLOWED_TYPES

class Settings(BaseSettings):
    # Required settings (must be provided in .env or environment variables)
    OPENAI_API_KEY: str  # API key for OpenAI (used for embeddings and LLMs)

    # Optional settings with defaults
    MAX_FILE_SIZE: int = MAX_FILE_SIZE  # Maximum size for a single file (50 MB)
    MAX_TOTAL_SIZE: int = MAX_TOTAL_SIZE  # Maximum total size for all uploaded files (200 MB)
    ALLOWED_TYPES: list = ALLOWED_TYPES  # Allowed file types for upload

    # Database settings (for ChromaDB)
    CHROMA_DB_PATH: str = "./chroma_db"  # Directory to persist ChromaDB data
    CHROMA_COLLECTION_NAME: str = "documents"  # Name of the collection in ChromaDB

    # Retrieval settings
    VECTOR_SEARCH_K: int = 5  # Number of documents to retrieve in vector search
    HYBRID_RETRIEVER_WEIGHTS: list = [0.4, 0.6]  # Weights for BM25 and vector retrievers

    # Logging settings
    LOG_LEVEL: str = "INFO"  # Default logging level

    class Config:
        env_file = ".env"  # Load environment variables from .env file
        env_file_encoding = "utf-8"  # Specify encoding for .env file

# Create an instance of Settings
settings = Settings()