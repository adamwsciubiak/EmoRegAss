"""
Configuration parameters for the RAG Ingestion Pipeline.

This module centralizes the hyperparameters used for chunking and processing 
therapeutic documents into the static knowledge base (Actions Catalog - DB2).
In the architecture proposed by Pico et al. (2024), the Dynamic Planner relies 
on this static knowledge to translate abstract emotion regulation strategies 
(e.g., 'Distraction') into concrete, step-by-step tactical plans.

Tuning these parameters directly affects the granularity and semantic 
quality of the tactics retrieved by the system.
"""

import os

# --- Chunking Parameters ---
# CHUNK_SIZE: The maximum number of characters per text chunk.
# Why 1000? It represents a sweet spot. Chunks that are too small lack context 
# (the LLM won't understand the therapeutic step), while chunks that are too large 
# dilute the semantic density of the vector embedding and consume too many tokens.
CHUNK_SIZE: int = 1000

# CHUNK_OVERLAP: The number of characters that overlap between consecutive chunks.
# Why 100? Overlap acts as a "semantic glue". It ensures that if a therapeutic 
# instruction is split across two chunks, the context at the boundary is not lost.
CHUNK_OVERLAP: int = 100

# SEPARATORS: The sequence of string separators used to split the text.
# The algorithm tries to split at the first separator ("\n\n" - paragraphs). 
# If a paragraph is still larger than CHUNK_SIZE, it falls back to the next 
# separator ("\n" - lines), then spaces (" "), and finally individual characters.
# This hierarchy preserves the natural grammatical structure of therapeutic manuals.
SEPARATORS: list[str] = ["\n\n", "\n", " ", ""]

# --- Pipeline Paths ---
# DEFAULT_SOURCE_DIR: The default local directory where the ingestion pipeline 
# looks for raw therapeutic manuals (PDFs, TXTs). It resolves to a folder 
# named "documents_for_RAG" located in the same directory as this config file.
DEFAULT_SOURCE_DIR: str = os.path.join(os.path.dirname(__file__), "documents_for_RAG")

# PROCESSED_FILES_LOG_PATH: Path to the log file that acts as a registry.
# It tracks which documents have already been uploaded. This ensures idempotency:
# running the pipeline multiple times won't duplicate embeddings in the database.
PROCESSED_FILES_LOG_PATH: str = os.path.join(os.path.dirname(__file__), "processed_files.log")