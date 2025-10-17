"""Build a LlamaIndex vector index for LocalRAGRetriever.

This script reads files under ``data/`` and builds a persistent LlamaIndex
vector store to the directory configured via ``LLAMA_PERSIST_DIR`` (defaults to
``storage/llama_index``). It uses a Hugging Face embedding model so everything
can run locally without external paid services.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import get_settings


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for index construction."""
    parser = argparse.ArgumentParser(description="Build LlamaIndex (vector) for local RAG.")
    parser.add_argument("--data-dir", type=Path, help="Directory containing raw documents.")
    parser.add_argument(
        "--persist-dir",
        type=Path,
        help="Directory to persist the LlamaIndex (defaults to settings.llama_index_dir)",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        help="HF embedding model name (defaults to settings.llama_embedding_model)",
    )
    return parser.parse_args()


def main() -> None:
    """Build and persist a LlamaIndex vector store from local documents."""
    args = parse_args()
    settings = get_settings()

    data_dir = (args.data_dir or settings.data_dir).resolve()
    persist_dir = (args.persist_dir or settings.llama_index_dir).resolve()
    model_name = args.embedding_model or settings.llama_embedding_model

    if persist_dir.exists():
        import shutil

        shutil.rmtree(persist_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)

    # Lazy imports to keep base runtime light
    from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding

    print(f"Building LlamaIndex from {data_dir} with model {model_name} -> {persist_dir}")
    documents = SimpleDirectoryReader(str(data_dir)).load_data()
    if not documents:
        raise RuntimeError(f"No documents discovered in {data_dir}.")

    embed_model = HuggingFaceEmbedding(model_name=model_name)
    storage_context = StorageContext.from_defaults()
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        embed_model=embed_model,
    )
    index.storage_context.persist(persist_dir=str(persist_dir))

    print(f"Indexed {len(documents)} documents.")
    print(f"Index persisted to {persist_dir}")





if __name__ == "__main__":
    main()
