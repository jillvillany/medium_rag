## quickstart

conda create -n medium_rag python=3.10 -y
conda activate medium_rag
poetry install --no-root

pinecone: https://www.pinecone.io
- sign up with google
- create an index, name it, use cosine metric, choose a model for embeddings and that will set the dimensions