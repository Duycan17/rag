# Hypothesis generators for property-based testing
from hypothesis import strategies as st
from uuid import UUID


# UUID generator
uuids = st.uuids().map(str)

# Non-empty text generator
non_empty_text = st.text(min_size=1, max_size=1000).filter(lambda x: x.strip())

# Document content generator
document_content = st.text(min_size=10, max_size=10000)

# Chunk size generator (reasonable bounds)
chunk_sizes = st.integers(min_value=100, max_value=5000)

# Embedding dimension (768 for Google embeddings)
EMBEDDING_DIM = 768
