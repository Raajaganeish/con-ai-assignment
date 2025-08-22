Project Overview: Financial Document RAG Pipeline
=================================================

This project implements a **Retrieval-Augmented Generation (RAG)** pipeline designed for specialized question answering on financial documents, especially PayPal annual reports. The pipeline processes PDF documents, segments them into structured sections, creates retrievable text chunks, builds hybrid retrieval indexes (dense + sparse), and exposes the entire QA system via an interactive Gradio interface. The project emphasizes transparency (with provenance for every answer), modular retrieval logic, and safety via query guardrails.

* * * * *

1\. **PDF Segmentation: Heading-Aware Sectioning**
--------------------------------------------------

**File:** `segment.py`

### **Purpose**

-   **Segment** PDFs into logical sections and subsections, leveraging font-size hierarchy to detect headings (H1 and H2).

-   **Preserve** tables, clean text, and retain heading structure for downstream retrieval.

### **How it Works**

-   **Font Analysis:** Scans all pages, extracting text lines with their font sizes using PyMuPDF.

-   **Heading Detection:** Clusters font sizes (via KMeans or percentiles) to identify section (`H1`) and subsection (`H2`) headings.

-   **Sectionization:** Associates text and tables to each detected section, preserving heading/subheading metadata.

-   **Output:** Emits a JSON (`sections_{docid}.json`) and text (`sections_{docid}.txt`) file per document, where each section contains clean text, tables, and metadata (including pages and H2 subheads).

* * * * *

2\. **Chunking: Sliding Window Tokenization**
---------------------------------------------

**File:** `chunk_util.py` (and see the logic in `step2_chunk.py`)

### **Purpose**

-   **Divide** each section's text into overlapping, fixed-size "chunks" optimized for retrieval (e.g., 100/400 tokens, with overlaps).

### **How it Works**

-   **Loads** segmented sections from the previous step.

-   **Tokenizes** text using the `sentence-transformers/all-MiniLM-L6-v2` tokenizer.

-   **Sliding Window:** Generates overlapping text chunks (e.g., 100 tokens with 20-token overlap, 400 tokens with 40-token overlap).

-   **Metadata:** For each chunk, saves metadata: unique ID, originating doc/section, pages, subheads, and token offsets.

-   **Output:** JSONL files for each chunk size (`chunks_100.jsonl`, `chunks_400.jsonl`), plus a manifest summarizing chunk stats.

* * * * *

3\. **Indexing: Dense (FAISS) and Sparse (BM25) Index Construction**
--------------------------------------------------------------------

**File:** `embedding_indexing.py` (logic from `step2_index.py`)

### **Purpose**

-   **Build** both dense vector (semantic) and sparse (lexical/BM25) indexes for fast hybrid retrieval.

### **How it Works**

-   **Embeddings:** Uses SentenceTransformers to encode each chunk's text into a normalized vector.

-   **FAISS:** Builds an inner-product FAISS index for efficient dense retrieval.

-   **BM25:** Tokenizes all chunks and fits a BM25Okapi index for fast lexical retrieval.

-   **Output:** Saves indexes, ID mappings, and embedding config for each chunk size.

* * * * *

4\. **Hybrid Retrieval & Reranking**
------------------------------------

**File:** `retrieve.py`

### **Purpose**

-   **Hybrid Retrieval:** Combines dense (semantic) and sparse (lexical) retrieval to maximize recall and precision.

-   **Reranking:** Uses a cross-encoder for final re-ranking of candidates based on query-context similarity.

### **How it Works**

-   **Query Preprocessing:** Optionally expands queries with finance-specific synonyms and years to improve recall.

-   **Dense Search:** Uses the FAISS index for semantic retrieval.

-   **Sparse Search:** Uses the BM25 index for keyword retrieval.

-   **Weighted Fusion:** Scores from dense and sparse retrieval are fused (with tunable weights).

-   **Biases and Filtering:** Prefers table data for numeric queries, applies section regexes, and boosts candidates with matching years or financial terms.

-   **Reranking:** Top candidates are re-ranked by a cross-encoder for more accurate results.

-   **Output:** Returns a ranked list of chunks, each with full provenance (ID, section, page, etc.).

* * * * *

5\. **Interactive QA Interface (RAG & Fine-Tuned Baselines)**
-------------------------------------------------------------

**File:** `interface.py`

### **Purpose**

-   **Expose** the pipeline via a Gradio web interface, allowing users to pose financial questions and see how answers are generated.

### **How it Works**

-   **Input:** User submits a query (with options for RAG or fine-tuned mode, chunk size, top-k results).

-   **Guardrails:** Queries are filtered for empty, irrelevant, or unsafe content.

-   **RAG Pipeline:** Uses hybrid retrieval to assemble the most relevant context, then runs a seq2seq model to generate the answer, reporting confidence and provenance.

-   **Fine-Tuned Baseline:** Optionally runs a direct QA model (without retrieval) for comparison.

-   **Display:** Shows answer, retrieval/generation confidence, provenance (which chunks contributed), and the model prompt for transparency.

* * * * *

6\. **HybridReranker Core (Retrieve Orchestration)**
----------------------------------------------------

**File:** `retrieve.py`

### **Purpose**

-   Provides a single interface for hybrid retrieval with all tunable parameters and reranking logic.

### **Key Capabilities**

-   **Query expansion and cleaning**

-   **Multi-stage candidate filtering and boosting**

-   **Numeric-intent detection (e.g., for financial metrics)**

-   **Cross-encoder reranking for final top-K selection**

-   **Support for hard filtering/boosting via regex**

* * * * *

7\. **Implementation Highlights**
---------------------------------

-   **Open-source, local pipeline:** No reliance on external APIs---entirely self-hosted.

-   **Metadata preservation:** Every chunk and retrieval result is tied back to its originating document, section, and page range.

-   **Table awareness:** Tables are specially preserved and can be preferred for numeric queries.

-   **Transparency:** Every answer includes provenance and model prompt for debugging and auditability.

-   **Extensibility:** Modular pipeline (segmentation → chunking → indexing → retrieval → QA), each part replaceable or tunable.

* * * * *

8\. **Example End-to-End Flow**
-------------------------------

1.  **Upload PDF** →

2.  **Run segmentation** (`segment.py`) →

3.  **Create chunks** (`chunk_util.py`/`step2_chunk.py`) →

4.  **Build indexes** (`embedding_indexing.py`/`step2_index.py`) →

5.  **Launch UI** (`interface.py`) →

6.  **Ask a question**:

    -   System validates the query.

    -   Hybrid retrieval assembles context.

    -   Seq2Seq model generates answer.

    -   All results shown with confidence, provenance, and debug info.

* * * * *

References to Key Files
=======================

-   **`segment.py`:** Heading detection and PDF segmentation logic.

-   **`chunk_util.py`:** Tokenization and chunking utilities (see also `step2_chunk.py`).

-   **`embedding_indexing.py`:** Embedding, FAISS, and BM25 index construction (see also `step2_index.py`).

-   **`retrieve.py`:** Hybrid retrieval, cross-encoder reranking, query expansion, and candidate filtering.

-   **`interface.py`:** Gradio UI, guardrails, QA pipelines, and user interaction.