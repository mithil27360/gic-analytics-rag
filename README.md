# GIC Insurance Analytics RAG

## Overview

This project demonstrates a Retrieval-Augmented Generation (RAG) pipeline for analyzing publicly available general insurance premium data in India.

It covers premium statistics for FY24 and FY25 (April–October periods), sourced from open publications by the Insurance Regulatory and Development Authority of India (**IRDAI**).

The system transforms structured data into semantic documents, stores them in a local vector database, and enables natural-language querying with responses grounded in the retrieved content.

**Main goals:**
- Showcase reproducible data processing and analytics
- Enable traceable, source-backed question answering
- Demonstrate practical use of RAG with open insurance statistics

**Important:** All data used is publicly released by IRDAI (monthly business figures / flash reports). No proprietary, confidential, or internal company data is included.

## Key Features

- Question-answering interface powered by RAG
- Pre-computed analytics (growth rates, segment comparisons, rankings)
- Semantic document generation for better retrieval quality
- Persistent local vector store (ChromaDB)
- Simple Streamlit chat-based UI
- Responses include references to generated document IDs

## Data Source

- Publicly available IRDAI monthly / cumulative premium data (non-life / general insurers)
- Time period: FY24 & FY25 (up to October)
- Scope: ~34 general insurers, major segments (Health, Motor, Fire, Miscellaneous incl. Crop, etc.)
- Source: IRDAI website – Gross Direct Premium underwritten flash figures
- All files are cleaned & aggregated versions of these public releases

**No private or restricted data is used at any stage.**

## Project Structure

```text
├── app.py                      # Streamlit application (chat interface)
├── requirements.txt            # Python dependencies
├── .env                        # API keys (not committed to git)
│
├── Rag/                        # Core logic
│   ├── __init__.py
│   ├── analytics.py           # Metric calculations (growth, volatility, rankings)
│   ├── document_generator.py  # Creates semantic text documents from analytics
│   └── rag.py                 # Retrieval + generation logic
│
├── data/
│   ├── raw/                   # Original downloaded files (Excel/CSV)
│   ├── cleaned/               # Preprocessed monthly data
│   └── processed/             # Final master file + generated RAG documents
│
├── analysis/
│   └── gic_ytd_premium_analysis_report.ipynb   # Exploratory / validation notebook
│
└── chroma_db/                 # Persistent vector database (gitignored / auto-created)


## Setup Instructions

Follow these steps in sequence:

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Groq API key (recommended)

Create a file named `.env` in the project root directory.

Add the following line (replace with your actual key):

```
GROQ_API_KEY=your_key_here
```

Get a free API key from: [https://console.groq.com/keys](https://console.groq.com/keys)

**Note:** Without a key, the app will still run but will use simpler fallback/template-based responses instead of full LLM generation.

### 3. Build the knowledge base (run once)

```bash
cd Rag
python document_generator.py
```

This step:
- Generates ~49 semantic/analytical text documents from processed data
- Embeds them using the sentence-transformers model
- Stores them in the local ChromaDB vector database
- Takes ~10–30 seconds depending on your machine

### 4. Launch the application

```bash
streamlit run app.py
```

The app will automatically open in your browser at:

```
http://localhost:8501
```

You can now start asking questions about the GIC (general insurance) premium data.

---

## Notes

- First run requires building the vector database
- Re-run `document_generator.py` only if data changes
- Works without Groq key but with limited response quality

