# Bridging Legal Knowledge and AI: Retrieval-Augmented Generation with Vector Stores, Knowledge Graphs, and Hierarchical Non-negative Matrix Factorization

**Authors:** Ryan C. Barron, Maksim E. Eren, Olga M. Serafimova, Cynthia Matuszek, Boian S. Alexandrov

**Date:** May 9, 2025 (v2)

**arXiv ID:** 2502.20364v2

**Links:** [arXiv](https://arxiv.org/abs/2502.20364v2) | [PDF](https://arxiv.org/pdf/2502.20364v2)

**Venue:** ICAIL 2025 (20th International Conference on Artificial Intelligence and Law)

---

## Summary

This paper presents a jurisdiction-specific legal AI system that combines Retrieval-Augmented Generation (RAG), Vector Stores (VS), Knowledge Graphs (KG), and Hierarchical Non-Negative Matrix Factorization (HNMFk) to improve legal information retrieval and reduce LLM hallucinations. The system was built and tested on New Mexico's legal corpus: 265 constitutional provisions, 28,251 statutory sections, 5,727 Supreme Court cases, and 10,072 Court of Appeals cases, all scraped from Justia.

The core innovation is using HNMFk (via the T-ELF library) to automatically discover latent topic clusters within legal documents and then integrating those topics into a Neo4j knowledge graph alongside citation links and metadata. When a user asks a legal question, the system performs both semantic vector search and knowledge graph traversal, then feeds the combined results to an LLM for grounded, citation-backed answers. In evaluations against GPT-4o, Claude 3 Opus, Gemini Pro, and Nemotron-70B, the system provided more accurate, reproducible, and citation-specific answers -- particularly for quantitative queries (e.g., counting cases mentioning "habeas corpus") and citation pattern queries where general-purpose LLMs either refused to answer or hallucinated fake case names.

---

## Key Findings

1. **96.5% retrieval improvement with topic-based indexing**: Using HNMFk-derived topics to segment vector stores dramatically improved Mean Reciprocal Rank (MRR) for case law retrieval -- from 0.19 to 0.59 for Court of Appeals cases and from 0.29 to 0.62 for Supreme Court cases -- compared to embedding the entire corpus as a single collection.

2. **General-purpose LLMs fail on jurisdiction-specific legal questions**: When asked "How many NM Supreme Court cases mention 'habeas corpus'?", GPT-4o and Claude 3 Opus refused to answer, GPT-3.5 hallucinated "approximately 72," and Nemotron estimated 127-170. The authors' system returned the verifiable answer of 215 cases directly from its knowledge graph.

3. **Citation hallucination is a major problem**: GPT-3.5 fabricated case citations like "Smith v. Jones, 123 N.M. 456 (2018)" that don't exist. The authors' system returned exact citation counts (e.g., "NMSA 41-5-1 cited in 36 malpractice cases") because it queries an actual citation graph rather than generating from memory.

4. **Hierarchical topic decomposition reveals legal structure**: NMFk automatically determined the optimal number of latent topics at each level -- 10 for constitutional provisions, 12 for statutes (depth-0), 27 for Court of Appeals cases, and 24 for Supreme Court cases -- producing interpretable clusters like "Fourth Amendment Protections Against Unlawful Searches" and "Workers' Compensation Process."

5. **Chunking helps unstructured text but hurts structured text**: Splitting case law into 300-character chunks with 500-word overlap improved retrieval, but applying the same chunking to compact statutory sections fragmented legal concepts and slightly reduced accuracy. The best strategy was topic-specific indexing with selective chunking.

6. **Knowledge graph scale**: The final Neo4j graph contained 190,283 nodes and 16,927,700 edges, with 289,102 legal citation links extracted using GPT-3.5-turbo as a named entity extractor.

---

## Technical Background You Need to Know

### Retrieval-Augmented Generation (RAG)
A technique where an LLM doesn't rely solely on its training data to answer questions. Instead, it first *retrieves* relevant documents from an external database, then *generates* an answer grounded in those documents. This reduces hallucinations because the model cites real sources rather than guessing. Think of it like an open-book exam versus a closed-book exam.

### Vector Store (VS) / Vector Database
A database that stores text as high-dimensional numerical vectors (embeddings) rather than raw strings. When you search, your query is also converted to a vector, and the database finds the most semantically similar documents using distance metrics (cosine similarity). This means "due process violations" can match documents about "constitutional rights infringements" even without shared keywords. The paper uses Milvus as its vector database and OpenAI's text-embedding-ada-002 for generating embeddings.

### Knowledge Graph (KG)
A structured database where information is stored as nodes (entities) connected by edges (relationships), forming triplets like (Case A) --[cites]--> (Statute B). Unlike vector stores that find *similar* documents, knowledge graphs can traverse *explicit relationships* -- e.g., "find all cases that cite this statute and were decided after 2010." The paper uses Neo4j, the most widely-used graph database.

### Non-Negative Matrix Factorization (NMF)
A dimensionality reduction technique that decomposes a large matrix into two smaller matrices where all values are non-negative (zero or positive). For text, you start with a TF-IDF matrix (documents x terms) and decompose it into: (1) a topics x terms matrix (what words define each topic) and (2) a documents x topics matrix (which topics each document belongs to). The non-negativity constraint makes results interpretable -- each topic is an additive combination of words, not a mix of positive and negative weights.

### TF-IDF (Term Frequency-Inverse Document Frequency)
A numerical statistic that reflects how important a word is to a document within a corpus. Words that appear frequently in one document but rarely across all documents get high scores. "Estoppel" appearing in 50 of 10,000 cases would score high; "the" appearing everywhere scores near zero. It's the input matrix that NMF decomposes.

### Hierarchical NMF with Automatic Model Selection (HNMFk)
An extension of NMF that: (1) automatically determines the optimal number of topics *k* using bootstrap resampling and silhouette scores (rather than requiring you to guess), and (2) applies NMF recursively -- first finding broad topics, then decomposing each into subtopics, creating a tree structure. The paper uses the T-ELF library (Tensor Extraction of Latent Features) developed at Los Alamos National Laboratory. Maximum decomposition depth was set to 2, with a minimum of 100 documents per cluster to continue decomposing.

### Silhouette Score
A metric measuring how well-separated clusters are. Ranges from -1 to 1: high values mean data points are well-matched to their own cluster and poorly matched to neighboring clusters. Used here to determine the optimal number of topics at each level of the hierarchy.

### Neo4j and Cypher
Neo4j is a graph database that stores data as nodes and relationships natively (not in tables). Cypher is its query language, similar to how SQL is for relational databases. Example: `MATCH (c:Case)-[:CITES]->(s:Statute) WHERE s.id = 'NMSA 41-5-1' RETURN c` finds all cases citing a specific statute.

### ROUGE-L
A metric for evaluating text summaries by measuring the longest common subsequence (LCS) between generated and reference text. Higher ROUGE-L means the generated text preserves more of the reference's sentence structure. Used here to evaluate AI-generated legal answers against expert reference answers.

### NLI (Natural Language Inference) Entailment
A classification task where a model determines if one text (hypothesis) logically follows from another (premise). Labels: entailment (follows), contradiction (conflicts), or neutral (unrelated). Used here to check if AI-generated answers are logically supported by reference legal texts.

### FactCC and SummaC
Evaluation metrics for factual consistency. FactCC fine-tunes a model on labeled correct/incorrect summaries to detect factual errors. SummaC aggregates entailment scores across sentence pairs. Both check whether generated text stays faithful to source documents -- critical in legal contexts where a fabricated citation could lead to sanctions.

### LEGAL-BERT
A version of BERT pre-trained on legal text (court opinions, legislation, contracts) rather than general web text. It better understands legal language nuances like "estoppel," "res judicata," and "habeas corpus." Referenced as a baseline for domain-specific embeddings.

### LexGLUE
A benchmark dataset for legal NLP with 7 tasks spanning contracts, court opinions, and legislation. Referenced as one of the standard evaluation frameworks for legal AI systems.

### Justia
A free online legal information platform providing access to case law, statutes, regulations, and legal encyclopedias. The authors scraped New Mexico's legal documents from Justia in compliance with its terms of service and robots.txt.

---

## Python Code Examples

### Example 1: Building a TF-IDF Matrix from Legal Documents (Core NMF Input)

```python
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# Simulated legal document corpus
legal_docs = [
    "The court finds the defendant liable for breach of contract under NMSA 1978 Section 56-8-1",
    "Habeas corpus petition denied based on procedural default doctrine",
    "Workers compensation claim for injury sustained during employment approved",
    "Fourth amendment protection against unreasonable search and seizure applies",
    "Due process requires adequate notice before deprivation of property rights",
    "Estoppel prevents the party from contradicting previous sworn testimony",
]

# Build TF-IDF matrix with parameters similar to the paper
# min_df=2 means term must appear in at least 2 docs
# max_df=0.8 means ignore terms appearing in >80% of docs
vectorizer = TfidfVectorizer(
    min_df=2,           # Paper used 5-50 depending on corpus
    max_df=0.8,         # Paper used 0.7-0.8 depending on corpus
    stop_words="english",
    token_pattern=r"(?u)\b[a-zA-Z]{3,}\b"  # Only words with 3+ chars
)

# X is the TF-IDF matrix: rows=documents, columns=terms
# This is what gets decomposed by NMF
X = vectorizer.fit_transform(legal_docs)
feature_names = vectorizer.get_feature_names_out()

print(f"TF-IDF Matrix shape: {X.shape}")
print(f"  {X.shape[0]} documents x {X.shape[1]} unique terms")
print(f"\nVocabulary sample: {list(feature_names[:10])}")
print(f"\nDocument 0 top terms:")
row = X[0].toarray().flatten()
top_idx = row.argsort()[-5:][::-1]
for idx in top_idx:
    if row[idx] > 0:
        print(f"  {feature_names[idx]}: {row[idx]:.3f}")
```

### Example 2: Applying NMF for Legal Topic Discovery

```python
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# Larger simulated corpus (in practice, the paper used 28,251 statutes)
legal_docs = [
    "water rights irrigation district allocation stream diversion",
    "water appropriation beneficial use prior rights adjudication",
    "criminal sentencing felony enhancement habitual offender",
    "criminal penalty incarceration probation parole revocation",
    "property tax assessment valuation county exemption revenue",
    "gross receipts tax business deduction exemption collection",
    "child custody parental rights termination welfare abuse",
    "child protective services foster care placement hearing",
]

vectorizer = TfidfVectorizer(stop_words="english")
X = vectorizer.fit_transform(legal_docs)
terms = vectorizer.get_feature_names_out()

# NMF decomposes X into W (docs x topics) and H (topics x terms)
# n_components = number of topics (paper used NMFk to auto-select this)
n_topics = 4
nmf = NMF(n_components=n_topics, random_state=42, max_iter=500)
W = nmf.fit_transform(X)  # Document-topic matrix
H = nmf.components_        # Topic-term matrix

# Display top keywords per topic (paper used top 50; we show top 5)
print("Discovered Legal Topics:\n")
for topic_idx, topic in enumerate(H):
    top_term_indices = topic.argsort()[-5:][::-1]
    top_terms = [terms[i] for i in top_term_indices]
    print(f"  Topic {topic_idx}: {', '.join(top_terms)}")

# Show which topic each document belongs to
print("\nDocument-Topic Assignments:")
for doc_idx, doc_topics in enumerate(W):
    dominant_topic = doc_topics.argmax()
    print(f"  Doc {doc_idx} -> Topic {dominant_topic} (score: {doc_topics[dominant_topic]:.2f})")
```

### Example 3: Building a Legal Knowledge Graph with Neo4j (Conceptual)

```python
# Demonstrates the knowledge graph schema used in the paper
# Requires: pip install neo4j

from neo4j import GraphDatabase

# Connect to Neo4j (paper used Neo4j for their KG)
driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "password"))

def build_legal_kg(session):
    """Create legal document nodes and citation relationships."""

    # Create statute nodes (paper had 28,251 of these)
    session.run("""
        CREATE (s:Statute {
            id: 'NMSA-41-5-1',
            title: 'Medical Malpractice Act',
            chapter: 41,
            article: 5,
            section: 1
        })
    """)

    # Create case law nodes (paper had 15,799 total cases)
    session.run("""
        CREATE (c:SupremeCourtCase {
            id: 'CERVANTES-v-FORBIS-1964',
            name: 'Cervantes v. Forbis',
            year: 1964,
            citation: '59 N.M. 312'
        })
    """)

    # Create citation edges (paper extracted 289,102 legal citations)
    session.run("""
        MATCH (c:SupremeCourtCase {id: 'CERVANTES-v-FORBIS-1964'})
        MATCH (s:Statute {id: 'NMSA-41-5-1'})
        CREATE (c)-[:CITES]->(s)
    """)

    # Create NMFk topic nodes and link to documents
    session.run("""
        CREATE (t:NMFkTopic {
            id: 'topic-8',
            label: 'Fourth Amendment Protections Against Unlawful Searches',
            document_count: 370,
            depth: 0
        })
    """)

    # Link topic to cases that belong to it
    session.run("""
        MATCH (t:NMFkTopic {id: 'topic-8'})
        MATCH (c:SupremeCourtCase {id: 'CERVANTES-v-FORBIS-1964'})
        CREATE (c)-[:HAS_TOPIC]->(t)
    """)

    # Query: Find all cases citing a statute via the KG
    # This is how the system answers "common citations for malpractice cases"
    result = session.run("""
        MATCH (c:Case)-[:CITES]->(s:Statute {id: 'NMSA-41-5-1'})
        RETURN count(c) AS citation_count
    """)
    for record in result:
        print(f"Cases citing NMSA 41-5-1: {record['citation_count']}")

# Usage:
# with driver.session() as session:
#     build_legal_kg(session)
```

---

## Why This Paper Matters

This paper demonstrates a practical, working system that solves a real problem: general-purpose LLMs are unreliable for jurisdiction-specific legal research. They hallucinate case citations, refuse quantitative queries, and can't trace their reasoning to specific sources. By combining vector search (for semantic similarity) with a knowledge graph (for structured relationships and exact counts) and NMF-derived topics (for interpretable document clustering), the system provides answers that are both accurate and verifiable.

For legal practitioners and legal tech developers, this is significant because it shows a concrete architecture for building trustworthy legal AI at the state level. The system doesn't just find "similar" documents -- it can count exactly how many cases cite a specific statute, identify citation patterns across thousands of cases, and explain its reasoning through the knowledge graph's explicit links. The approach is reproducible for any jurisdiction with publicly accessible legal texts, making it a template for building state-specific or country-specific legal research tools.

---

## Methodology Note

- **Study type:** System design and empirical evaluation
- **Dataset:** New Mexico legal corpus -- 265 constitutional provisions, 28,251 statutory sections, 5,727 Supreme Court cases, 10,072 Court of Appeals cases (scraped from Justia)
- **Method:** Modular pipeline: web scraping -> TF-IDF -> HNMFk topic decomposition -> Neo4j knowledge graph construction -> Milvus vector store -> RAG with LLM agent. Evaluated with 25 ChatGPT-generated questions (verified by SME lawyer) and 60 expert-formulated questions.
- **Evaluation metrics:** ROUGE-L, NLI entailment, SummaC, FactCC, human evaluation, MRR, and top-10 hit rate
- **Models compared:** GPT-4o, GPT-3.5, Claude 3 Opus, Gemini Pro, Nemotron-70B-Instruct, and the authors' system
- **Key limitations:** Incomplete author attribution in scraped data; limited to statutes, constitution, and case law (does not include administrative codes, judicial rules, or NMRA); citation extraction relies on GPT-3.5-turbo which may miss or misattribute citations; evaluation questions are relatively small-scale (25 and 60); no comparison with commercial legal research tools (Westlaw, LexisNexis)

---

*Summary created: 2026-02-10 | Source: arXiv 2502.20364v2*
