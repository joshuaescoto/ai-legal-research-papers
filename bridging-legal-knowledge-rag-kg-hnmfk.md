# Bridging Legal Knowledge and AI: Retrieval-Augmented Generation with Vector Stores, Knowledge Graphs, and Hierarchical Non-negative Matrix Factorization

**Authors:** Ryan C. Barron, Maksim E. Eren, Olga M. Serafimova, Cynthia Matuszek, Boian S. Alexandrov

**Date:** May 9, 2025 (v2)

**Links:** [arXiv](https://arxiv.org/abs/2502.20364v2) | [PDF](https://arxiv.org/pdf/2502.20364v2)

The paper presents a jurisdiction-specific legal AI system that combines Retrieval-Augmented Generation (RAG), Vector Stores (VS), Knowledge Graphs (KG), and Hierarchical Non-Negative Matrix Factorization (HNMFk) to improve legal information retrieval and reduce LLM hallucinations. The system was built and tested on New Mexico's legal corpus: 265 constitutional provisions, 28,251 statutory sections, 5,727 Supreme Court cases, and 10,072 Court of Appeals cases, all scraped from Justia.

The core innovation is using HNMFk (via the T-ELF library) to automatically discover latent topic clusters within legal documents and then integrating those topics into a Neo4j knowledge graph alongside citation links and metadata. When a user asks a legal question, the system performs both semantic vector search and knowledge graph traversal, then feeds the combined results to an LLM for grounded, citation-backed answers. In evaluations against GPT-4o, Claude 3 Opus, Gemini Pro, and Nemotron-70B, the system provided more accurate, reproducible, and citation-specific answers -- particularly for quantitative queries (e.g., counting cases mentioning "habeas corpus") and citation pattern queries where general-purpose LLMs either refused to answer or hallucinated fake case names.

**96.5% retrieval improvement with topic-based indexing**: Using HNMFk-derived topics to segment vector stores dramatically improved Mean Reciprocal Rank (MRR) for case law retrieval -- from 0.19 to 0.59 for Court of Appeals cases and from 0.29 to 0.62 for Supreme Court cases -- compared to embedding the entire corpus as a single collection.

**General-purpose LLMs fail on jurisdiction-specific legal questions**: When asked "How many NM Supreme Court cases mention 'habeas corpus'?", GPT-4o and Claude 3 Opus refused to answer, GPT-3.5 hallucinated "approximately 72," and Nemotron estimated 127-170. The authors' system returned the verifiable answer of 215 cases directly from its knowledge graph.

**Citation hallucination is a major problem**: GPT-3.5 fabricated case citations like "Smith v. Jones, 123 N.M. 456 (2018)" that don't exist. The authors' system returned exact citation counts (e.g., "NMSA 41-5-1 cited in 36 malpractice cases") because it queries an actual citation graph rather than generating from memory.

**Hierarchical topic decomposition reveals legal structure**: NMFk automatically determined the optimal number of latent topics at each level -- 10 for constitutional provisions, 12 for statutes (depth-0), 27 for Court of Appeals cases, and 24 for Supreme Court cases -- producing interpretable clusters like "Fourth Amendment Protections Against Unlawful Searches" and "Workers' Compensation Process."

**Chunking helps unstructured text but hurts structured text**: Splitting case law into 300-character chunks with 500-word overlap improved retrieval, but applying the same chunking to compact statutory sections fragmented legal concepts and slightly reduced accuracy. The best strategy was topic-specific indexing with selective chunking.

**Knowledge graph scale**: The final Neo4j graph contained 190,283 nodes and 16,927,700 edges, with 289,102 legal citation links extracted using GPT-3.5-turbo as a named entity extractor.

**Retrieval-Augmented Generation (RAG)** is a technique where an LLM doesn't rely solely on its training data to answer questions. Instead, it first *retrieves* relevant documents from an external database, then *generates* an answer grounded in those documents. This reduces hallucinations because the model cites real sources rather than guessing. Think of it like an open-book exam versus a closed-book exam.

**Vector Store (VS) / Vector Database** is a database that stores text as high-dimensional numerical vectors (embeddings) rather than raw strings. When you search, your query is also converted to a vector, and the database finds the most semantically similar documents using distance metrics (cosine similarity). This means "due process violations" can match documents about "constitutional rights infringements" even without shared keywords. The paper uses Milvus as its vector database and OpenAI's text-embedding-ada-002 for generating embeddings.

**Knowledge Graph (KG)** is a structured database where information is stored as nodes (entities) connected by edges (relationships), forming triplets like (Case A) --[cites]--> (Statute B). Unlike vector stores that find *similar* documents, knowledge graphs can traverse *explicit relationships* -- e.g., "find all cases that cite this statute and were decided after 2010." The paper uses Neo4j, the most widely-used graph database.

**Non-Negative Matrix Factorization (NMF)** is a dimensionality reduction technique that decomposes a large matrix into two smaller matrices where all values are non-negative (zero or positive). For text, you start with a TF-IDF matrix (documents x terms) and decompose it into: (1) a topics x terms matrix (what words define each topic) and (2) a documents x topics matrix (which topics each document belongs to). The non-negativity constraint makes results interpretable -- each topic is an additive combination of words, not a mix of positive and negative weights.

**TF-IDF (Term Frequency-Inverse Document Frequency)** is a numerical statistic that reflects how important a word is to a document within a corpus. Words that appear frequently in one document but rarely across all documents get high scores. "Estoppel" appearing in 50 of 10,000 cases would score high; "the" appearing everywhere scores near zero. It's the input matrix that NMF decomposes.

**Hierarchical NMF with Automatic Model Selection (HNMFk)** is an extension of NMF that: (1) automatically determines the optimal number of topics *k* using bootstrap resampling and silhouette scores (rather than requiring you to guess), and (2) applies NMF recursively -- first finding broad topics, then decomposing each into subtopics, creating a tree structure. The paper uses the T-ELF library (Tensor Extraction of Latent Features) developed at Los Alamos National Laboratory. Maximum decomposition depth was set to 2, with a minimum of 100 documents per cluster to continue decomposing.

**Silhouette Score** is a metric measuring how well-separated clusters are. Ranges from -1 to 1: high values mean data points are well-matched to their own cluster and poorly matched to neighboring clusters. Used here to determine the optimal number of topics at each level of the hierarchy.

**Neo4j** is a graph database that stores data as nodes and relationships natively (not in tables). **Cypher** is its query language, similar to how SQL is for relational databases. Example: `MATCH (c:Case)-[:CITES]->(s:Statute) WHERE s.id = 'NMSA 41-5-1' RETURN c` finds all cases citing a specific statute.

**ROUGE-L** is a metric for evaluating text summaries by measuring the longest common subsequence (LCS) between generated and reference text. Higher ROUGE-L means the generated text preserves more of the reference's sentence structure. Used here to evaluate AI-generated legal answers against expert reference answers.

**NLI (Natural Language Inference) Entailment** is a classification task where a model determines if one text (hypothesis) logically follows from another (premise). Labels: entailment (follows), contradiction (conflicts), or neutral (unrelated). Used here to check if AI-generated answers are logically supported by reference legal texts.

**FactCC and SummaC** is an evaluation metrics for factual consistency. FactCC fine-tunes a model on labeled correct/incorrect summaries to detect factual errors. SummaC aggregates entailment scores across sentence pairs. Both check whether generated text stays faithful to source documents -- critical in legal contexts where a fabricated citation could lead to sanctions.

**LEGAL-BERT** is a version of BERT pre-trained on legal text (court opinions, legislation, contracts) rather than general web text. It better understands legal language nuances like "estoppel," "res judicata," and "habeas corpus." Referenced as a baseline for domain-specific embeddings.

**LexGLUE** is a benchmark dataset for legal NLP with 7 tasks spanning contracts, court opinions, and legislation. Referenced as one of the standard evaluation frameworks for legal AI systems.

