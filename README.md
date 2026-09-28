# Explainable Temporal Multi-Relational Hypergraph Neural Network for Link Prediction

A deep learning framework for **link prediction in evolving social networks** using Hypergraph Neural Networks (HGNNs), temporal interaction modeling, sparse hypergraph propagation, and an interactive visualization dashboard.

The project uses the **Reddit Hyperlink Network** to study how relationships between subreddit communities can be represented and predicted from historical interaction patterns.

---

## Overview

Traditional Graph Neural Networks represent relationships as **pairwise edges**:

```text
Community A ───── Community B
```

However, interactions in real-world social networks can contain richer structural and temporal information. A single interaction may be associated with a post, timestamp, interaction type, and community context.

This project investigates a **temporal hypergraph-based approach** for learning these interaction patterns.

The overall system transforms raw Reddit hyperlink interactions into a structured learning pipeline:

```text
Reddit Hyperlink Dataset
          │
          ▼
Data Cleaning & Preprocessing
          │
          ▼
Subreddit / Node Encoding
          │
          ▼
Temporal Weighting
          │
          ▼
Temporal Hypergraph Construction
          │
          ▼
Sparse Hypergraph Propagation
          │
          ▼
HGNN Representation Learning
          │
          ▼
Node Embeddings
          │
          ▼
Link Prediction
          │
          ▼
Evaluation & Baseline Comparison
          │
          ▼
Interactive Flask + D3.js Dashboard
```

The project combines **graph representation learning, hypergraph learning, temporal network analysis, link prediction, explainability, and web-based visualization** into a single end-to-end system.

---

## Problem Statement

Social networks are dynamic systems in which relationships continuously evolve.

Conventional link prediction methods often model networks as simple graphs containing pairwise relationships:

```text
u ───── v
```

This representation can miss structural information associated with the interaction itself.

The objective of this project is to investigate whether **hypergraph-based representation learning** can capture richer interaction structures and improve link prediction between social communities.

Formally, given historical interactions between communities, the model learns representations of the participating nodes and estimates the probability of a future or candidate interaction:

$$
P((u,v)\in E)
$$

where:

* \(u\) and \(v\) are subreddit communities
* \(E\) represents observed interactions
* the learned representations encode structural and temporal information from the network

---

# Why Hypergraphs?

A conventional graph represents relationships using edges connecting two nodes.

A hypergraph generalizes this concept by allowing a hyperedge to represent an interaction involving multiple nodes:

```text
Traditional Graph

A ───── B
B ───── C


Hypergraph

       ┌───────────────┐
       │   Hyperedge   │
       │               │
       A       B       C
       │       │       │
       └───────┴───────┘
```

This provides a more flexible representation for interaction systems where relationships are associated with a shared event, post, group, or context.

In this project:

* **Nodes** → subreddit communities
* **Hyperedges** → interaction events
* **Weights** → temporal importance
* **Relations** → interaction semantics
* **Snapshots** → different stages of network evolution

The research direction extends this representation toward **temporal and relation-aware hypergraph learning**.

---

# Dataset

The project uses the **SNAP Reddit Hyperlink Network**, which contains hyperlink interactions between subreddit communities.

Each interaction contains information such as:

| Attribute        | Description                                |
| ---------------- | ------------------------------------------ |
| Source subreddit | Community containing the hyperlink         |
| Target subreddit | Community referenced by the hyperlink      |
| Post ID          | Identifier associated with the interaction |
| Timestamp        | Time of the interaction                    |
| Link sentiment   | Interaction relation information           |

Dataset:

**Reddit Hyperlink Network — SNAP**

[Dataset source](https://snap.stanford.edu/data/soc-RedditHyperlinks.html)

For computational efficiency, the project works with a **10,000-interaction sample using random seed 42**.

After preprocessing, the research pipeline represents the network using approximately:

* **5,943 subreddit nodes**
* **40 temporal snapshots**
* Temporal interaction weights
* Sparse hypergraph representations
* Multiple interaction relation categories

---

# Data Preprocessing

Before constructing the learning representation, the raw Reddit data passes through a preprocessing pipeline.

### 1. Data Loading

The original TSV dataset is loaded and its fields are standardized for processing.

```text
SOURCE_SUBREDDIT
TARGET_SUBREDDIT
POST_ID
TIMESTAMP
LINK_SENTIMENT
```

These fields are mapped into the internal representation used by the pipeline.

### 2. Data Cleaning

The preprocessing stage removes:

* Missing records
* Duplicate interactions
* Invalid entries

This produces a cleaner interaction dataset for graph construction.

### 3. Subreddit Encoding

Every subreddit is mapped to a numerical node identifier.

For example:

```text
AskReddit       → 0
technology      → 1
MachineLearning → 2
gaming          → 3
...
```

This allows the neural network to operate on numerical node representations.

### 4. Temporal Encoding

Interaction timestamps are converted into normalized temporal values.

Recent interactions receive greater temporal importance, allowing the representation to preserve information about **when interactions occurred**.

### 5. Temporal Snapshot Generation

The interaction data is divided into temporal snapshots.

Conceptually:

```text
Snapshot 1 → Early interactions
Snapshot 2 → Subsequent interactions
Snapshot 3 → Later interactions
...
Snapshot 40 → Latest interactions
```

This provides a representation of how the social network evolves over time.

---

# Temporal Hypergraph Construction

After preprocessing, the interaction data is converted into a sparse hypergraph representation.

The hypergraph is represented as:

$$
G=(V,E,R,W,T)
$$

where:

* \(V\) → set of subreddit nodes
* \(E\) → interaction hyperedges
* \(R\) → relation types
* \(W\) → temporal weights
* \(T\) → timestamps

The interaction structure is represented using an **incidence matrix**:

$$
H_{ij} =
\begin{cases}
w_{ij}, & \text{if node } i \text{ participates in hyperedge } j\\
0, & \text{otherwise}
\end{cases}
$$

Sparse matrix representations are used to avoid unnecessary memory consumption.

---

# Sparse Hypergraph Propagation

A major component of the model is hypergraph message propagation.

Instead of propagating information only between directly connected node pairs, the hypergraph formulation allows information to propagate through the node–hyperedge structure.

The normalized propagation operator is based on:

$$
\Theta =
D_v^{-1/2}
H
D_e^{-1}
H^T
D_v^{-1/2}
$$

where:

* \(H\) is the hypergraph incidence matrix
* \(D_v\) is the node degree matrix
* \(D_e\) is the hyperedge degree matrix

This enables the model to aggregate structural information through the hypergraph.

---

# HGNN Architecture

The core implementation learns dense vector representations for subreddit nodes.

The implemented HGNN pipeline contains:

```text
Node Embeddings
      │
      ▼
HGNN Layer 1
      │
      ▼
HGNN Layer 2
      │
      ▼
Skip / Residual Aggregation
      │
      ▼
Final Node Representations
      │
      ▼
MLP Link Predictor
      │
      ▼
Interaction Probability
```

The implemented model uses:

* Node embedding dimension: **32**
* Two HGNN propagation layers
* Sparse hypergraph propagation
* Linear transformations
* Layer normalization
* ELU activation
* Dropout
* Skip aggregation

The model learns node representations by repeatedly propagating information through the hypergraph structure.

---

# Learning Node Representations

For each node \(v\), the model learns an embedding:

$$
h_v \in \mathbb{R}^{d}
$$

where \(d\) represents the embedding dimension.

The HGNN layers transform the representations by combining:

1. The node's current representation
2. Information propagated through the hypergraph
3. Learnable transformations
4. Non-linear activation

This allows structurally related subreddit communities to develop similar or informative latent representations.

---

# Link Prediction

Once node representations have been learned, the system predicts whether a link exists between two subreddit communities.

For two nodes \(u\) and \(v\):

$$
h_u,\ h_v
$$

are combined and passed to an MLP-based link predictor.

Conceptually:

```text
Subreddit A ──► Embedding A ──┐
                              ├──► MLP ──► Link Probability
Subreddit B ──► Embedding B ──┘
```

The output is a probability indicating how strongly the model predicts an interaction between the two communities.

The implemented training pipeline uses:

* Binary link prediction
* Positive interaction samples
* Random negative samples
* Binary Cross-Entropy loss
* Adam optimization

---

# Baseline Models

To evaluate the usefulness of hypergraph representation learning, the project compares the HGNN with multiple established approaches.

### Common Neighbors

Predicts links based on the number of common neighboring nodes.

### Adamic-Adar

Weights common neighbors according to their structural rarity.

### Node2Vec

Learns node embeddings using biased random walks over the network.

### Graph Convolutional Network

Uses graph convolution and learned node representations for link prediction.

### HGNN

The proposed hypergraph-based representation learning approach.

The comparison provides a useful perspective on how different representation-learning strategies perform on the same social-network prediction task.

---

# Experimental Evaluation

The models are evaluated using both **classification and ranking metrics**.

### AUC

Measures the model's ability to distinguish positive interactions from negative interactions across classification thresholds.

### Mean Reciprocal Rank — MRR

Measures how highly the correct interaction is ranked among candidate links.

### Hits@10

Measures how frequently the correct link appears among the top 10 predictions.

These metrics provide complementary information about the model's prediction and ranking behavior.

---

# Results

The current implementation produces the following results from the stored evaluation output:

| Model            |        AUC |        MRR |    Hits@10 |
| ---------------- | ---------: | ---------: | ---------: |
| **Node2Vec**     | **0.9848** | **0.8668** | **0.9857** |
| **HGNN**         | **0.9648** | **0.4186** | **0.7628** |
| GCN              |     0.9608 |     0.2986 |     0.5989 |
| Common Neighbors |     0.6470 |     0.6079 |     0.8977 |
| Adamic-Adar      |     0.6472 |     0.5778 |     0.8774 |

The results show that the implemented **HGNN performs strongly against the graph-based GCN baseline**, while Node2Vec achieves higher performance on this particular evaluation run.

This comparison is useful because the objective is not simply to demonstrate a high score, but to investigate how **different network representations affect link prediction performance**.

---

# Explainability

An important part of the project is making predictions easier to interpret.

Instead of treating a prediction as only a probability, the system can examine structural and temporal information associated with the candidate subreddit pair.

The explainability component considers information such as:

* Neighborhood structure
* Related subreddit interactions
* Interaction recency
* Historical connectivity
* Relevant interaction information

This provides additional context around why a particular pair of communities may receive a higher link score.

The project therefore explores the broader goal of **interpretable graph intelligence**, where predictions can be examined alongside the network evidence that surrounds them.

---

# Interactive Visualization Dashboard

The project also includes a web-based interface for interacting with the trained model.

### Backend

The backend is implemented using:

**Flask**

It exposes prediction functionality for subreddit pairs.

Example workflow:

```text
User selects subreddit A
          │
          ▼
User selects subreddit B
          │
          ▼
Flask prediction endpoint
          │
          ▼
Trained HGNN model
          │
          ▼
Link probability
          │
          ▼
Dashboard visualization
```

### Frontend

The frontend uses:

* HTML
* CSS
* JavaScript
* D3.js

D3.js is used for interactive network and analytical visualizations.

The dashboard provides a practical interface for exploring the learned link-prediction system instead of interacting with the model only through command-line scripts.

---

# End-to-End Workflow

The complete project can be understood as six major stages.

### Stage 1 — Collect and Prepare Data

```text
Reddit Hyperlink Dataset
          ↓
Sampling
          ↓
Cleaning
          ↓
Node Encoding
          ↓
Timestamp Processing
```

### Stage 2 — Build the Network Representation

```text
Processed Interactions
          ↓
Temporal Weighting
          ↓
Hyperedge Construction
          ↓
Sparse Incidence Matrix
          ↓
Temporal Hypergraph
```

### Stage 3 — Learn Network Representations

```text
Node Features
     ↓
HGNN Propagation
     ↓
Layer Normalization
     ↓
Non-linear Transformation
     ↓
Skip Aggregation
     ↓
Node Embeddings
```

### Stage 4 — Predict Links

```text
Node Embedding u
        +
Node Embedding v
        ↓
       MLP
        ↓
Link Probability
```

### Stage 5 — Benchmark the Approach

```text
                 ┌── Common Neighbors
                 ├── Adamic-Adar
Input Network ───┼── Node2Vec
                 ├── GCN
                 └── HGNN
                        ↓
                 AUC / MRR / Hits@10
```

### Stage 6 — Visualize and Interact

```text
Trained Model
      ↓
Flask API
      ↓
D3.js Dashboard
      ↓
Interactive Prediction & Analysis
```

---

# Project Structure

```text
explainable-temporal-hgnn-link-prediction/
│
├── src/
│   ├── preprocess.py       # Data preprocessing
│   ├── hypergraph.py       # Hypergraph construction
│   ├── temporal.py         # Temporal processing
│   ├── model.py            # HGNN architecture
│   ├── train.py            # Model training
│   ├── evaluate.py         # Evaluation and metrics
│   ├── explain.py          # Prediction explanations
│   └── final_verification.py
│
├── baselines/
│   ├── common_neighbors.py
│   ├── heuristics.py
│   ├── node2vec_baseline.py
│   ├── node2vec_model.py
│   ├── gcn_baseline.py
│   └── gcn_model.py
│
├── frontend/
│   ├── index.html
│   ├── css/
│   │   └── style.css
│   └── js/
│       ├── app.js
│       └── visualization.js
│
├── models/
│   ├── hgnn_model.pt
│   ├── predictor.pt
│   └── user_map.json
│
├── results/
│   ├── data.json
│   └── RESEARCH_REPORT.md
│
├── plots/
│
├── data/
│
├── docs/
│   └── Project_Report_Explainable_Temporal_HGNN.pdf
│
├── app.py
├── server.py
├── requirements.txt
├── LICENSE
└── README.md
```

---

# Technologies Used

### Machine Learning

* Python
* PyTorch
* NumPy
* Pandas
* Scikit-learn

### Graph & Network Learning

* Hypergraph Neural Networks
* Graph Convolutional Networks
* Node2Vec
* Network-based heuristic methods
* Sparse matrix operations

### Backend

* Flask
* FastAPI

### Frontend & Visualization

* HTML
* CSS
* JavaScript
* D3.js

### Development

* Git
* GitHub
* Virtual environments

---

# Running the Project

## 1. Clone the Repository

```bash
git clone https://github.com/abhishekranjith23/explainable-temporal-hgnn-link-prediction.git
cd explainable-temporal-hgnn-link-prediction
```

## 2. Create a Virtual Environment

### Linux / macOS

```bash
python -m venv .venv
source .venv/bin/activate
```

### Windows

```powershell
python -m venv .venv
.venv\Scripts\activate
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## 4. Add the Dataset

Download:

```text
soc-redditHyperlinks-title.tsv
```

from the SNAP dataset and place it inside:

```text
data/
```

## 5. Preprocess the Data

```bash
python src/preprocess.py
```

This prepares the interaction data and generates the node mapping used by the model.

## 6. Train the HGNN

```bash
python src/train.py
```

Training produces the learned model weights and training artifacts.

## 7. Evaluate the Model

```bash
python src/evaluate.py
```

Evaluation generates prediction metrics and result files.

## 8. Run the Dashboard

```bash
python app.py
```

Then open:

```text
http://127.0.0.1:5000
```

---

# Research Direction

The project explores the combination of several areas of graph intelligence:

```text
Graph Learning
      +
Hypergraph Learning
      +
Temporal Network Analysis
      +
Representation Learning
      +
Link Prediction
      +
Explainability
      +
Interactive Visualization
```

The research design described in the accompanying report extends this direction toward:

* Relation-aware hypergraph propagation
* Temporal sequential learning
* GRU-based temporal representation learning
* Temporal attention
* Residual HGNN architectures
* Focal loss
* Hard-negative mining
* Sparse temporal computation
* Advanced explainability

These ideas form the broader research direction investigated during the project.

---

# Future Improvements

Several extensions can make the system more suitable for research-scale experimentation:

* Temporal train/validation/test splitting
* Stronger temporal evaluation for future-link prediction
* Relation-specific hypergraph propagation
* GRU and temporal-attention integration
* Hard-negative mining
* Focal loss
* Transformer-based temporal learning
* Meaningful higher-order hyperedge construction
* Richer node and interaction features
* Advanced explainability methods
* Real-time streaming hypergraph learning
* Larger-scale distributed training
* Automated experiment tracking and reproducibility

---

# Project Context

This work was developed as a **course project for 23CSE356 – Social Network Analytics** at Amrita Vishwa Vidyapeetham, Amritapuri.

The project was completed as a **three-member group project**, following research directions and ideas suggested as part of the course. The work involved studying the research problem, designing the network representation, implementing the learning pipeline, evaluating alternative approaches, and developing an interactive demonstration system.

### Team

| Register No.     | Contributor          |
| ---------------- | -------------------- |
| AM.AI.U4AID23023 | **Abhishek Ranjith** |
| AM.AI.U4AID23061 | T. V. Tarun Kumar    |
| AM.AI.U4AID23063 | Devaduthan V         |

---

# Project Report

The complete academic report containing the research motivation, methodology, architecture, experimentation, and future research directions is available here:

**[View the Project Report](docs/Project_Report_Explainable_Temporal_HGNN.pdf)**

---

# Key Takeaway

This project demonstrates an end-to-end approach to **social network link prediction using hypergraph-based deep learning**.

Starting from raw Reddit hyperlink interactions, the system:

```text
Raw Social Network Data
        ↓
Data Engineering
        ↓
Temporal Network Representation
        ↓
Sparse Hypergraph Construction
        ↓
HGNN Representation Learning
        ↓
Node Embeddings
        ↓
Link Prediction
        ↓
Baseline Comparison
        ↓
Explainability
        ↓
Interactive Visualization
```

The project brings together **data preprocessing, graph algorithms, deep learning, temporal modelling, evaluation, backend development, and interactive visualization** into a single research-oriented implementation.
