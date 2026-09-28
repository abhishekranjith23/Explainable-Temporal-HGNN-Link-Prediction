

# Explainable Temporal Multi-Relational Hypergraph Neural Network for Link Prediction

<p align="left">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/Graph%20Neural%20Networks-6A1B9A?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white" />
  <img src="https://img.shields.io/badge/D3.js-F9A03C?style=for-the-badge&logo=d3.js&logoColor=white" />
  <img src="https://img.shields.io/badge/Git-F05032?style=for-the-badge&logo=git&logoColor=white" />
</p>
A deep learning framework for **link prediction in evolving social networks** using Hypergraph Neural Networks (HGNNs), temporal interaction modeling, relation-aware propagation, and explainable predictions.

The project uses the **Reddit Hyperlink Network** to study how relationships between subreddit communities can be represented and predicted from historical interaction patterns.

---

## Overview

Traditional Graph Neural Networks represent relationships mainly as pairwise edges:

Community A ───── Community B

However, interactions in real-world social networks can contain richer structural and temporal information. An interaction may be associated with a timestamp, interaction type, post context, and relationships involving multiple communities.

This project investigates a **temporal multi-relational hypergraph-based approach** for learning these interaction patterns.

The overall pipeline transforms raw Reddit hyperlink interactions into a structured learning representation:

```text
Reddit Hyperlink Dataset
          │
          ▼
Data Preprocessing
          │
          ▼
Temporal Snapshot Generation
          │
          ▼
Multi-Relational Hypergraph Construction
          │
          ▼
Sparse Hypergraph Propagation
          │
          ▼
Temporal Relation-Aware HGNN
          │
          ▼
GRU + Temporal Attention
          │
          ▼
Node Representations
          │
          ▼
MLP Link Prediction
          │
          ▼
Evaluation + Explainability
````

---

## Key Features

* Temporal hypergraph representation of social-network interactions
* Multi-relational hypergraph learning
* Sparse hypergraph propagation
* Temporal snapshot-based learning
* Relation-aware message propagation
* GRU-based temporal representation learning
* Temporal attention
* Residual HGNN aggregation
* Focal loss and hard-negative mining
* Link prediction using learned node representations
* Comparison with classical and graph-based baselines
* Structural and temporal explainability
* Interactive network visualization dashboard

---

## Problem Statement

The objective is to predict whether a relationship is likely to exist between two subreddit communities based on their historical interaction patterns.

Given a sequence of historical interactions:

```text
Past Interactions
       │
       ▼
Temporal Network Representation
       │
       ▼
Learn Node Representations
       │
       ▼
Predict Candidate Links
```

The project investigates whether combining:

* higher-order interactions,
* temporal evolution,
* multiple relation types, and
* neural representation learning

can provide useful representations for dynamic link prediction.

---

# Dataset

The project uses the **SNAP Reddit Hyperlink Network**, which contains hyperlink interactions between subreddit communities.

Dataset:

`soc-redditHyperlinks-title.tsv`

Each interaction contains information such as:

| Attribute        | Description                            |
| ---------------- | -------------------------------------- |
| Source subreddit | Community containing the hyperlink     |
| Target subreddit | Community being linked to              |
| Post ID          | Associated Reddit post                 |
| Timestamp        | Time of the interaction                |
| Interaction type | Relationship / interaction information |

Dataset source:

[https://snap.stanford.edu/data/soc-RedditHyperlinks.html](https://snap.stanford.edu/data/soc-RedditHyperlinks.html)

### Dataset Statistics

The final research experiment uses:

| Property                  |                Value |
| ------------------------- | -------------------: |
| Sampled interactions      |               10,000 |
| Global subreddit nodes    |                5,943 |
| Temporal snapshots        |                   40 |
| Relation types            |                    2 |
| Train / Test split        |            80% / 20% |
| Positive : Negative ratio |                1 : 1 |
| Negative sampling         | Hard Negative Mining |
| Random seed               |                   42 |

---

# Data Preprocessing

The raw Reddit interaction data is transformed into a representation suitable for temporal hypergraph learning.

### 1. Data Cleaning

Duplicate and incomplete interactions are removed before model construction.

```text
Raw Dataset
     │
     ├── Remove missing values
     ├── Remove duplicates
     └── Normalize interaction data
```

### 2. Node Encoding

Each subreddit is mapped to a unique numerical identifier.

```text
subreddit name
      │
      ▼
integer node ID
```

This allows the communities to be represented as nodes in the neural network.

### 3. Temporal Encoding

Interaction timestamps are converted into normalized temporal values.

Recent interactions receive greater temporal importance so that the model can preserve information about when interactions occurred.

### 4. Temporal Snapshot Generation

The interaction network is divided into **40 temporal snapshots**.

```text
Snapshot 1  → Early interactions
Snapshot 2  → Subsequent interactions
Snapshot 3  → Later interactions
     ...
Snapshot 40 → Latest interactions
```

This provides a representation of how the social network evolves over time.

---

# Hypergraph Representation

A conventional graph represents an interaction using a pairwise edge:

```text
A ───── B
```

A hypergraph can represent relationships involving multiple nodes:

```text
        ┌──────────── Hyperedge ────────────┐
        │                                   │
        A              B                    C
        │              │                    │
        └──────────────┴────────────────────┘
```

This allows higher-order interaction structure to be incorporated into representation learning.

For this project:

| Component            | Representation              |
| -------------------- | --------------------------- |
| Nodes                | Subreddit communities       |
| Hyperedges           | Interaction events          |
| Relations            | Interaction categories      |
| Temporal information | Interaction timestamps      |
| Snapshots            | Network evolution over time |

---

# Sparse Hypergraph Propagation

The HGNN uses a normalized hypergraph propagation operator:

$$
\Theta =
D_v^{-1/2} H D_e^{-1} H^T D_v^{-1/2}
$$

where:

* \(H\) is the hypergraph incidence matrix
* \(D_v\) is the node-degree matrix
* \(D_e\) is the hyperedge-degree matrix

The sparse formulation allows the model to perform hypergraph propagation without explicitly constructing dense matrices.

Conceptually:

```text
Node Features
      │
      ▼
Hypergraph Incidence Structure
      │
      ▼
Sparse Hypergraph Propagation
      │
      ▼
Updated Node Representations
```

---

# Temporal Multi-Relational HGNN

The proposed architecture combines hypergraph learning with temporal and relation-aware modeling.

```text
Temporal Hypergraph
        │
        ▼
Relation-Aware HGNN Layers
        │
        ▼
Sparse Hypergraph Propagation
        │
        ▼
Snapshot-Level Node Representations
        │
        ▼
GRU
        │
        ▼
Temporal Attention
        │
        ▼
Final Node Representations
```

The research architecture incorporates:

* Relation-specific transformations
* Relation-aware attention
* Hypergraph message propagation
* Temporal sequence learning
* GRU-based temporal modeling
* Temporal attention
* Residual aggregation

---

# Relation-Aware Learning

The network contains multiple interaction relations.

Instead of treating every relationship identically, the model learns relation-specific transformations and aggregates information across relations.

```text
                    ┌── Relation 1
                    │
Hypergraph ─────────┼── Relation 2
                    │
                    └── Relation-aware Aggregation
                                │
                                ▼
                         Node Representation
```

This enables the model to capture differences between interaction types during representation learning.

---

# Temporal Learning

Social-network relationships change over time.

A relationship that is important in one period may become less important later.

The project therefore models the network as a sequence of temporal snapshots:

```text
Snapshot 1 ──┐
Snapshot 2   │
Snapshot 3   │
     ...     ├──► GRU ──► Temporal Attention
Snapshot 40  │
             ┘
                    │
                    ▼
          Temporal Node Representation
```

The GRU captures sequential dependencies between snapshots, while temporal attention identifies important temporal information.

---

# Link Prediction

After learning node representations, the model predicts the likelihood of a relationship between two candidate subreddit communities.

```text
Node A Embedding ─────┐
                      │
                      ▼
                MLP Predictor
                      │
                      ▼
                 Link Score
                      ▲
                      │
Node B Embedding ─────┘
```

The learned representations are therefore converted into a link prediction score for candidate node pairs.

---

# Training Optimizations

Several optimization strategies are incorporated into the research framework.

| Technique            | Purpose                              |
| -------------------- | ------------------------------------ |
| Focal Loss           | Focus learning on difficult examples |
| Hard Negative Mining | Select challenging negative links    |
| Cosine Annealing     | Improve learning-rate scheduling     |
| Residual Aggregation | Improve information flow             |
| Dropout              | Reduce overfitting                   |
| Weight Decay         | Regularization                       |
| Gradient Clipping    | Improve training stability           |

The reported training loss decreased from:

```text
0.4265 → 0.0315
```

representing approximately **92.6% loss reduction**.

---

# Baseline Models

The Temporal HGNN is evaluated against several baseline approaches.

## Common Neighbors

A classical link-prediction heuristic based on the number of shared neighbors between two nodes.

$$
Score(u,v)=|N(u)\cap N(v)|
$$

---

## Adamic-Adar

An extension of Common Neighbors that assigns greater importance to less-connected shared neighbors.

$$
Score(u,v)=
\sum_{z\in N(u)\cap N(v)}
\frac{1}{\log |N(z)|}
$$

---

## Node2Vec

Node2Vec learns node embeddings using biased random walks and Skip-gram optimization.

The experimental configuration uses:

* Embedding dimension: 32
* Walk length: 5
* Walks per node: 20
* Window size: 3

---

## Graph Convolutional Network

A two-layer GCN is used as a graph-based deep learning baseline with an MLP-based link predictor.

---

## Temporal HGNN

The proposed approach combines:

* Hypergraph representation learning
* Temporal learning
* Relation-aware propagation
* Sparse computation
* Attention mechanisms
* Neural link prediction

---

# Evaluation Metrics

The project evaluates both classification and ranking performance.

### Classification Metrics

**AUC**

Measures the ability to distinguish positive and negative links.

**Accuracy**

Measures the proportion of correctly classified links.

**Precision**

Measures the proportion of predicted positive links that are actually positive.

**Recall**

Measures the proportion of actual positive links successfully retrieved.

**F1-Score**

Measures the balance between precision and recall.

### Ranking Metrics

**Mean Reciprocal Rank (MRR)**

Measures the ranking position of the first correct prediction.

**Hits@10**

Measures whether the correct prediction appears within the top 10 candidates.

---

# Final Experimental Results

The following table contains the **final experimental results reported in the academic project report**.

| Model             |        AUC |   Accuracy |  Precision | Recall |   F1-Score |
| ----------------- | ---------: | ---------: | ---------: | -----: | ---------: |
| Adamic-Adar       |     0.6462 |     0.7184 |     0.1873 | 0.5001 |     0.2707 |
| Common Neighbors  |     0.6459 |     0.7590 |     0.3125 | 0.8332 |     0.4557 |
| Node2Vec          | **0.9854** |     0.3335 |     0.3334 | 1.0000 |     0.5001 |
| GCN               |     0.7849 |     0.6720 |     0.6255 | 0.6733 |     0.6457 |
| **Temporal HGNN** |     0.8765 | **0.8391** | **0.8319** | 0.6482 | **0.7287** |

### Temporal HGNN Ranking Performance

| Metric  |  Value |
| ------- | -----: |
| MRR     | 0.3513 |
| Hits@10 | 0.5969 |

### Final Temporal HGNN Metrics

```text
AUC       = 0.8765
Accuracy  = 0.8391
Precision = 0.8319
Recall    = 0.6482
F1-Score  = 0.7287
MRR       = 0.3513
Hits@10   = 0.5969
```

The experiments demonstrate the performance of the proposed temporal hypergraph framework across classification and ranking metrics while providing a comparison with heuristic, embedding-based, and graph neural network approaches.

---

# Explainability

The project includes an explainability component to provide contextual information behind model predictions.

The explanation framework considers factors such as:

* Structural relationships
* Shared neighboring communities
* Temporal interaction patterns
* Relation-aware information
* Model prediction reasoning

For example, a prediction can be interpreted using information about the historical interaction structure between the candidate communities rather than presenting only a numerical score.

---

# Interactive Visualization Dashboard

The project includes an interactive web-based dashboard for exploring the network and link prediction results.

### Frontend

* HTML
* CSS
* JavaScript
* D3.js

### Backend

* Flask
* Python

The dashboard connects the trained model with an interactive visualization layer for exploring network structure and predictions.

Conceptually:

```text
User Input
    │
    ▼
Subreddit Pair
    │
    ▼
Backend API
    │
    ▼
Trained HGNN
    │
    ▼
Prediction + Explanation
    │
    ▼
Interactive Visualization
```

---

# End-to-End Architecture

```text
                    Reddit Hyperlink Dataset
                              │
                              ▼
                    Data Preprocessing
                              │
                              ▼
                  Temporal Snapshot Generation
                              │
                              ▼
                Multi-Relational Hypergraph
                       Construction
                              │
                              ▼
                 Sparse Hypergraph Propagation
                              │
                              ▼
                 Relation-Aware HGNN Layers
                              │
                              ▼
                     GRU Temporal Learning
                              │
                              ▼
                      Temporal Attention
                              │
                              ▼
                    Node Representations
                              │
                              ▼
                     MLP Link Predictor
                              │
                              ▼
                   Link Prediction Scores
                              │
                 ┌────────────┴────────────┐
                 ▼                         ▼
          Model Evaluation          Explainability
                 │                         │
                 └────────────┬────────────┘
                              ▼
                   Interactive Dashboard
```

---

# Repository Structure

```text
explainable-temporal-hgnn-link-prediction/
│
├── src/
│   ├── preprocess.py
│   ├── hypergraph.py
│   ├── temporal.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   ├── explain.py
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

### Graph and Network Learning

* Hypergraph Neural Networks
* Graph Convolutional Networks
* Node2Vec
* Common Neighbors
* Adamic-Adar
* Temporal Graph Learning
* Sparse Matrix Computation

### Web and Visualization

* Flask
* HTML
* CSS
* JavaScript
* D3.js

### Development

* Git
* GitHub

---

# Getting Started

## Clone the Repository

```bash
git clone https://github.com/abhishekranjith23/explainable-temporal-hgnn-link-prediction.git

cd explainable-temporal-hgnn-link-prediction
```

## Create a Virtual Environment

### Windows

```powershell
python -m venv .venv

.venv\Scripts\activate
```

### Linux / macOS

```bash
python -m venv .venv

source .venv/bin/activate
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Add the Dataset

Download:

```text
soc-redditHyperlinks-title.tsv
```

from the SNAP Reddit Hyperlink Dataset and place it inside:

```text
data/
```

## Preprocess the Data

```bash
python src/preprocess.py
```

## Train the Model

```bash
python src/train.py
```

## Evaluate the Model

```bash
python src/evaluate.py
```

## Run the Dashboard

```bash
python app.py
```

Then open:

```text
http://127.0.0.1:5000
```

---

# Research Contributions

The project combines multiple areas of machine learning and network analysis:

### Higher-Order Representation Learning

Uses hypergraph structures to represent interactions beyond simple pairwise edges.

### Temporal Network Learning

Models the evolution of social-network interactions across multiple temporal snapshots.

### Relation-Aware Learning

Uses relation-specific transformations to capture differences between interaction types.

### Sparse Hypergraph Learning

Uses sparse matrix operations for more efficient hypergraph propagation.

### Neural Link Prediction

Uses learned node representations with an MLP-based predictor.

### Explainability

Provides structural and temporal context around model predictions.

### Interactive Deployment

Connects the research model with a web-based visualization dashboard.

---

# Future Work

Potential extensions include:

* Transformer-based temporal learning
* Contrastive hypergraph learning
* Richer node and hyperedge features
* Advanced explainability methods
* Streaming temporal hypergraphs
* Distributed HGNN training
* Federated graph learning
* More extensive temporal evaluation

---

# Project Context

This project was developed as a **three-member course project for 23CSE356 – Social Network Analytics** at **Amrita Vishwa Vidyapeetham, Amritapuri Campus**.

The project was based on research directions suggested during the course and involved the design, implementation, experimentation, and evaluation of a temporal hypergraph-based link prediction framework.

### Team

| Register Number  | Contributor          |
| ---------------- | -------------------- |
| AM.AI.U4AID23023 | **Abhishek Ranjith** |
| AM.AI.U4AID23061 | T. V. Tarun Kumar    |
| AM.AI.U4AID23063 | Devaduthan V         |

---

# Project Report

The complete academic report is available in the repository:

[**View Project Report**](docs/Project_Report_Explainable_Temporal_HGNN.pdf)

---

# Project at a Glance

| Category                | Details                                      |
| ----------------------- | -------------------------------------------- |
| Domain                  | Social Network Analytics                     |
| Task                    | Dynamic Link Prediction                      |
| Dataset                 | SNAP Reddit Hyperlink Network                |
| Sampled Interactions    | 10,000                                       |
| Global Nodes            | 5,943                                        |
| Temporal Snapshots      | 40                                           |
| Relation Types          | 2                                            |
| Primary Model           | Temporal Multi-Relational HGNN               |
| Baselines               | Node2Vec, GCN, Common Neighbors, Adamic-Adar |
| Temporal Learning       | GRU + Temporal Attention                     |
| Link Predictor          | MLP                                          |
| Explainability          | Structural + Temporal + Relation-aware       |
| Deep Learning Framework | PyTorch                                      |
| Dashboard               | Flask + D3.js                                |
| Final HGNN AUC          | **0.8765**                                   |
| Final HGNN Accuracy     | **0.8391**                                   |
| Final HGNN F1-Score     | **0.7287**                                   |
| Final HGNN MRR          | **0.3513**                                   |
| Final HGNN Hits@10      | **0.5969**                                   |

---

# Summary

This project presents an **Explainable Temporal Multi-Relational Hypergraph Neural Network for dynamic link prediction in social networks**.

The system transforms Reddit hyperlink interactions into temporal hypergraph representations and combines:

**Hypergraph Learning + Temporal Modeling + Relation-Aware Learning + Neural Link Prediction + Explainability**

The project includes the complete research pipeline from preprocessing and hypergraph construction to model training, baseline evaluation, explainability, and interactive visualization.

It demonstrates the application of **deep learning and graph representation learning to evolving social-network data**, with a focus on higher-order interactions and temporal dynamics.

