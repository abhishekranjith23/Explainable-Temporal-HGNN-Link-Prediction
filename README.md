# Explainable Temporal Multi-Relational Hypergraph Neural Network for Link Prediction in Social Networks

<p align="center">

  <img src="https://img.shields.io/badge/Python-3.x-blue?style=for-the-badge&logo=python" alt="Python">

  <img src="https://img.shields.io/badge/PyTorch-Deep%20Learning-red?style=for-the-badge&logo=pytorch" alt="PyTorch">

  <img src="https://img.shields.io/badge/Hypergraph%20Neural%20Network-HGNN-purple?style=for-the-badge" alt="HGNN">

  <img src="https://img.shields.io/badge/Temporal%20Learning-GRU-orange?style=for-the-badge" alt="Temporal Learning">

  <img src="https://img.shields.io/badge/Task-Link%20Prediction-green?style=for-the-badge" alt="Link Prediction">

</p>

<p align="center">
  <b>Temporal Relation-Aware Sparse Hypergraph Neural Network for Dynamic Link Prediction and Explainable Graph Intelligence</b>
</p>

---

## 📌 Overview

This project presents an **Explainable Temporal Multi-Relational Hypergraph Neural Network (HGNN)** for **dynamic link prediction in evolving social networks**.

Traditional graph-based link prediction methods generally represent interactions as pairwise relationships between two nodes. However, real-world networks often contain:

- Higher-order group interactions
- Multiple relationship types
- Temporal evolution
- Dynamic interaction patterns
- Complex structural dependencies

To address these challenges, this project models Reddit hyperlink interactions as a **temporal multi-relational hypergraph**, where subreddit communities are represented as nodes and higher-order interactions are represented using hyperedges.

The proposed framework combines:

- Temporal hypergraph representation
- Relation-aware hypergraph propagation
- Sparse hypergraph computation
- Multi-layer HGNN learning
- GRU-based temporal modeling
- Temporal attention
- Residual aggregation
- Focal loss
- Hard negative mining
- Regularization and gradient stabilization
- Dynamic link prediction
- Explainability mechanisms
- Baseline comparison

The project is designed to investigate how **higher-order relationships, temporal evolution, and multiple interaction relations** can be jointly modeled for dynamic link prediction.

---

## 🎯 Problem Statement

Social networks are dynamic systems in which relationships continuously evolve over time.

A conventional graph represents an interaction as:

```text
Node A ───────── Node B
