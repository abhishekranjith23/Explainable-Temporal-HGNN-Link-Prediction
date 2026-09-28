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

This project develops an **Explainable Temporal Multi-Relational Hypergraph Neural Network (HGNN)** for **dynamic link prediction in evolving social networks**.

The system models Reddit hyperlink interactions as a **temporal multi-relational hypergraph**, enabling the model to capture:

- Higher-order group interactions
- Multiple interaction relations
- Temporal evolution of the network
- Structural dependencies between communities

The proposed framework combines **sparse hypergraph learning, relation-aware propagation, GRU-based temporal modeling, temporal attention, advanced optimization, and explainability**.

---

## 🎯 Problem Statement

Conventional graph-based link prediction primarily models pairwise relationships between nodes.

However, real-world social networks contain:

- Group-level interactions
- Multiple relationship types
- Evolving connections over time
- Complex higher-order dependencies

This project addresses these limitations by representing the network as a **temporal multi-relational hypergraph** and learning evolving node representations for dynamic link prediction.

---

## 💡 Proposed Approach

The complete pipeline is:

```text
Reddit Hyperlink Dataset
          ↓
Data Preprocessing
          ↓
Temporal Snapshot Generation
          ↓
Multi-Relational Hypergraph Construction
          ↓
Sparse Hypergraph Propagation
          ↓
Relation-Aware HGNN
          ↓
Temporal GRU
          ↓
Temporal Attention
          ↓
Residual Aggregation
          ↓
MLP Link Prediction
          ↓
Explainability & Evaluation
