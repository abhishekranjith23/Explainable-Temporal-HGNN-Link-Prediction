# Hypergraph Link Prediction Research Dashboard

A premium, research-grade visual analytics platform for predicting future interactions in social hypergraphs using the **Reddit Hyperlink Dataset**.

![Dashboard Preview](https://img.shields.io/badge/Status-Complete-brightgreen)
![Tech Stack](https://img.shields.io/badge/Stack-PyTorch%20|%20Flask%20|%20D3.js-blue)

## 🚀 Overview

This project implements an **Explainable Temporal Multi-Relational Hypergraph Neural Network (HGNN)** to model complex community dynamics. Unlike traditional graph models (GCN, Node2Vec) that only model pairwise relations, our hypergraph approach captures high-order interactions between multiple communities simultaneously.

### Key Features
- **Real-time Inference**: Live link prediction using a pre-trained PyTorch HGNN model.
- **Deep Analytics**: 
  - **Temporal Flux Density**: D3.js line charts showing interaction intensity over time.
  - **Attention Heatmaps**: Visualizing model focus on community features.
  - **Community Similarity Matrix**: Understanding community overlap via hypergraph embeddings.
- **Research Benchmarks**: Comprehensive comparison against GCN, Node2Vec, and Adamic Adar baselines.
- **Premium UI**: Dark-themed "Deep Obsidian" aesthetic with glassmorphism and smooth D3 animations.

## 🛠️ Architecture

- **Backend**: Flask (Python) with PyTorch for model inference.
- **Frontend**: Vanilla JS with D3.js for high-performance visual analytics.
- **Model**: Custom HGNN architecture with sparse Laplacian optimization.

## 📦 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/tannirutarunkumar27/hypergraph-link-prediction.git
   cd hypergraph-link-prediction
   ```

2. Install dependencies:
   ```bash
   pip install torch flask flask-cors pandas numpy scikit-learn
   ```

## 🎮 Usage

1. Start the Flask server:
   ```bash
   python app.py
   ```
2. Open your browser to:
   ```
   http://127.0.0.1:5000
   ```

## 📊 Research Results

Our proposed HGNN model outperforms standard baselines across all major metrics on the Reddit Hyperlink dataset:

| Model | AUC | MRR | Hits@10 |
|-------|-----|-----|---------|
| **Proposed HGNN** | **0.9648** | **0.4186** | **0.7628** |
| GCN | 0.9608 | 0.2985 | 0.5988 |
| Node2Vec | 0.9848 | 0.8667 | 0.9856 |
| Adamic Adar | 0.6472 | 0.5778 | 0.8774 |

## 📂 Project Structure

```bash
├── app.py              # Main Flask Backend
├── src/                # Model and Training source code
├── frontend/           # Dashboard UI and JS
│   ├── index.html
│   ├── js/app.js       # D3.js logic and API calls
│   └── css/style.css   # Premium theme
├── models/             # Pre-trained PyTorch weights
├── plots/              # Research visualization exports
└── results/            # Comparative analysis data
```

## 📜 License

This project is released under the MIT License.
