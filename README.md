# Explainable Temporal Multi-Relational Hypergraph Neural Network (HGNN) for Link Prediction in Social Networks

Course project for **23CSE356 – Social Network Analytics**, School of Computing, Amrita Vishwa Vidyapeetham (Amritapuri), May 2026.

The project predicts links between subreddits in the [SNAP Reddit Hyperlink Network](https://snap.stanford.edu/data/soc-RedditHyperlinks.html) using a hypergraph neural network, compares it with GCN, Node2Vec and heuristic baselines, and ships an interactive Flask + D3.js dashboard.

**Full report:** [`docs/Project_Report_Explainable_Temporal_HGNN.pdf`](docs/Project_Report_Explainable_Temporal_HGNN.pdf)

## Authors

| Register No | Name |
|---|---|
| AM.AI.U4AID23023 | Abhishek Ranjith |
| AM.AI.U4AID23061 | T.V. Tarun Kumar |
| AM.AI.U4AID23063 | Devaduthan V |

The code history is preserved from the original repository ([tannirutarunkumar27/hypergraph-link-prediction](https://github.com/tannirutarunkumar27/hypergraph-link-prediction)).

> **Status: research prototype.** The pipeline runs end to end and trained weights are included, but the code in this repository is a simpler model than the one described in the report, and some claims are not yet validated. Please read [Report vs. code](#report-vs-code) and [Known limitations](#known-limitations).

## What the code in this repository implements

- **Data:** 10,000-row random sample (seed 42) of `soc-redditHyperlinks-title.tsv`; subreddits are nodes, each post is a hyperedge, incidence weights are the normalised timestamp (recency weight).
- **Model:** node embeddings (dim 32), two HGNN layers using the sparse propagation `Dv^-1/2 H De^-1 H^T Dv^-1/2` (linear, LayerNorm, ELU, dropout), skip sum of both layers, MLP link scorer, BCE loss with random negatives, Adam.
- **Baselines:** GCN, Node2Vec, Adamic-Adar, Common Neighbors.
- **Metrics:** AUC, MRR and Hits@10 (rank vs. 100 random negatives).
- **Dashboard:** Flask app serving link scores for a subreddit pair.

## Results

Numbers produced by the code in this repository (`results/data.json`, single run):

| Model | AUC | MRR | Hits@10 |
|-------|-----|-----|---------|
| Node2Vec | **0.9848** | **0.8668** | **0.9857** |
| HGNN (this repo) | 0.9648 | 0.4186 | 0.7628 |
| GCN | 0.9608 | 0.2986 | 0.5989 |
| Common Neighbors | 0.6470 | 0.6079 | 0.8977 |
| Adamic-Adar | 0.6472 | 0.5778 | 0.8774 |

The HGNN beats GCN on all three metrics, but Node2Vec beats the HGNN on all three. These numbers are not held-out results (see limitation 1).

## Report vs. code

The report describes a richer architecture than the code in this repository, and its figures and tables do not all agree with each other. Until these are reconciled, treat the code and `results/data.json` as the reproducible reference.

| Report describes | In this repo? |
|---|---|
| 3 residual HGNN layers, hidden dim 96 | No: 2 layers, hidden dim 32 (`src/train.py`) |
| Relation-aware propagation and relation attention | No: link sentiment is encoded but not used by the model |
| GRU + temporal attention over 40 snapshots | No: `src/temporal.py` splits by time but is not used in training |
| Focal loss, hard negative mining, cosine-annealing LR | No: BCE, random negatives, constant LR |
| Attention-based explanations | No: `src/explain.py` prints neighbour and recency facts from the data |
| Reported HGNN AUC 0.8765, accuracy 0.8391, F1 0.7287, MRR 0.3513 | Not reproduced by this code (it produces AUC 0.9648, MRR 0.4186) |

Inconsistencies inside the report itself: the text and Table 2.2 give HGNN AUC 0.8765 and GCN 0.7849, while Fig. 2.2 (ROC legend 0.96) and Fig. 2.3 (bars 0.9648 and 0.9608) show the values from this repository; Fig. 2.1 shows loss falling from about 0.7 to 0.2, while the text states 0.4265 to 0.0315; Listing 4.6 hard-codes the final metrics instead of computing them.

## Known limitations

1. **No held-out test set.** The hypergraph, training and evaluation all use the same edges, so metrics measure fit to seen edges, not prediction of future links.
2. **Hyperedges have exactly two nodes** (one post = one source + one target), so the hypergraph is effectively a weighted graph and "higher-order" modelling is not demonstrated.
3. **Dashboard analytics are placeholders.** In `app.py` the temporal chart values and the 5x5 heatmap are random, the confidence value is a formula of the score, and the endpoint returns a random probability if the model or `data/processed_data.csv` is missing. Only the link probability from a loaded model is real.
4. **Explainability is heuristic** and does not explain the neural network's decision.
5. `results/RESEARCH_REPORT.md` is from an earlier run (AUC 0.9665) and does not match `results/data.json`.
6. `server.py` (FastAPI) is an experimental second backend.

## Getting started

```bash
git clone https://github.com/abhishekranjith23/explainable-temporal-hgnn-link-prediction.git
cd explainable-temporal-hgnn-link-prediction
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

1. Download `soc-redditHyperlinks-title.tsv` from SNAP into `data/`.
2. Run from the repo root:

```bash
python src/preprocess.py    # data/processed_data.csv + models/user_map.json
python src/train.py         # models/*.pt + plots/loss_curve.png
python src/evaluate.py      # results/data.json + plots/
python app.py               # http://127.0.0.1:5000
```

`models/user_map.json` must match the weights, so retrain after re-running preprocessing.

## Project structure

```
app.py          Flask backend (dashboard + /predict)
server.py       Experimental FastAPI backend
src/            preprocess, hypergraph, temporal, model, train, evaluate, explain
baselines/      GCN, Node2Vec, Adamic-Adar, Common Neighbors
frontend/       index.html, js (D3), css
models/         Trained weights and subreddit index
results/        data.json, RESEARCH_REPORT.md (older run)
docs/           Project report (PDF)
plots/          Generated figures
```

## Roadmap

- [ ] **Temporal train/val/test split** with the hypergraph built from training edges only.
- [ ] **Bring the code in line with the report**: relation-aware layers, GRU + temporal attention, focal loss, hard negative mining. If a newer local version exists, add it here.
- [ ] Regenerate all report tables and figures from one evaluation run; report mean ± std over several seeds.
- [ ] Meaningful hyperedges (group by author or time window) and an ablation against a pairwise graph.
- [ ] Replace the placeholder dashboard panels with real outputs; return a clear "model not loaded" error instead of random values.
- [ ] Add a smoke-test CI workflow.

## License

MIT, see [LICENSE](LICENSE). Dataset: Kumar et al., *Community Interaction and Conflict on the Web*, WWW 2018 (SNAP).
