# ProDeL: Prospective Decoupled Learning for Future Node Importance Scoring in Dynamic Networks

This repository provides the official implementation code for **ProDeL**, a two-stage prospective decoupled learning framework for predicting future node importance in dynamic networks.

The code implements the complete pipeline, including temporal network preprocessing, sliding-window construction, spatio-temporal representation learning, two-stage optimization, evaluation, and visualization.


---

## Framework Overview


### （1）Temporal Network Construction and Prediction Setting


The framework first transforms continuous temporal interactions into a sequence of temporal graph snapshots.

A sliding-window strategy is then applied to construct prediction samples, where historical snapshots are used as model inputs and future snapshots are used as prediction targets.


<p align="center">
<img src="./Figures/split+window.png" width="60%">
</p>


This design separates observed historical information from future prediction targets and prevents information leakage during prospective node importance prediction.


---

### （2）Overall Two-Stage Learning Framework


The proposed framework adopts a two-stage learning paradigm to predict future node importance.


<p align="center">
<img src="./Figures/Two_stage.png" width="90%">
</p>


**Stage I: Categorization-based Coarse Screening**

The first stage formulates future node importance prediction as a classification task and identifies potential future top-k important nodes from historical temporal graph sequences.


**Stage II: Regression-based Refined Scoring**

The second stage further refines the importance estimation by predicting continuous node importance scores and generating final rankings.


The framework integrates:

- Spatial graph representation learning
- Temporal dependency modeling
- Node evolution modeling
- Auxiliary graph reconstruction


---

## Repository Structure

```text
ProDeL/
│
├── Data/
│   ├── datasets/
│   │   ├── raw/
│   │   └── processed_data/
│   │
│   ├── data_processing.py
│   ├── dataloader.py
│   └── temporal_degree_labels.py
│
├── Figures/
│   ├── Two_stage.png
│   ├── split+window.png
│   └── results/
│
├── Model/
│   ├── gnn.py
│   └── graph.py
│
├── Project/
│   ├── train.py
│   ├── metrics.py
│   └── visulation.py
│
├── README.md
│
└── main.py

```

------------------------------------------------------------------------

## Code Organization

The implementation is organized into four main components:

1. Data preprocessing  
2. Temporal graph construction  
3. Model architecture  
4. Training and evaluation  


| Component | File | Description |
|-----------|------|-------------|
| Data Preprocessing | `Data/data_processing.py` | Preprocesses raw temporal interaction data, including data cleaning, timestamp processing, temporal snapshot construction, and sliding-window generation. |
| Data Preprocessing | `Data/dataloader.py` | Loads processed temporal graph data and converts graph information into model-ready inputs. |
| Label Generation | `Data/temporal_degree_labels.py` | Generates future node importance labels used for supervised learning and prediction evaluation. |
| Graph Construction | `Model/graph.py` | Constructs temporal graph datasets and organizes sequential graph snapshots for spatio-temporal learning. |
| Spatial-Temporal Model | `Model/gnn.py` | Implements the graph neural network modules, including spatial representation learning and temporal dependency modeling. |
| Training Pipeline | `Project/train.py` | Provides the complete training procedure, including model optimization, loss computation, validation, and model selection. |
| Evaluation | `Project/metrics.py` | Implements evaluation metrics for node importance ranking and prediction performance assessment. |
| Visualization | `Project/visulation.py` | Provides visualization utilities for experimental results and model analysis. |
| Experiment Entry | `main.py` | Serves as the main entry point for experiments, including configuration settings, dataset loading, model initialization, and training execution. |

------------------------------------------------------------------------
## Training Strategy

The training process follows a two-stage learning paradigm, where the model first performs coarse candidate screening and then refines continuous node importance scores.

| Stage | Task Type | Main Objective | Loss Function | Description |
|-------|-----------|----------------|---------------|-------------|
| Stage I: Categorization-based Coarse Screening | Classification | Identify potential future top-k important nodes from historical temporal graph sequences. | Binary Classification Loss (`L_cls`) | The model learns discriminative representations to distinguish future important nodes from other nodes and provides candidate nodes for subsequent refinement. |
| Stage II: Regression-based Refined Scoring | Regression | Predict continuous future node importance scores and generate final rankings. | Regression Loss (`L_MSE`) + Graph Reconstruction Loss (`L_GAE`) | The model fine-tunes learned representations to estimate importance scores while preserving structural information through auxiliary graph reconstruction. |

---

## Requirements

Python >= 3.8

Main dependencies:

torch torch-geometric numpy pandas networkx scikit-learn matplotlib tqdm

------------------------------------------------------------------------

## Running

After preparing datasets and modifying paths:

python main.py

------------------------------------------------------------------------

## License

This repository is released for academic research purposes.



