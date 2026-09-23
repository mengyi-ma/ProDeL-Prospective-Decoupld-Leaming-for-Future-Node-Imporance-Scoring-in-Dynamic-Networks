# ProDeL-Prospective-Decoupld-Leaming-for-Future-Node-Imporance-Scoring-in-Dynamic-Networks
ProDeL: Prospective Decoupled Learning for Future Node Importance Scoring in Dynamic Networks

This repository provides the official implementation code for ProDeL, a two-stage prospective decoupled learning framework for predicting future
node importance in dynamic networks.

The code includes the complete pipeline for temporal network preprocessing, sliding-window construction, graph representation learning, two-stage training, evaluation, and visualization.

------------------------------------------------------------------------

## Framework Overview

### Temporal Network Construction and Sliding-Window Partition

The temporal network preprocessing procedure converts continuous interactions into temporal graph snapshots and constructs prediction samples using a sliding-window strategy.

<p align="center">
<img src="./Figures/split+window.png" width="60%">
</p>

The sliding-window strategy separates historical observations from future prediction targets. Historical snapshots are used as model inputs, while future snapshots serve as prediction targets. This design prevents information leakage and enables rigorous evaluation for prospective node importance prediction.

------------------------------------------------------------------------

### Overall Two-Stage Learning Framework

The complete model architecture is illustrated below.

<p align="center">
<img src="./Figures/Two_stage.png" width="90%">
</p>

The framework adopts a two-stage learning paradigm.

Stage I: Categorization-based Coarse Screening

The first stage performs coarse classification to identify potential future top-k important nodes.

Stage II: Regression-based Refined Scoring

The second stage inherits learned representations and fine-tunes partial upstream parameters to predict continuous node importance scores and generate final rankings.

The framework integrates:

-   Spatial graph encoding
-   Temporal dependency modeling
-   Temporal decay mechanism
-   Node micro-evolution modeling
-   Auxiliary graph reconstruction

------------------------------------------------------------------------

## Repository Structure

```text
ProDeL/
│
├── Data/
│   ├── datasets/
│   │   ├── raw/
│   │   │
│   │   └── processed_data/
│   │
│   ├── data_processing.py
│   │
│   ├── dataloader.py
│   │
│   └── temporal_degree_labels.py
│
├── Figures/
│   ├── Two_stage.png
│   │
│   ├── split+window.png
│   │
│   └── results/
│       Experimental results and visualization outputs.
│
├── Model/
│   ├── gnn.py
│   │
│   └── graph.py
│
├── Project/
│   ├── train.py
│   │
│   ├── metrics.py
│   │
│   └── visulation.py
│
├── README.md
│
└── main.py
```

------------------------------------------------------------------------

### Data Processing Pipeline

The preprocessing pipeline mainly contains:

data_processing.py

dataloader.py

The procedure includes:

-   Raw interaction loading
-   Temporal snapshot construction
-   Sliding-window partition
-   Graph structure generation
-   Node feature extraction

------------------------------------------------------------------------

### Model Architecture

The model consists of three main components.

Spatial Encoder

Extracts topology-aware node representations from temporal graph snapshots.

Temporal Encoder

Captures historical dependencies using temporal modeling mechanisms.

Node Importance Prediction

The learned representations are used for:

-   Future top-k node identification
-   Continuous importance score prediction
-   Node ranking generation

------------------------------------------------------------------------

## Training Strategy

The training process follows a two-stage learning paradigm, where the model first performs coarse candidate screening and then refines continuous node importance scores.

| Stage | Task Type | Main Objective | Loss Function | Description |
|-------|-----------|----------------|---------------|-------------|
| Stage I: Categorization-based Coarse Screening | Classification | Identify potential future top-k important nodes from historical temporal graph sequences. | Binary Classification Loss (`L_cls`) | The model learns discriminative representations to distinguish future important nodes from other nodes and provides candidate nodes for subsequent refinement. |
| Stage II: Regression-based Refined Scoring | Regression | Predict continuous future node importance scores and generate final rankings. | Regression Loss (`L_MSE`) + Graph Reconstruction Loss (`L_GAE`) | The model fine-tunes learned representations to estimate importance scores while preserving structural information through auxiliary graph reconstruction. |

---

---

## Code Organization

The implementation is organized into four main components:

1. Data preprocessing  
2. Temporal graph construction  
3. Model architecture  
4. Training and evaluation  


| Component | File | Description |
|-----------|------|-------------|
| Data Preprocessing | `Data/data_processing.py` and `Data/dataloader.py` | Processes raw temporal interaction data, including data cleaning, timestamp processing, temporal snapshot construction, sliding-window generation, and loading processed temporal graph data for model training. |
| Label Generation | `Data/temporal_degree_labels.py` | Generates future node importance labels used for supervised learning and prediction evaluation. |
| Graph Construction | `Model/graph.py` | Constructs temporal graph datasets and organizes sequential graph snapshots for spatio-temporal learning. |
| Spatial-Temporal Model | `Model/gnn.py` | Implements graph neural network modules, including spatial representation learning and temporal dependency modeling. |
| Training Pipeline | `Project/train.py` | Provides the complete training procedure, including model optimization, loss computation, validation, and model selection. |
| Evaluation | `Project/metrics.py` | Implements evaluation metrics for node importance ranking and prediction performance assessment. |
| Visualization | `Project/visulation.py` | Provides visualization utilities for experimental results and model analysis. |
| Experiment Entry | `main.py` | Serves as the main entry point for experiments, including configuration settings, dataset loading, model initialization, and training execution. |

---
------------------------------------------------------------------------

## Requirements

Python >= 3.8

Main dependencies:

torch torch-geometric numpy pandas networkx scikit-learn matplotlib tqdm

------------------------------------------------------------------------

## Running

After preparing datasets and modifying paths:

python main.py

------------------------------------------------------------------------

License

This repository is released for academic research purposes.



