# ProDeL-Prospective-Decoupld-Leaming-for-Future-Node-Imporance-Scoring-in-Dynamic-Networks
ProDeL: Prospective Decoupled Learning for Future Node Importance Scoring in Dynamic Networks

This repository provides the official implementation code for ProDeL, a two-stage prospective decoupled learning framework for predicting future
node importance in dynamic networks.

The code includes the complete pipeline for temporal network preprocessing, sliding-window construction, graph representation learning, two-stage training, evaluation, and visualization.

------------------------------------------------------------------------

Framework Overview

1. Temporal Network Construction and Sliding-Window Partition

The temporal network preprocessing procedure converts continuous interactions into temporal graph snapshots and constructs prediction samples using a sliding-window strategy.

The figure is directly shown below:
<img width="7748" height="5612" alt="split+window_01" src="https://github.com/user-attachments/assets/3e6f55b5-fadf-4be1-9b42-59b42111b226" />

The sliding-window strategy separates historical observations from future prediction targets. Historical snapshots are used as model inputs, while future snapshots serve as prediction targets. This design prevents information leakage and enables rigorous evaluation for prospective node importance prediction.

------------------------------------------------------------------------

2. Overall Two-Stage Learning Framework

The complete model architecture is illustrated below.

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


