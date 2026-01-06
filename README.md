# Glider AOA Inference + TDA

This repository provides support code for:
- AOA (angle-of-attack) inference via Actor-Critic/DDPG (TensorFlow2)
- TDA (transducer data assimilaiton)

The code is designed to reproduce the *methodological pipeline* described in the paper.
A public demo parameter set and synthetic data are provided to ensure that all scripts are runnable.

## Data Description
This repository does **not** contain the original sea-trial datasets used in the paper due to
confidentiality and project restrictions.

To ensure reproducibility and transparency of the proposed methods, a **synthetic demo dataset**
(`data/demo.csv`) is provided. The demo data are generated to preserve the statistical characteristics
(e.g., scale, range, and temporal smoothness) of the measured signals, while **not corresponding to any
real sea-trial trajectory**.

The demo dataset allows users to:
- Execute all scripts in this repository without access to private data
- Verify the correctness of the algorithmic pipeline
- Reproduce the methodological workflow described in the paper

The original experimental data were used only for offline validation and figure generation and are
available from the authors upon reasonable request and subject to confidentiality approval.
