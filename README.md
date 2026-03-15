This repository contains the code and data for our study on evaluating and enhancing the reliability of Large Language Models (LLMs) in data annotation tasks. We systematically compare multiple approaches across different LLMs and provide a revised boilerplate for requirement classification.

📋 Overview
This project investigates:

The advantages and limitations of LLMs in data annotation

Performance evaluation of various Chinese LLMs

Methods for enhancing and validating LLM output reliability

A comparative analysis of six different approaches to requirement classification

📊 Dataset
The dataset file dataset.xlsx contains annotated requirements with two columns: requirement (the original requirement text to be classified) and label (the ground truth category for the requirement). The dataset includes both structural and interactive requirement types, annotated following our revised boilerplate framework.

🧠 Implemented Methods
This repository implements and compares six approaches:

BERT — Baseline BERT-based classification model

CDA-MRCV (Cross-Domain Adaptation with Multi-Round Cross-Validation) — Adapts across domains with iterative validation

MG-CCE (Multi-Granularity Cross-Contextual Embeddings) — Captures requirements at multiple granularity levels

PROGRESS (PROgressive Granularity Enhancement for Semantic Segmentation) — Progressively enhances semantic understanding

RCPP (Reliability-Consistent Prediction Propagation) — Propagates predictions while maintaining consistency

SAIP (Self-Attention with Iterative Prompting) — Combines self-attention mechanisms with iterative prompting

Each method's implementation is organized in its respective directory under methods/.

🚀 Setup and Installation
Prerequisites
Python 3.8 or higher

pip package manager

Installation
Clone the repository:

bash

git clone https://github.com/yourusername/llm-annotation-reliability.git
cd llm-annotation-reliability
Install required packages:

bash

pip install -r requirements.txt
Configure environment variables

Create a .env file in the root directory with the following variables:

API_BASE_URL — Base URL for LLM API access
API_KEY — API authentication key
API_SECRET — API secret for authentication
MODEL_ENDPOINT — Specific model endpoint to use
REQUEST_TIMEOUT — (Optional) API request timeout in seconds, defaults to 30
MAX_RETRIES — (Optional) Maximum retry attempts for failed requests, defaults to 3

⚠️ Important: Never commit your .env file to version control. It has been added to .gitignore for your safety.

💻 Usage
Running experiments

To run a specific method:

bash
python methods/bert/train.py
python methods/cda-mrcv/train.py
Evaluating results
Results will be saved in the results/ directory, including per-method performance metrics, cross-method comparison tables, and visualizations of key findings.

📈 Results
Our experiments demonstrate detailed performance comparison across all six methods, analysis of LLM reliability in annotation tasks, key findings and identified research gaps, and validation metrics for annotation quality. See our paper for complete results and analysis.

📁 Repository Structure
Root Directory Files

  dataset.xlsx — Annotated requirement dataset

  .env — Environment configuration (create this)

  .gitignore — Git ignore rules

  requirements.txt — Python dependencies

  run_experiments.py — Main experiment runner

  README.md — This file

Methods Directory

  methods/bert/ — BERT implementation

  methods/cda-mrcv/ — CDA-MRCV implementation

  methods/mg-cce/ — MG-CCE implementation

  methods/progress/ — PROGRESS implementation

  methods/rcpp/ — RCPP implementation

  methods/saip/ — SAIP implementation

Other Directories

  results/ — Experiment results

  utils/ — Utility functions

  config/ — Configuration files


📝 Citation
If you use this code or dataset in your research, please cite our paper:
He, J., Osman, M. H., Hassan, S., & Yap, N. K. (2026). PROGRESS: A Large Language Model-Based Chinese Software Requirements Annotation System. Knowledge-Based System.
