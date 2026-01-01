# Misconception Miner: A Hybrid Retrieval Approach for Mapping Mathematical Errors

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-orange)
![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Transformers-yellow)
![License](https://img.shields.io/badge/License-MIT-green)

**Authors:** Talha K., M. Beşir A.

## 📖 Overview
In education, knowing a student's answer is wrong is not enough; we need to know *why*. 
**Misconception Miner** is an NLP-based retrieval system designed to map incorrect student answers (distractors) to specific misconceptions. 

For example, if a student answers `2/5 + 3/7 = 5/12`, the system identifies the underlying misconception: *"Adds numerators and denominators when adding fractions."*

This project was developed in the context of the **Eedi - Mining Misconceptions in Mathematics** Kaggle competition.

## 🚀 Key Features & Pipeline
We implemented a multi-stage **Retrieval-Augmented Generation (RAG)** funnel designed to handle over 2,500 distinct misconception classes.

### 1. Data Augmentation (Generative AI)
* **Problem:** Severe class imbalance (long-tail distribution).
* **Solution:** Fine-tuned **`google/flan-t5-large`** with Few-Shot prompting to generate synthetic student errors for rare misconceptions, enriching the training space.

### 2. Hybrid Retrieval (Stage 1)
* **Lexical:** TF-IDF to capture exact mathematical terminology (e.g., "hypotenuse", "coefficient").
* **Semantic:** Dense retrieval using Bi-Encoders.
* **Finding:** We compared 12 models (including BGE-Large, GTE-Large). Surprisingly, the older **`all-distilroberta-v1`** outperformed modern large models in this specific domain with a **Recall@200 of 0.801**.

### 3. Cross-Encoder Re-Ranking (Stage 2)
* Top-200 candidates from Stage 1 are re-ranked using **`ms-marco-MiniLM-L-6-v2`**.
* This stage captures deep token-level interactions between the Question/Answer pair and the Misconception text.

## 📊 Results

Our experiments highlighted a trade-off between retrieval recall and ranking precision.

| Metric | Hybrid Retrieval Only | With Cross-Encoder (L-6) | Improvement |
| :--- | :--- | :--- | :--- |
| **Recall@1** | 0.004 | **0.074** | +17x |
| **Recall@5** | 0.012 | **0.177** | +14x |
| **Recall@25** | 0.120 | **0.470** | +3.9x |
| **Recall@200**| 0.801 | 0.801 | (Ceiling) |

**Key Insight:** The Cross-Encoder dramatically improves top-ranking performance, but the system is limited by the Stage 1 retrieval ceiling (80.1%).

## 🛠️ Installation & Usage

### Prerequisites
* Python 3.10+
* Jupyter Notebook / Google Colab (Recommended for GPU support)

### Dependencies
```bash
pip install torch transformers sentence-transformers pandas scikit-learn seaborn matplotlib
