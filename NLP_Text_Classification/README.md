# 🧠 NLP Text Classification with Transformers: RoBERTa & XLNet

---

## 📖 Introduction

This project demonstrates a complete **end-to-end pipeline** for loading, fine-tuning, and evaluating transformer-based models for **text classification** tasks using NLP.

Two powerful transformer architectures are explored to classify human emotions from text, leveraging the Hugging Face ecosystem:

1. **RoBERTa** — *A Robustly Optimized BERT Pretraining Approach*
2. **XLNet** — *Generalized Autoregressive Pretraining for Language Understanding*

The project also analyzes and compares the architectures, training strategies, and optimization techniques of both models in the context of **emotion classification**.

---

## 📊 Dataset

**[Hugging Face Emotion Dataset](https://huggingface.co/datasets/emotion)**

A collection of English-language Twitter messages labeled with six basic emotions:

* Anger
* Fear
* Joy
* Love
* Sadness
* Surprise

Dataset split:

| Split      | Rows   | Columns |
| ---------- | ------ | ------- |
| Train      | 16,000 | 2       |
| Validation | 2,000  | 2       |
| Test       | 2,000  | 2       |

---

## ⚙️ Project Workflow

The pipeline builds both **RoBERTa** and **XLNet** models following these steps:

1. Data Exploration & Analysis
2. Data Pre-processing
3. Model Creation (RoBERTa / XLNet)
4. Model Compilation
5. Model Training with Fine-Tuning
6. Model Evaluation & Validation
7. Performance Metrics & Analysis
8. Saving Final Optimized Model
9. Testing Final Model on Unseen Data

---

## 🛠️ Tools & Technologies

* **Python**
* **NumPy**
* **Pandas**
* **ktrain**
* **transformers** (by Hugging Face)
* **TensorFlow**
* **scikit-learn**
* **Matplotlib**

---
