# Schizophrenia ERP Classifier

> **Clinical ML pipeline for schizophrenia detection via EEG Event-Related Potentials**  
> Built during research internship at IEEE Engineering in Medicine and Biology Society (EMBS)

[![Live Demo](https://img.shields.io/badge/🤗%20HuggingFace-Live%20Demo-blue)](https://huggingface.co/spaces/Lord-Bane/schizophrenia-erp-classifier)
[![Python](https://img.shields.io/badge/Python-3.9+-green)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange)](https://tensorflow.org)

---

## The Problem

Schizophrenia affects approximately 24 million people globally. Early and accurate diagnosis remains a clinical challenge — traditional methods rely heavily on subjective assessment, which is time-consuming and prone to inconsistency.

EEG-based biomarkers, specifically Event-Related Potentials (ERPs), offer an objective, non-invasive signal that can distinguish schizophrenia patients from healthy controls. This project builds a complete, deployable pipeline that makes that signal actionable.

---

## What This Does

A full end-to-end clinical ML system — from raw EEG data to live web inference:

```
Raw EEG Data → Feature Engineering → Model Training → Deployment → Live Inference
```

Given a CSV of ERP features, the system predicts whether the EEG pattern is consistent with schizophrenia or a healthy control — with confidence scores per trial.

---

## Dataset & Clinical Context

- **Source:** Public Kaggle EEG dataset
- **Scale:** 4,000+ ERP samples across **81 subjects** and **9 electrode sites**
- **Signal:** Event-Related Potentials (ERPs) — brain responses time-locked to stimuli
- **Key biomarkers extracted:**
  - **N100 amplitude** — early auditory processing component (~100ms post-stimulus)
  - **P200 amplitude** — later cognitive processing component (~200ms post-stimulus)

These components are established clinical markers — reduced P200 amplitude in particular is consistently associated with schizophrenia in the neuroscience literature.

---

## Technical Architecture

### 1. Preprocessing Pipeline
- Trial-level ERP feature extraction across 9 electrode sites
- StandardScaler normalization (artifact saved for reproducible inference)
- Sequence padding for variable-length inputs
- Config JSON serialization for deployment consistency

### 2. Model
- **Architecture:** Keras sequential classifier
- **Benchmarked against:** 5 classifiers (Logistic Regression, Random Forest, SVM, XGBoost, Keras DNN)
- **Saved artifacts:** trained model + scaler + config JSON → ensures inference matches training exactly

### 3. Deployment
- **Platform:** Hugging Face Spaces (Gradio)
- **Interface:** CSV upload → automatic feature reconstruction → scaling → sequence padding → prediction → tabular output with confidence scores
- **Live and publicly accessible** — no setup required

---

## Live Demo

🔗 **[Try it here → huggingface.co/spaces/Lord-Bane/schizophrenia-erp-classifier](https://huggingface.co/spaces/Lord-Bane/schizophrenia-erp-classifier)**

Upload a CSV of ERP features and get trial-level predictions instantly.

**Input format:** CSV with N100 and P200 amplitude features per trial per electrode  
**Output:** Per-trial classification (Schizophrenia / Healthy Control) with confidence score

---

## Stack

| Component | Technology |
|-----------|-----------|
| Data processing | Python, Pandas, NumPy, SciPy |
| ML & modelling | TensorFlow/Keras, Scikit-learn |
| Deployment | Gradio, Hugging Face Spaces |
| Reproducibility | Joblib (scaler), JSON (config) |

---

## Project Context

Built during a **Research Internship at IEEE EMBS (June–July 2025)** as part of an applied ML research initiative in computational neuroscience.

This project demonstrates:
- End-to-end ownership of a clinical ML pipeline — from raw signal to live deployment
- Domain-specific feature engineering (ERP biomarkers) rather than generic ML application
- Deployment-first thinking — reproducible artifacts, not just a notebook

---

## Disclaimer

This tool is intended for **research purposes only** and is not a clinical diagnostic instrument. All predictions should be interpreted in the context of comprehensive clinical assessment by qualified professionals.
