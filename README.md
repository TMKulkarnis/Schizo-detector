# 🧠 Schizophrenia ERP Classifier

> **Clinical EEG sequence classification using a Transformer + Bidirectional GRU hybrid**  
> Research internship project — IEEE Engineering in Medicine and Biology Society (EMBS)

[![Live Demo](https://img.shields.io/badge/🤗%20HuggingFace-Live%20Demo-blue)](https://huggingface.co/spaces/Lord-Bane/schizophrenia-erp-classifier)
[![Dataset](https://img.shields.io/badge/Kaggle-Button--Tone--SZ-20BEFF)](https://www.kaggle.com/datasets/broach/button-tone-sz)
[![Python](https://img.shields.io/badge/Python-3.9+-green)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow%2FKeras-2.x-orange)](https://tensorflow.org)

---

## Clinical Background

This project is built on a foundational finding by Ford et al. (2013): patients with schizophrenia show **reduced suppression of auditory cortical responses** when they self-deliver tones via button press, compared to healthy controls. This reflects dysfunction in the brain's *efference copy* and *corollary discharge* mechanisms — systems that allow the brain to predict and dampen responses to self-generated sensations.

In healthy individuals, pressing a button to deliver a tone triggers a motor signal (efference copy) that prepares auditory cortex to expect the sound — resulting in suppression of the N1 ERP component. In schizophrenia patients, this predictive mechanism is impaired.

This classifier learns to detect those differences from trial-level ERP features.

> **Reference:** Ford JM, Palzes VA, Roach BJ, Mathalon DH. *Did I Do That? Abnormal Predictive Processes in Schizophrenia when Button Pressing to Deliver a Tone.* Schizophrenia Bulletin, 2013. doi:10.1093/schbul/sbt072

---

## Dataset

**Source:** [Button-Tone-SZ — Kaggle](https://www.kaggle.com/datasets/broach/button-tone-sz)  
**Originally collected by:** Ford et al., UCSF / San Francisco VA Medical Center

| Property | Detail |
|----------|--------|
| Subjects | 81 (26 SZ patients + 22 HC in original study; expanded Kaggle release) |
| Electrode sites | 9 |
| ERP components | N100, P200, P300, N400 |
| Trials | 4,000+ |
| Task | Button-press to self-deliver 1000 Hz tone |

**Key biomarkers:**
- **N100** (~100ms) — early auditory processing; suppressed in healthy controls during self-delivery
- **P200** (~200ms) — later cognitive processing component
- **P300** (~300ms) — attention and memory updating
- **N400** (~400ms) — semantic processing

---

## Architecture

Each subject's ERP trials are treated as a **temporal sequence** — capturing how neural responses evolve across trials rather than treating each trial in isolation.

```
ERP Features (N100, P200, P300, N400 per electrode per trial)
      │
      ▼
Feature Engineering
  ├── Trial-to-trial delta (within-subject change)
  └── Rolling means (windows of 2 and 3 trials)
      │
      ▼
MinMaxScaler → Sequence Padding (mask_value=0.0)
      │
      ▼
┌─────────────────────────────────────┐
│   Positional Embedding              │
│   (token projection + pos. embed)   │
└─────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────┐
│   4 × Transformer Blocks            │
│   Multi-Head Attention (8 heads)    │
│   embed_dim=64, ff_dim=128          │
│   LayerNorm + Dropout (0.2)         │
└─────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────┐
│   Bidirectional GRU (64 units)      │
└─────────────────────────────────────┘
      │
      ▼
TimeDistributed Dense → Softmax
(per-trial prediction: SZ / HC)
```

**Optional demographic fusion:** if `demographic.csv` is provided, subject-level vectors are projected to `embed_dim` and fused before the GRU layer.

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Transformer + BiGRU hybrid | Attention captures cross-trial feature relationships; GRU models temporal order across the session |
| GroupKFold (5-fold, subject-level) | Prevents data leakage — entire subjects are held out, not individual trials |
| Masking layer | Handles variable-length sequences without distorting attention weights |
| Label smoothing (0.1) | Reduces overconfidence on noisy clinical labels |
| Saved scaler + model artifacts | Ensures inference scaling exactly matches training distribution |

---

## Training Config

```python
EMBED_DIM = 64        # transformer embedding dimension
NUM_HEADS = 8         # multi-head attention heads
FF_DIM = 128          # feedforward network dimension
TRANSFORMER_BLOCKS = 4
GRU_UNITS = 64
DROPOUT_RATE = 0.2
BATCH_SIZE = 16
EPOCHS = 100
PATIENCE_ES = 12      # early stopping patience
LABEL_SMOOTHING = 0.1
N_SPLITS = 5          # GroupKFold folds
```

---

## Live Demo

🔗 **[Try it → huggingface.co/spaces/Lord-Bane/schizophrenia-erp-classifier](https://huggingface.co/spaces/Lord-Bane/schizophrenia-erp-classifier)**

Upload a CSV of ERP features → get **per-trial predictions** (SZ / HC) with confidence scores.

**Input:** CSV with N100, P200, P300, N400 amplitude features per trial per electrode  
**Output:** Trial-level classification with confidence score

---

## Stack

| Component | Technology |
|-----------|-----------|
| Model | TensorFlow/Keras (custom Transformer + BiGRU) |
| Data processing | Pandas, NumPy, SciPy |
| ML utilities | Scikit-learn (GroupKFold, MinMaxScaler, OneHotEncoder) |
| Deployment | Gradio on Hugging Face Spaces |

---

## Disclaimer

This tool is for **research purposes only**. It is not a validated clinical diagnostic instrument. All outputs must be interpreted by qualified clinical professionals in the context of comprehensive patient assessment.

The dataset used is publicly available and was originally collected under IRB approval at UCSF and San Francisco VA Medical Center.

---

*IEEE EMBS Research Internship | MIT ADT University, Pune | 2025*
