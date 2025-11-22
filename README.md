Schizophrenia Detection via ERP Data (Hybrid Transformer-GRU)

This repository contains a deep learning pipeline designed to classify Schizophrenia subjects versus Healthy Controls using Event-Related Potential (ERP) data. The model leverages a hybrid architecture combining Transformer blocks for attention-based feature extraction and Bidirectional GRUs for temporal sequence modeling, along with an optional fusion mechanism for demographic data.

Dataset

The model is designed to train on the Button Tone SZ dataset available on Kaggle.

Source: Kaggle: Button Tone SZ Dataset: https://www.kaggle.com/datasets/broach/button-tone-sz/

Required Files

To run this pipeline, you must download the dataset and place the following CSV files in your working directory (or update the paths in main.py):
ERPdata.csv (or mergedTrialData.csv): Contains the time-series ERP features.
demographic.csv (Optional): Contains subject-level demographic information for fusion.
Note: The script automatically attempts to load data from standard paths including /mnt/data/ and the local directory.

Architecture Overview

The model (main.py) implements a sophisticated sequence processing pipeline:

Data Preprocessing:
Feature Engineering: Automatically extracts standard ERP components (N100, P200, P300, N400) and calculates delta features and rolling means.
Imputation: Handles missing values using subject-wise and global means.
Normalization: Applies MinMax scaling per fold.

Hybrid Model Structure:
Positional Embedding: Adds temporal context to the ERP sequence.
Transformer Blocks: 4 layers of Multi-Head Attention (8 heads) to capture long-range dependencies in the ERP signal.
Demographic Fusion (Optional): If demographic data is present, it is embedded, tiled, and concatenated with the time-series features.
Bidirectional GRU: Processes the attention-enriched sequence to capture sequential dynamics.
Classification Head: A TimeDistributed Dense layer with Softmax activation for trial-level classification.

Training Strategy:
Group K-Fold Cross-Validation: Uses 5 splits, ensuring that subjects are strictly separated between training and validation sets to prevent data leakage.
Optimization: Adam optimizer with Label Smoothing (0.1) to reduce overfitting.
Callbacks: Early Stopping, ReduceLROnPlateau, and Model Checkpointing.

Installation & Requirements
Ensure you have Python 3.8+ installed. Install the required dependencies using pip:
pip install numpy pandas scikit-learn tensorflow

Dependencies
TensorFlow: Deep learning backend.
Pandas: Data manipulation and CSV loading.
NumPy: Numerical operations.
Scikit-Learn: Cross-validation (GroupKFold), preprocessing (MinMaxScaler, OneHotEncoder), and metrics (classification_report).

Usage
Prepare Data: Download the dataset from Kaggle and unzip it.
Place Files: Ensure ERPdata.csv (or mergedTrialData.csv) is in the same directory as main.py.

Run Training:
python main.py


Output
The script will output:
Logs for data loading and feature engineering.
Fold-by-fold training progress (Accuracy/Loss).
A classification report for each fold.
Final Summary: Mean accuracy and Macro F1 score across all 5 folds.
Artifact: The best model is saved as final_schizo_detector.keras.

Key Configuration (Hyperparameters)
You can modify the configurations at the top of main.py to tune performance:

N_SPLITS (Default: 5)
Number of cross-validation folds.

EMBED_DIM (Default: 64)
Dimension of the embedding layer.

NUM_HEADS (Default: 8)
Number of attention heads in Transformer.

TRANSFORMER_BLOCKS (Default: 4)
Depth of the Transformer stack.

GRU_UNITS (Default: 64)
Units in the GRU layer.

EPOCHS (Default: 100)
Maximum training epochs (early stopping enabled).

FEATURE_PATTERNS (Default: ['n100', 'p200', 'p300', 'n400'])
Substrings used to filter relevant ERP columns.

License
This code is provided for educational and research purposes. Please cite the original dataset authors when publishing results.
