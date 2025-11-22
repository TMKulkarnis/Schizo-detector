import os
import numpy as np
import pandas as pd
import warnings
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, accuracy_score
from tensorflow import keras
from tensorflow.keras import layers, Model, Sequential
import tensorflow as tf

# -------------------------
# Config / Hyperparams
# -------------------------
RANDOM_STATE = 42
N_SPLITS = 5
EMBED_DIM = 64
NUM_HEADS = 8
FF_DIM = 128
TRANSFORMER_BLOCKS = 4
GRU_UNITS = 64
DROPOUT_RATE = 0.2
BATCH_SIZE = 16
EPOCHS = 100
PATIENCE_ES = 12
LABEL_SMOOTHING = 0.1
MAX_ROLL_WINDOWS = [2, 3]  # rolling windows to create features
FEATURE_PATTERNS = ['n100', 'p200', 'p300', 'n400']  # try to include these
DEVICE = "/GPU:0" if tf.config.list_physical_devices("GPU") else "/CPU:0"

np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)
warnings.filterwarnings('ignore')

# -------------------------
# Load data (tries multiple paths)
# -------------------------
print("Loading data...")
paths = [
    "/mnt/data/ERPdata.csv",
    "/mnt/data/mergedTrialData.csv",
    "mergedTrialData.csv",
    "ERPdata.csv",
    "mergedTrialData.csv"
]
df = None
for p in paths:
    if os.path.exists(p):
        df = pd.read_csv(p)
        print(f"Loaded {p}")
        break
if df is None:
    raise FileNotFoundError("No ERP data file found. Put 'mergedTrialData.csv' or 'ERPdata.csv' in the working dir or /mnt/data.")

# normalize columns
df.columns = df.columns.str.strip().str.lower()

# optional demographics
demog_path = "/mnt/data/demographic.csv"
demog_df = None
if os.path.exists(demog_path):
    demog_df = pd.read_csv(demog_path)
    demog_df.columns = demog_df.columns.str.strip().str.lower()
    print(f"Loaded demographics from {demog_path}")
else:
    print("No demographics file found at /mnt/data/demographic.csv (optional).")

# -------------------------
# Basic checks and ids
# -------------------------
required_id_cols = ['subject', 'condition', 'trial']
for c in required_id_cols:
    if c not in df.columns:
        raise ValueError(f"Required column '{c}' not found in data.")

# -------------------------
# Feature selection & engineering
# -------------------------
print("Selecting and engineering features...")

# pick columns matching patterns
feature_cols = [c for c in df.columns if any(p in c for p in FEATURE_PATTERNS)]
# if none found, fall back to numeric non-id cols
if len(feature_cols) == 0:
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    feature_cols = [c for c in numeric_cols if c not in required_id_cols]
    print("No ERP patterns found; falling back to numeric features:", feature_cols)
else:
    print("Found ERP feature columns (sample):", feature_cols[:10])

# keep only id + features (we'll add engineered features to df later)
df = df[required_id_cols + feature_cols].copy()

# Create delta features (trial-to-trial differences) per subject
for col in feature_cols:
    df[f"{col}_delta"] = df.groupby('subject')[col].diff().fillna(0.0)

# Rolling means for short windows
for w in MAX_ROLL_WINDOWS:
    for col in feature_cols:
        df[f"{col}_r{w}"] = df.groupby('subject')[col].rolling(window=w, min_periods=1).mean().reset_index(level=0, drop=True)

# Update list of features
engineered_cols = [c for c in df.columns if c not in required_id_cols]
feature_cols = engineered_cols
print(f"Total features after engineering: {len(feature_cols)}")

# -------------------------
# Imputation: subject-wise mean then global
# -------------------------
print("Imputing missing values subject-wise (then global)...")
df[feature_cols] = df.groupby('subject')[feature_cols].transform(lambda x: x.fillna(x.mean()))
df[feature_cols] = df[feature_cols].fillna(df[feature_cols].mean())

# -------------------------
# Global sequence length (pad length)
# -------------------------
# Create per-subject sequences to compute global max length
seq_lengths = df.groupby('subject').size().values
max_sequence_length = int(seq_lengths.max())
print("Max sequence length (global padding) =", max_sequence_length)

# -------------------------
# Optional demographic fusion
# -------------------------
use_demog = False
demog_embedding_map = {}
if demog_df is not None:
    # choose some demographic numeric/categorical cols to embed
    # merge on 'subject'
    if 'subject' in demog_df.columns:
        demog_df = demog_df.set_index('subject')
        common_subjects = set(df['subject'].unique()).intersection(demog_df.index.astype(str))
        if len(common_subjects) > 0:
            use_demog = True
            print("Fusing demographics for subjects:", len(common_subjects))
            # simple numeric columns (drop subject index)
            demog_numeric = demog_df.select_dtypes(include=[np.number])
            # fill na
            demog_numeric = demog_numeric.fillna(demog_numeric.mean())
            # map subject -> vector
            demog_embedding_map = {str(idx): demog_numeric.loc[idx].values for idx in demog_numeric.index}
        else:
            print("No matching subject ids between demographics and ERP data. Skipping demog fusion.")
    else:
        print("Demographics file lacks 'subject' column — skipping demographic fusion.")

# -------------------------
# Helper: create sequences (per-subject)
# -------------------------
def create_sequences_from_df(df_full, feature_cols):
    X_seqs = []
    y_seqs = []
    subject_ids = []
    for subject, group in df_full.groupby('subject'):
        g = group.sort_values('trial')
        X_seqs.append(g[feature_cols].values.astype(np.float32))
        y_seqs.append((g['condition'].values - 1).astype(int))  # zero-index classes
        subject_ids.append(str(subject))
    return X_seqs, y_seqs, subject_ids

# -------------------------
# Prepare global sequences and encoder
# -------------------------
X_all_seqs, y_all_seqs, subject_list = create_sequences_from_df(df, feature_cols)
all_labels = np.unique([lab for seq in y_all_seqs for lab in seq])

# sklearn OneHotEncoder compatibility: older sklearn used 'sparse=False', newer uses 'sparse_output'
# try to construct in a compatible way
try:
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
except TypeError:
    encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')

encoder.fit(all_labels.reshape(-1, 1))
num_classes = len(all_labels)
num_features = len(feature_cols)

print(f"Detected {len(subject_list)} subjects, {num_features} features, {num_classes} classes.")

# -------------------------
# Model building utilities
# -------------------------
class PositionalEmbedding(layers.Layer):
    def __init__(self, sequence_length, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.token_projection = layers.Dense(embed_dim)
        self.position_embeddings = layers.Embedding(input_dim=sequence_length, output_dim=embed_dim)
        self.sequence_length = sequence_length
        self.embed_dim = embed_dim

    def call(self, inputs):
        # inputs: (batch, seq_len, feat_dim)
        seq_len = tf.shape(inputs)[1]
        positions = tf.range(start=0, limit=seq_len, delta=1)
        x = self.token_projection(inputs)  # (batch, seq, embed)
        pos = self.position_embeddings(positions)  # (seq, embed)
        return x + pos

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"sequence_length": self.sequence_length, "embed_dim": self.embed_dim})
        return cfg

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = Sequential([layers.Dense(ff_dim, activation='relu'), layers.Dense(embed_dim)])
        self.ln1 = layers.LayerNormalization(epsilon=1e-6)
        self.ln2 = layers.LayerNormalization(epsilon=1e-6)
        self.drop1 = layers.Dropout(rate)
        self.drop2 = layers.Dropout(rate)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate

    def call(self, inputs, training=None):
        attn = self.att(inputs, inputs, training=training)
        attn = self.drop1(attn, training=training)
        out1 = self.ln1(inputs + attn)
        ffn = self.ffn(out1)
        ffn = self.drop2(ffn, training=training)
        return self.ln2(out1 + ffn)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"embed_dim": self.embed_dim, "num_heads": self.num_heads, "ff_dim": self.ff_dim, "rate": self.rate})
        return cfg

def build_model(max_seq_len, num_features, num_classes, demog_dim=None, initial_lr=1e-3):
    inp = layers.Input(shape=(max_seq_len, num_features), name='inputs')
    mask_layer = layers.Masking(mask_value=0.0)(inp)
    x = layers.LayerNormalization()(mask_layer)
    x = PositionalEmbedding(max_seq_len, EMBED_DIM)(x)
    x = layers.Dropout(DROPOUT_RATE)(x)

    for _ in range(TRANSFORMER_BLOCKS):
        x = TransformerBlock(EMBED_DIM, NUM_HEADS, FF_DIM, rate=DROPOUT_RATE)(x)

    # optionally fuse demographic embedding (broadcasted)
    if demog_dim is not None:
        dem_inp = layers.Input(shape=(demog_dim,), name='demog_input')
        dem_proj = layers.Dense(EMBED_DIM)(dem_inp)  # project to embed_dim
        # tile across time
        dem_tiled = layers.RepeatVector(max_seq_len)(dem_proj)
        x = layers.Concatenate()([x, dem_tiled])
        # project back to embed_dim
        x = layers.Dense(EMBED_DIM)(x)
        # then pass to GRU
        x = layers.Bidirectional(layers.GRU(GRU_UNITS, return_sequences=True))(x)
        out = layers.TimeDistributed(layers.Dense(num_classes, activation='softmax'))(x)
        model = Model(inputs=[inp, dem_inp], outputs=out)
    else:
        x = layers.Bidirectional(layers.GRU(GRU_UNITS, return_sequences=True))(x)
        out = layers.TimeDistributed(layers.Dense(num_classes, activation='softmax'))(x)
        model = Model(inputs=inp, outputs=out)

    # compile with label smoothing
    loss = keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING)

    # IMPORTANT FIX:
    # Use a *float* learning rate with the optimizer so callbacks like ReduceLROnPlateau can modify it.
    initial_lr = 1e-3
    opt = keras.optimizers.Adam(learning_rate=initial_lr)

    model.compile(optimizer=opt, loss=loss, metrics=['accuracy'])
    return model

# -------------------------
# Cross-validation training
# -------------------------
gkf = GroupKFold(n_splits=N_SPLITS)
subjects = np.array(subject_list)

fold = 0
fold_reports = []
fold_accuracies = []
best_val_acc = -np.inf
best_model_path = "final_schizo_detector.keras"

for train_idx, test_idx in gkf.split(X_all_seqs, groups=subjects):
    fold += 1
    print(f"\n\n=== Fold {fold}/{N_SPLITS} ===")
    # split sequences
    X_train_seqs = [X_all_seqs[i] for i in train_idx]
    y_train_seqs = [y_all_seqs[i] for i in train_idx]
    subj_train = [subjects[i] for i in train_idx]
    X_test_seqs = [X_all_seqs[i] for i in test_idx]
    y_test_seqs = [y_all_seqs[i] for i in test_idx]
    subj_test = [subjects[i] for i in test_idx]

    # fit scaler on flattened trials of training subjects
    flat_train = np.vstack(X_train_seqs)
    scaler = MinMaxScaler()
    scaler.fit(flat_train)
    X_train_scaled = [scaler.transform(s) for s in X_train_seqs]
    X_test_scaled = [scaler.transform(s) for s in X_test_seqs]

    # pad sequences to global max_sequence_length
    def pad_seq_list(seq_list, pad_len, pad_val=0.0):
        out = np.zeros((len(seq_list), pad_len, seq_list[0].shape[1]), dtype=np.float32)
        for i, s in enumerate(seq_list):
            out[i, :s.shape[0], :] = s
        return out

    X_train = pad_seq_list(X_train_scaled, max_sequence_length)
    X_test = pad_seq_list(X_test_scaled, max_sequence_length)

    # pad labels (use -1 for padded)
    def pad_y_list(y_list, pad_len, pad_val=-1):
        out = np.full((len(y_list), pad_len), pad_val, dtype=int)
        for i, y in enumerate(y_list):
            out[i, :len(y)] = y
        return out

    y_train_padded = pad_y_list(y_train_seqs, max_sequence_length)
    y_test_padded = pad_y_list(y_test_seqs, max_sequence_length)

    # one-hot encode labels; encoder expects column vector
    y_train_onehot = encoder.transform(y_train_padded.reshape(-1, 1)).reshape(y_train_padded.shape[0], max_sequence_length, -1)
    y_test_onehot = encoder.transform(y_test_padded.reshape(-1, 1)).reshape(y_test_padded.shape[0], max_sequence_length, -1)

    # optionally prepare demographic vectors per subject (averaged if needed)
    demog_dim = None
    if use_demog:
        # for train and test, create per-subject demog vector (same across sequence)
        def build_demog_array(subj_list):
            arr = []
            for s in subj_list:
                sval = str(s)
                vec = demog_embedding_map.get(sval)
                if vec is None:
                    # fallback zeros
                    vec = np.zeros(len(next(iter(demog_embedding_map.values()))))
                arr.append(vec.astype(np.float32))
            return np.stack(arr, axis=0)
        X_train_dem = build_demog_array(subj_train)
        X_test_dem = build_demog_array(subj_test)
        demog_dim = X_train_dem.shape[1]
    else:
        X_train_dem = None
        X_test_dem = None

    # build model
    with tf.device(DEVICE):
        if demog_dim is not None:
            model = build_model(max_sequence_length, num_features + 0, num_classes, demog_dim=demog_dim)
        else:
            model = build_model(max_sequence_length, num_features, num_classes, demog_dim=None)
    model.summary()

    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=PATIENCE_ES, restore_best_weights=True, verbose=1),
        # ReduceLROnPlateau requires the optimizer.learning_rate to be a settable tensor/variable.
        # That is why optimizer was created with a float lr above.
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=4, verbose=1),
        keras.callbacks.ModelCheckpoint(f"fold_{fold}_best.keras", save_best_only=True, monitor='val_loss', verbose=1)
    ]

    # fit
    if demog_dim is not None:
        history = model.fit([X_train, X_train_dem], y_train_onehot,
                            validation_data=([X_test, X_test_dem], y_test_onehot),
                            epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=callbacks, verbose=2)
    else:
        history = model.fit(X_train, y_train_onehot,
                            validation_data=(X_test, y_test_onehot),
                            epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=callbacks, verbose=2)

    # evaluate
    if demog_dim is not None:
        val_loss, val_acc = model.evaluate([X_test, X_test_dem], y_test_onehot, verbose=0)
        y_pred_probs = model.predict([X_test, X_test_dem])
    else:
        val_loss, val_acc = model.evaluate(X_test, y_test_onehot, verbose=0)
        y_pred_probs = model.predict(X_test)

    print(f"Fold {fold} val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")
    fold_accuracies.append(val_acc)

    # aggregate predictions onto unpadded sequences for classification report
    y_pred_flat = []
    y_true_flat = []
    for i in range(len(y_test_seqs)):
        true_len = len(y_test_seqs[i])
        preds = y_pred_probs[i, :true_len, :]
        y_pred_flat.extend(np.argmax(preds, axis=1))
        y_true_flat.extend(y_test_seqs[i].tolist())

    report = classification_report(y_true_flat, y_pred_flat, output_dict=True)
    fold_reports.append(report)
    print("Classification report (fold):")
    print(classification_report(y_true_flat, y_pred_flat))

    # save best model globally
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        try:
            model.save(best_model_path)
            print("Saved new best model to", best_model_path)
        except Exception as e:
            print("Could not save model:", e)

# -------------------------
# Aggregate results
# -------------------------
print("\n\n=== Cross-validation summary ===")
print(f"Fold accuracies: {fold_accuracies}")
print(f"Mean val accuracy: {np.mean(fold_accuracies):.4f} ± {np.std(fold_accuracies):.4f}")

# produce aggregated classification report by summing supports and averaging metrics
# (simple average of per-fold macro f1)
macro_f1s = [r['macro avg']['f1-score'] for r in fold_reports]
print(f"Mean macro F1 over folds: {np.mean(macro_f1s):.4f} ± {np.std(macro_f1s):.4f}")

print(f"Best validation accuracy across folds: {best_val_acc:.4f}")
print("Best model saved at:", best_model_path)
