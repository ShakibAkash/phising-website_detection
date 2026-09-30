"""
Phishing Website Detection - Extended Training & Benchmark Pipeline
Includes:
- Dataset expansion (phishing_site_urls.csv + malicious_phish.csv)
- Lexical feature extraction (11 features)
- Models:
  1. Logistic Regression (Baseline Linear Classifier)
  2. Decision Tree Classifier
  3. Random Forest Classifier
  4. XGBoost Classifier
  5. Deep Neural Network (DNN / MLP on tabular features)
  6. Character-level 1D-CNN (on raw URL character sequences)
  7. Character-level Transformer (on raw URL character sequences)
- Full Performance Benchmarking (Accuracy, Precision, Recall, F1, Training Time)
- Confusion Matrices Generation
- Misclassification Analysis (Tracking mismatched prediction indices)
- Model Artifact Serialization for App Deployment
"""

import os
import re
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# Ensure output directory exists
os.makedirs("models", exist_ok=True)
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

FEATURE_COLUMNS = [
    'url_length', 'domain_length', 'count_dots', 'count_hyphens',
    'count_at', 'count_double_slash', 'count_question', 'count_slash',
    'count_equal', 'count_http', 'has_ip'
]

# ==========================================
# 1. Dataset Loading & Integration
# ==========================================
def load_and_combine_datasets(include_malicious_phish: bool = True) -> pd.DataFrame:
    print("--- 1. Loading Datasets ---")
    orig_path = "Dataset/phishing_site_urls.csv"
    df_orig = pd.read_csv(orig_path)
    df_orig = df_orig.rename(columns={"URL": "url", "Label": "label"})
    df_orig["label"] = df_orig["label"].map({"good": 0, "bad": 1})
    print(f"Original dataset: {len(df_orig):,} rows")

    if include_malicious_phish and os.path.exists("Dataset/malicious_phish.csv"):
        print("Integrating additional dataset: Dataset/malicious_phish.csv...")
        df_mal = pd.read_csv("Dataset/malicious_phish.csv")
        # Map benign to 0 (safe) and phishing/defacement/malware to 1 (bad)
        label_map = {
            "benign": 0,
            "phishing": 1,
            "defacement": 1,
            "malware": 1
        }
        df_mal["label"] = df_mal["type"].map(label_map)
        df_mal = df_mal[["url", "label"]]
        print(f"Additional dataset: {len(df_mal):,} rows")
        
        combined_df = pd.concat([df_orig, df_mal], ignore_index=True)
    else:
        combined_df = df_orig

    # Clean & Deduplicate
    initial_len = len(combined_df)
    combined_df = combined_df.dropna(subset=["url", "label"])
    combined_df["url"] = combined_df["url"].astype(str).str.strip()
    combined_df = combined_df.drop_duplicates(subset=["url"]).reset_index(drop=True)
    print(f"Total unique URLs after deduplication: {len(combined_df):,} (from {initial_len:,})")
    print(f"Class distribution: Safe (0) = {(combined_df['label'] == 0).sum():,}, Phishing (1) = {(combined_df['label'] == 1).sum():,}\n")
    return combined_df

# ==========================================
# 2. Feature Extraction
# ==========================================
def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    print("--- 2. Extracting Lexical Features ---")
    start_t = time.time()
    
    df['url_length'] = df['url'].str.len()
    extracted_domain = df['url'].str.extract(r'^(?:https?://)?([^/?#]+)')[0].fillna('')
    df['domain_length'] = extracted_domain.str.len()
    
    df['count_dots'] = df['url'].str.count(r'\.')
    df['count_hyphens'] = df['url'].str.count(r'-')
    df['count_at'] = df['url'].str.count(r'@')
    df['count_double_slash'] = df['url'].str.count(r'//')
    df['count_question'] = df['url'].str.count(r'\?')
    df['count_slash'] = df['url'].str.count(r'/')
    df['count_equal'] = df['url'].str.count(r'=')
    df['count_http'] = df['url'].str.count(r'http')
    
    ip_pattern = r'(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)'
    df['has_ip'] = df['url'].str.contains(ip_pattern, regex=True).astype(int)
    
    print(f"Feature extraction completed in {time.time() - start_t:.2f}s\n")
    return df

# URL Character Tokenizer for PyTorch Deep Learning Models
def urls_to_sequences(urls: pd.Series, max_len: int = 150) -> np.ndarray:
    seqs = np.zeros((len(urls), max_len), dtype=np.int64)
    for i, u in enumerate(urls):
        chars = [min(ord(c), 127) for c in str(u)[:max_len]]
        seqs[i, :len(chars)] = chars
    return seqs

# ==========================================
# 3. Deep Learning Architectures (PyTorch)
# ==========================================
class TabularDNN(nn.Module):
    """Deep Neural Network (MLP) for Tabular Lexical Features"""
    def __init__(self, in_features: int = 11):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

class CharCNN(nn.Module):
    """1D-CNN operating directly on Raw URL character sequences"""
    def __init__(self, vocab_size: int = 128, embed_dim: int = 32):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.conv1 = nn.Conv1d(embed_dim, 64, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        emb = self.embed(x).transpose(1, 2)
        c1 = torch.relu(self.conv1(emb))
        c2 = torch.relu(self.conv2(c1))
        pooled = self.pool(c2).squeeze(-1)
        return self.fc(pooled).squeeze(-1)

class CharTransformer(nn.Module):
    """Lightweight Transformer Encoder operating on Raw URL character tokens"""
    def __init__(self, vocab_size: int = 128, embed_dim: int = 64, num_heads: int = 4, max_len: int = 150):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, embed_dim) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=128, dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        seq_len = x.size(1)
        emb = self.embed(x) + self.pos_embed[:, :seq_len, :]
        out = self.transformer(emb)
        pooled = out.mean(dim=1)
        return self.fc(pooled).squeeze(-1)

# PyTorch Training & Evaluation Helpers
def train_torch_model(model: nn.Module, loader: DataLoader, epochs: int = 3, lr: float = 0.001) -> float:
    start_t = time.time()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for ep in range(epochs):
        for bx, by in loader:
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
    return time.time() - start_t

def predict_torch_model(model: nn.Module, X_tensor: torch.Tensor, batch_size: int = 1024) -> np.ndarray:
    model.eval()
    all_preds = []
    with torch.no_grad():
        for i in range(0, len(X_tensor), batch_size):
            bx = X_tensor[i:i + batch_size]
            logits = model(bx)
            probs = torch.sigmoid(logits)
            all_preds.append((probs >= 0.5).int().cpu().numpy())
    return np.concatenate(all_preds)

# ==========================================
# 4. Main Training Pipeline & Benchmark
# ==========================================
def main():
    print("==========================================================")
    print("   Phishing Website Detection - Model Training Suite     ")
    print("==========================================================")
    
    # 1. Load data
    df = load_and_combine_datasets(include_malicious_phish=True)
    df = extract_features(df)
    
    # 2. Prepare tabular features & labels
    X = df[FEATURE_COLUMNS]
    y = df['label'].values
    raw_urls = df['url']
    
    # 3. Train-test split (80/20 stratified)
    X_train_df, X_test_df, y_train, y_test, urls_train, urls_test = train_test_split(
        X, y, raw_urls, test_size=0.20, random_state=RANDOM_SEED, stratify=y
    )
    print(f"Dataset split: Train = {len(y_train):,} samples, Test = {len(y_test):,} samples\n")
    
    # Scale tabular features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_df)
    X_test_scaled = scaler.transform(X_test_df)
    joblib.dump(scaler, "models/scaler.joblib")
    print("StandardScaler fitted and saved to models/scaler.joblib")
    
    # Dictionaries to store evaluation metrics
    trained_models = {}
    training_times = {}
    test_predictions = {}
    
    # ----------------------------------------------------
    # Model 1: Logistic Regression (Baseline Linear Model)
    # ----------------------------------------------------
    print("\n--- Training Model 1: Logistic Regression ---")
    t0 = time.time()
    lr_model = LogisticRegression(max_iter=500, random_state=RANDOM_SEED)
    lr_model.fit(X_train_scaled, y_train)
    lr_time = time.time() - t0
    trained_models["Logistic Regression"] = lr_model
    training_times["Logistic Regression"] = lr_time
    test_predictions["Logistic Regression"] = lr_model.predict(X_test_scaled)
    joblib.dump(lr_model, "models/logistic_regression.joblib")
    print(f"Logistic Regression trained in {lr_time:.2f}s")
    
    # ----------------------------------------------------
    # Model 2: Decision Tree Classifier
    # ----------------------------------------------------
    print("\n--- Training Model 2: Decision Tree Classifier ---")
    t0 = time.time()
    dt_model = DecisionTreeClassifier(max_depth=15, random_state=RANDOM_SEED)
    dt_model.fit(X_train_scaled, y_train)
    dt_time = time.time() - t0
    trained_models["Decision Tree"] = dt_model
    training_times["Decision Tree"] = dt_time
    test_predictions["Decision Tree"] = dt_model.predict(X_test_scaled)
    joblib.dump(dt_model, "models/decision_tree.joblib")
    print(f"Decision Tree trained in {dt_time:.2f}s")
    
    # ----------------------------------------------------
    # Model 3: Random Forest Classifier
    # ----------------------------------------------------
    print("\n--- Training Model 3: Random Forest Classifier ---")
    t0 = time.time()
    rf_model = RandomForestClassifier(
        n_estimators=100, max_depth=15, n_jobs=-1, random_state=RANDOM_SEED
    )
    rf_model.fit(X_train_scaled, y_train)
    rf_time = time.time() - t0
    trained_models["Random Forest"] = rf_model
    training_times["Random Forest"] = rf_time
    test_predictions["Random Forest"] = rf_model.predict(X_test_scaled)
    joblib.dump(rf_model, "models/random_forest.joblib")
    print(f"Random Forest trained in {rf_time:.2f}s")
    
    # ----------------------------------------------------
    # Model 4: XGBoost Classifier
    # ----------------------------------------------------
    print("\n--- Training Model 4: XGBoost Classifier ---")
    t0 = time.time()
    xgb_model = XGBClassifier(
        n_estimators=100, max_depth=6, learning_rate=0.1,
        n_jobs=-1, random_state=RANDOM_SEED, eval_metric="logloss"
    )
    xgb_model.fit(X_train_scaled, y_train)
    xgb_time = time.time() - t0
    trained_models["XGBoost"] = xgb_model
    training_times["XGBoost"] = xgb_time
    test_predictions["XGBoost"] = xgb_model.predict(X_test_scaled)
    joblib.dump(xgb_model, "models/xgboost.joblib")
    print(f"XGBoost trained in {xgb_time:.2f}s")
    
    # ----------------------------------------------------
    # Model 5: Deep Neural Network (DNN / MLP on Tabular Features)
    # ----------------------------------------------------
    print("\n--- Training Model 5: Deep Neural Network (Tabular DNN) ---")
    X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32)
    
    dnn_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=1024, shuffle=True)
    dnn_model = TabularDNN(in_features=11)
    dnn_time = train_torch_model(dnn_model, dnn_loader, epochs=4, lr=0.001)
    
    trained_models["Deep Neural Network (DNN)"] = dnn_model
    training_times["Deep Neural Network (DNN)"] = dnn_time
    test_predictions["Deep Neural Network (DNN)"] = predict_torch_model(dnn_model, X_test_t)
    torch.save(dnn_model.state_dict(), "models/dnn_model.pt")
    print(f"Tabular DNN trained in {dnn_time:.2f}s (saved to models/dnn_model.pt)")
    
    # ----------------------------------------------------
    # Models 6 & 7: Sequence Deep Learning (Char-CNN & Char-Transformer)
    # ----------------------------------------------------
    # Use representative stratified subset of 60,000 URLs for sequence models to keep CPU runtime optimal
    print("\n--- Preparing Sequence Data for CNN & Transformer ---")
    seq_sample_size = min(60000, len(y_train))
    idx_sample = np.random.choice(len(y_train), seq_sample_size, replace=False)
    
    urls_train_sample = urls_train.iloc[idx_sample]
    y_train_sample = y_train[idx_sample]
    
    test_sample_size = min(20000, len(y_test))
    idx_test_sample = np.random.choice(len(y_test), test_sample_size, replace=False)
    urls_test_sample = urls_test.iloc[idx_test_sample]
    y_test_sample = y_test[idx_test_sample]
    
    seq_train = torch.tensor(urls_to_sequences(urls_train_sample), dtype=torch.long)
    seq_test = torch.tensor(urls_to_sequences(urls_test_sample), dtype=torch.long)
    y_seq_train = torch.tensor(y_train_sample, dtype=torch.float32)
    
    seq_loader = DataLoader(TensorDataset(seq_train, y_seq_train), batch_size=512, shuffle=True)
    
    # Model 6: Char 1D-CNN
    print("\n--- Training Model 6: Character-level 1D-CNN ---")
    cnn_model = CharCNN(vocab_size=128, embed_dim=32)
    cnn_time = train_torch_model(cnn_model, seq_loader, epochs=2, lr=0.001)
    trained_models["Char 1D-CNN"] = cnn_model
    training_times["Char 1D-CNN"] = cnn_time
    cnn_preds_sample = predict_torch_model(cnn_model, seq_test)
    torch.save(cnn_model.state_dict(), "models/char_cnn.pt")
    print(f"Char 1D-CNN trained in {cnn_time:.2f}s (saved to models/char_cnn.pt)")
    
    # Model 7: Char Transformer
    print("\n--- Training Model 7: Character-level Transformer ---")
    trans_model = CharTransformer(vocab_size=128, embed_dim=64, num_heads=4)
    trans_time = train_torch_model(trans_model, seq_loader, epochs=2, lr=0.001)
    trained_models["Char Transformer"] = trans_model
    training_times["Char Transformer"] = trans_time
    trans_preds_sample = predict_torch_model(trans_model, seq_test)
    torch.save(trans_model.state_dict(), "models/char_transformer.pt")
    print(f"Char Transformer trained in {trans_time:.2f}s (saved to models/char_transformer.pt)")
    
    # ==========================================
    # 5. Model Evaluation & Benchmark Summary
    # ==========================================
    print("\n==========================================================")
    print("                 MODEL BENCHMARK RESULTS                  ")
    print("==========================================================")
    benchmark_records = []
    
    # Full test set models
    for name in ["Logistic Regression", "Decision Tree", "Random Forest", "XGBoost", "Deep Neural Network (DNN)"]:
        preds = test_predictions[name]
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, zero_division=0)
        rec = recall_score(y_test, preds, zero_division=0)
        f1 = f1_score(y_test, preds, zero_division=0)
        t_sec = training_times[name]
        benchmark_records.append({
            "Model": name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1-Score": f1,
            "Training Time": f"{t_sec:.2f}s",
            "Evaluated Samples": len(y_test)
        })
    
    # Sequence models (evaluated on representative test subset)
    for name, s_preds in [("Char 1D-CNN", cnn_preds_sample), ("Char Transformer", trans_preds_sample)]:
        acc = accuracy_score(y_test_sample, s_preds)
        prec = precision_score(y_test_sample, s_preds, zero_division=0)
        rec = recall_score(y_test_sample, s_preds, zero_division=0)
        f1 = f1_score(y_test_sample, s_preds, zero_division=0)
        t_sec = training_times[name]
        benchmark_records.append({
            "Model": name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1-Score": f1,
            "Training Time": f"{t_sec:.2f}s",
            "Evaluated Samples": len(y_test_sample)
        })
    
    summary_df = pd.DataFrame(benchmark_records)
    print(summary_df.to_string(index=False))
    summary_df.to_csv("models/extended_benchmark_summary.csv", index=False)
    print("\nBenchmark saved to models/extended_benchmark_summary.csv")
    
    # Identify top model
    best_row = summary_df.sort_values(by="F1-Score", ascending=False).iloc[0]
    best_model_name = best_row["Model"]
    print(f"\nTop Model by F1-Score: {best_model_name} (F1: {best_row['F1-Score']:.4f}, Accuracy: {best_row['Accuracy']:.4f})")
    
    # Save best model reference
    if best_model_name in trained_models and hasattr(trained_models[best_model_name], "predict"):
        joblib.dump(trained_models[best_model_name], "models/best_model.joblib")
    else:
        # Fallback to top sklearn/tree model
        joblib.dump(trained_models["Random Forest"], "models/best_model.joblib")
    
    # ==========================================
    # 6. Misclassification Analysis (Requirement 2)
    # ==========================================
    print("\n--- 6. Misclassification Analysis (Tracking Mismatched Indices) ---")
    rf_preds = test_predictions["Random Forest"]
    mismatch_mask = (y_test != rf_preds)
    mismatch_indices = np.where(mismatch_mask)[0]
    
    print(f"Total misclassified test samples in Random Forest: {len(mismatch_indices):,} (out of {len(y_test):,})")
    print(f"First 10 misclassified indices: {mismatch_indices[:10].tolist()}")
    
    # Export sample misclassified URLs with their actual vs predicted labels
    misclassified_df = pd.DataFrame({
        "Test_Index": mismatch_indices[:200],
        "URL": urls_test.iloc[mismatch_indices[:200]].values,
        "Actual_Label": ["Phishing (1)" if y == 1 else "Legitimate (0)" for y in y_test[mismatch_indices[:200]]],
        "Predicted_Label": ["Phishing (1)" if p == 1 else "Legitimate (0)" for p in rf_preds[mismatch_indices[:200]]]
    })
    misclassified_df.to_csv("models/misclassified_samples.csv", index=False)
    print("Sample misclassified cases saved to models/misclassified_samples.csv for deep error analysis.")
    
    # ----------------------------------------------------
    # Inter-Model Disagreement Analysis (Requirement 2 - Meaning 2)
    # ----------------------------------------------------
    print("\n--- Inter-Model Disagreement Analysis (Decision Tree vs Random Forest vs XGBoost) ---")
    dt_preds = test_predictions["Decision Tree"]
    xgb_preds = test_predictions["XGBoost"]
    
    dt_rf_disagree_mask = (dt_preds != rf_preds)
    dt_rf_disagree_indices = np.where(dt_rf_disagree_mask)[0]
    total_dt_rf = len(dt_rf_disagree_indices)
    
    rf_correct = int(np.sum(rf_preds[dt_rf_disagree_indices] == y_test[dt_rf_disagree_indices]))
    dt_correct = int(np.sum(dt_preds[dt_rf_disagree_indices] == y_test[dt_rf_disagree_indices]))
    
    print(f"Total test URLs where DT and RF disagreed: {total_dt_rf:,}")
    print(f"Random Forest was correct in {rf_correct:,} disagreements ({(rf_correct/total_dt_rf)*100:.2f}%)")
    print(f"Decision Tree was correct in {dt_correct:,} disagreements ({(dt_correct/total_dt_rf)*100:.2f}%)")
    
    sample_dis_idx = dt_rf_disagree_indices[:200]
    who_was_right = []
    for idx in sample_dis_idx:
        rf_ok = (rf_preds[idx] == y_test[idx])
        dt_ok = (dt_preds[idx] == y_test[idx])
        if rf_ok and not dt_ok:
            who_was_right.append("Random Forest (Ensemble)")
        elif dt_ok and not rf_ok:
            who_was_right.append("Decision Tree")
        else:
            who_was_right.append("Neither (Both Wrong)")
            
    disagreement_df = pd.DataFrame({
        "Test_Index": sample_dis_idx,
        "URL": urls_test.iloc[sample_dis_idx].values,
        "Actual_Label": ["Phishing (1)" if yt == 1 else "Legitimate (0)" for yt in y_test[sample_dis_idx]],
        "DT_Prediction": ["Phishing (1)" if p == 1 else "Legitimate (0)" for p in dt_preds[sample_dis_idx]],
        "RF_Prediction": ["Phishing (1)" if p == 1 else "Legitimate (0)" for p in rf_preds[sample_dis_idx]],
        "XGB_Prediction": ["Phishing (1)" if p == 1 else "Legitimate (0)" for p in xgb_preds[sample_dis_idx]],
        "More_Logical_Winner": who_was_right
    })
    disagreement_df.to_csv("models/model_disagreements.csv", index=False)
    print("Model disagreement analysis cases saved to models/model_disagreements.csv")
    
    # ==========================================
    # 7. Generate Confusion Matrix Plot
    # ==========================================
    print("\n--- 7. Generating Confusion Matrices Plot ---")
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    models_to_plot = [
        ("Logistic Regression", test_predictions["Logistic Regression"], y_test),
        ("Decision Tree", test_predictions["Decision Tree"], y_test),
        ("Random Forest", test_predictions["Random Forest"], y_test),
        ("XGBoost", test_predictions["XGBoost"], y_test),
        ("Deep Neural Network (DNN)", test_predictions["Deep Neural Network (DNN)"], y_test),
        ("Char 1D-CNN", cnn_preds_sample, y_test_sample)
    ]
    
    for i, (name, preds, true_labels) in enumerate(models_to_plot):
        cm = confusion_matrix(true_labels, preds)
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues", ax=axes[i],
            xticklabels=["Safe (0)", "Phish (1)"],
            yticklabels=["Safe (0)", "Phish (1)"]
        )
        axes[i].set_title(f"{name}\nConfusion Matrix", fontsize=12, fontweight="bold")
        axes[i].set_xlabel("Predicted")
        axes[i].set_ylabel("Actual")
        
    plt.tight_layout()
    plt.savefig("models/confusion_matrices_all_models.png", dpi=300)
    plt.close()
    print("Confusion matrices plot saved to models/confusion_matrices_all_models.png")
    print("\nTraining and evaluation pipeline completed successfully!")

if __name__ == "__main__":
    main()
