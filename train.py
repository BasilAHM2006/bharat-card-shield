import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_fscore_support
import joblib
import os

# Create folders if they don't exist
os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)

print("⚡ Generating financial transaction dataset with 0.2% fraud imbalance...")
np.random.seed(42)
n_samples = 30000

# 99.8% Legitimate (0) and 0.2% Fraud (1)
fraud_prob = 0.002
is_fraud = np.random.binomial(1, fraud_prob, n_samples)
fraud_indices = np.where(is_fraud == 1)[0]
legit_indices = np.where(is_fraud == 0)[0]

# Amounts: Normal transactions are smaller; fraud transactions have larger deviations
amounts = np.zeros(n_samples)
amounts[legit_indices] = np.random.exponential(scale=70.0, size=len(legit_indices)) + 1.5
amounts[fraud_indices] = np.random.exponential(scale=520.0, size=len(fraud_indices)) + 140.0

# Transaction hours: fraud clusters during odd hours (late night / early morning)
hours = np.zeros(n_samples, dtype=int)
hours[legit_indices] = np.random.choice(np.arange(6, 24), size=len(legit_indices))
hours[fraud_indices] = np.random.choice([0, 1, 2, 3, 4, 5, 22, 23], size=len(fraud_indices))

# Behavioral signals (PCA components V1 - V4)
v1 = np.random.normal(0, 1, n_samples)
v2 = np.random.normal(0, 1, n_samples)
v3 = np.random.normal(0, 1, n_samples)
v4 = np.random.normal(0, 1, n_samples)

# Induce anomaly shifts for fraud cases
v1[fraud_indices] -= 3.4
v2[fraud_indices] += 2.9
v3[fraud_indices] -= 4.2
v4[fraud_indices] += 3.7

df = pd.DataFrame({
    'Hour': hours,
    'Amount': np.round(amounts, 2),
    'V1': np.round(v1, 4),
    'V2': np.round(v2, 4),
    'V3': np.round(v3, 4),
    'V4': np.round(v4, 4),
    'Is_International': np.random.choice([0, 1], p=[0.92, 0.08], size=n_samples),
    'Device_Trust_Score': np.random.uniform(0.15, 1.0, n_samples),
    'Class': is_fraud
})

df.loc[fraud_indices, 'Device_Trust_Score'] = np.random.uniform(0.01, 0.32, len(fraud_indices))
df.to_csv("data/transactions.csv", index=False)
print(f"✅ Ingested {len(df)} transactions | Total Frauds: {df['Class'].sum()} ({df['Class'].mean()*100:.2f}%)")

# Train / Test Split
X = df.drop(columns=['Class'])
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

print("🧠 Training Weighted Random Forest Classifier (Optimized for Recall)...")
model = RandomForestClassifier(
    n_estimators=120,
    max_depth=12,
    class_weight={0: 1, 1: 50},  # Heavy penalty for missed fraud
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
auc = roc_auc_score(y_test, y_prob)

print("\n--- MODEL PERFORMANCE REPORT ---")
print(f"ROC-AUC Score : {auc:.4f}")
print(f"Recall (Fraud): {recall:.4f}")
print(f"Precision     : {precision:.4f}")
print(f"F1-Score      : {f1:.4f}")
print("--------------------------------\n")

# Save model and metadata
joblib.dump(model, "models/fraud_model.pkl")
joblib.dump({
    'features': list(X.columns),
    'auc': auc,
    'recall': recall,
    'precision': precision,
    'f1': f1,
    'total_transactions': len(df),
    'total_frauds': int(df['Class'].sum())
}, "models/model_metrics.pkl")

print("💾 Model artifacts saved to models/fraud_model.pkl")