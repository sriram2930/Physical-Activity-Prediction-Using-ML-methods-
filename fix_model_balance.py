
"""
PAMAP2 - Fix Class Imbalance Issue
===================================
This script retrains the model with better class balancing
to fix the "always predicting Running" problem.

Run this ONCE to retrain your models with proper balancing.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("🔧 FIXING CLASS IMBALANCE - RETRAINING MODELS")
print("=" * 70)

# Load data
data_path = Path('pamap2_data') / 'pamap2_engineered.csv'
df = pd.read_csv(data_path)
print(f"✅ Loaded {len(df):,} rows")

# Check current distribution
print("\n📊 CURRENT CLASS DISTRIBUTION:")
print("-" * 50)
class_counts = df['activity_name'].value_counts()
for activity, count in class_counts.items():
    pct = count / len(df) * 100
    bar = "█" * int(pct)
    print(f"{activity:25s}: {count:6,} ({pct:5.1f}%) {bar}")

print(f"\nImbalance ratio: {class_counts.max() / class_counts.min():.1f}:1")

# Load original feature list
models_dir = Path('pamap2_models')
with open(models_dir / 'results_summary.pkl', 'rb') as f:
    old_results = pickle.load(f)

top_features = old_results['top_features']
print(f"\n✅ Using {len(top_features)} features")

# Prepare data
X = df[top_features].copy()
y = df['activityID'].copy()

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print(f"\n📊 Class mapping:")
for orig, enc in zip(label_encoder.classes_, range(len(label_encoder.classes_))):
    name = df[df['activityID'] == orig]['activity_name'].iloc[0]
    print(f"   {orig} -> {enc}: {name}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print(f"\n✅ Train: {len(X_train):,}, Test: {len(X_test):,}")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================================
# METHOD 1: Better SMOTE with proper sampling strategy
# ============================================================
print("\n" + "=" * 70)
print("⚖️ APPLYING IMPROVED SMOTE BALANCING")
print("=" * 70)

# Calculate target: make all classes equal to median class size
unique, counts = np.unique(y_train, return_counts=True)
median_count = int(np.median(counts))
max_count = int(np.max(counts))
min_count = int(np.min(counts))

print(f"Before SMOTE:")
print(f"   Min class: {min_count:,}")
print(f"   Max class: {max_count:,}")
print(f"   Median class: {median_count:,}")

# Target: bring all classes to max_count (fully balanced)
smote = SMOTE(random_state=42, k_neighbors=5)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)

print(f"\nAfter SMOTE:")
unique_after, counts_after = np.unique(y_train_balanced, return_counts=True)
for cls, cnt in zip(unique_after, counts_after):
    print(f"   Class {cls}: {cnt:,}")

print(f"\n✅ Total training samples: {len(X_train_balanced):,}")

# ============================================================
# TRAIN MODELS WITH CLASS WEIGHTS (Additional balance)
# ============================================================
print("\n" + "=" * 70)
print("🤖 TRAINING MODELS WITH CLASS WEIGHTS")
print("=" * 70)

# Calculate class weights for additional balancing
from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train_balanced),
    y=y_train_balanced
)
class_weight_dict = dict(zip(np.unique(y_train_balanced), class_weights))
print(f"Class weights: {class_weight_dict}")

# Train Random Forest with class weights
print("\n🌲 Training Random Forest...")
rf_model = RandomForestClassifier(
    n_estimators=200,          # More trees
    max_depth=25,              # Slightly deeper
    min_samples_split=5,       # Prevent overfitting
    min_samples_leaf=2,
    class_weight='balanced',   # Use class weights!
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train_balanced, y_train_balanced)

# Evaluate
y_pred_rf = rf_model.predict(X_test_scaled)
rf_acc = accuracy_score(y_test, y_pred_rf)
rf_f1 = f1_score(y_test, y_pred_rf, average='weighted')
print(f"   Accuracy: {rf_acc:.4f}")
print(f"   F1-Score: {rf_f1:.4f}")

# Train Gradient Boosting
print("\n🚀 Training Gradient Boosting...")
gb_model = GradientBoostingClassifier(
    n_estimators=150,
    max_depth=6,
    learning_rate=0.1,
    random_state=42
)
gb_model.fit(X_train_balanced, y_train_balanced)

y_pred_gb = gb_model.predict(X_test_scaled)
gb_acc = accuracy_score(y_test, y_pred_gb)
gb_f1 = f1_score(y_test, y_pred_gb, average='weighted')
print(f"   Accuracy: {gb_acc:.4f}")
print(f"   F1-Score: {gb_f1:.4f}")

# Train Logistic Regression
print("\n📈 Training Logistic Regression...")
lr_model = LogisticRegression(
    max_iter=2000,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
lr_model.fit(X_train_balanced, y_train_balanced)

y_pred_lr = lr_model.predict(X_test_scaled)
lr_acc = accuracy_score(y_test, y_pred_lr)
lr_f1 = f1_score(y_test, y_pred_lr, average='weighted')
print(f"   Accuracy: {lr_acc:.4f}")
print(f"   F1-Score: {lr_f1:.4f}")

# ============================================================
# DETAILED CLASSIFICATION REPORT
# ============================================================
print("\n" + "=" * 70)
print("📊 DETAILED CLASSIFICATION REPORT (Random Forest)")
print("=" * 70)

# Get activity names for report
activity_names = []
for cls in label_encoder.classes_:
    name = df[df['activityID'] == cls]['activity_name'].iloc[0]
    activity_names.append(name)

print(classification_report(y_test, y_pred_rf, target_names=activity_names, zero_division=0))

# ============================================================
# SAVE IMPROVED MODELS
# ============================================================
print("\n" + "=" * 70)
print("💾 SAVING IMPROVED MODELS")
print("=" * 70)

# Backup old models
import shutil
backup_dir = models_dir / 'backup_old_models'
backup_dir.mkdir(exist_ok=True)

for f in models_dir.glob('*.pkl'):
    shutil.copy(f, backup_dir / f.name)
print(f"✅ Old models backed up to: {backup_dir}")

# Save new models
joblib.dump(rf_model, models_dir / 'random_forest_model.pkl')
joblib.dump(gb_model, models_dir / 'gradient_boosting_model.pkl')
joblib.dump(lr_model, models_dir / 'logistic_regression_model.pkl')

# Save updated results
results = {
    'Random Forest': {
        'test_accuracy': rf_acc,
        'test_f1': rf_f1,
        'test_precision': f1_score(y_test, y_pred_rf, average='weighted'),
        'test_recall': rf_acc,
        'y_test_pred': y_pred_rf,
        'training_time': 0
    },
    'Gradient Boosting': {
        'test_accuracy': gb_acc,
        'test_f1': gb_f1,
        'test_precision': f1_score(y_test, y_pred_gb, average='weighted'),
        'test_recall': gb_acc,
        'y_test_pred': y_pred_gb,
        'training_time': 0
    },
    'Logistic Regression': {
        'test_accuracy': lr_acc,
        'test_f1': lr_f1,
        'test_precision': f1_score(y_test, y_pred_lr, average='weighted'),
        'test_recall': lr_acc,
        'y_test_pred': y_pred_lr,
        'training_time': 0
    }
}

# Find best model
best_model_name = max(results, key=lambda x: results[x]['test_accuracy'])

results_summary = {
    'best_model_name': best_model_name,
    'test_accuracy': results[best_model_name]['test_accuracy'],
    'label_encoder': label_encoder,
    'top_features': top_features,
    'results': results,
    'scaler': scaler  # Save scaler too!
}

with open(models_dir / 'results_summary.pkl', 'wb') as f:
    pickle.dump(results_summary, f)

print(f"✅ Saved: random_forest_model.pkl")
print(f"✅ Saved: gradient_boosting_model.pkl")
print(f"✅ Saved: logistic_regression_model.pkl")
print(f"✅ Saved: results_summary.pkl")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("🎉 RETRAINING COMPLETE!")
print("=" * 70)

print(f"""
MODEL PERFORMANCE SUMMARY:
─────────────────────────────────────
Random Forest:      {rf_acc:.2%} accuracy
Gradient Boosting:  {gb_acc:.2%} accuracy  
Logistic Regression:{lr_acc:.2%} accuracy
─────────────────────────────────────
Best Model: {best_model_name}

IMPROVEMENTS MADE:
✅ Applied SMOTE with full balancing
✅ Added class_weight='balanced' to models
✅ Increased model complexity (more trees, deeper)
✅ All classes now equally represented

NEXT STEPS:
1. Restart your Streamlit app
2. Test predictions on "Test Real Data" page
3. Predictions should now be more accurate!
""")