"""
PAMAP2 ML VISUALIZATIONS ONLY
==============================
Run this AFTER train_ml_models.py
Generates 6 ML visualizations (ch5_05 to ch5_10)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report

sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300

print("=" * 70)
print("📊 PAMAP2 ML VISUALIZATIONS")
print("=" * 70)

# Paths
models_dir = Path('pamap2_models')
output_dir = Path('pamap2_visualizations')
data_dir = Path('pamap2_data')

# Load results
print("\n📂 Loading saved results...")
with open(models_dir / 'results_summary.pkl', 'rb') as f:
    results_summary = pickle.load(f)

best_model_name = results_summary['best_model_name']
results = results_summary['results']
top_features = results_summary['top_features']

print(f"✅ Best model: {best_model_name}")

# Reload data to recreate test set
print("📂 Loading data...")
df = pd.read_csv(data_dir / 'pamap2_engineered.csv')

activity_mapping = {aid: df[df['activityID']==aid]['activity_name'].iloc[0] 
                   for aid in df['activityID'].unique()}

X = df[top_features].copy()
y = df['activityID'].copy()

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

n_classes = len(np.unique(y_test))
activity_labels = label_encoder.inverse_transform(np.arange(n_classes))
activity_names = [activity_mapping.get(l, f'Class {l}') for l in activity_labels]

y_pred = results[best_model_name]['y_test_pred']
best_acc = results[best_model_name]['test_accuracy']

print(f"✅ Test samples: {len(y_test):,}")
print(f"✅ Classes: {n_classes}")

# ============== VIZ 1: CONFUSION MATRIX ==============
print("\n📊 Creating Confusion Matrix...")
fig, axes = plt.subplots(1, 2, figsize=(18, 8))
fig.suptitle(f'Confusion Matrix - {best_model_name} (Accuracy: {best_acc:.2%})', 
             fontsize=14, fontweight='bold')

cm = confusion_matrix(y_test, y_pred)
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
           xticklabels=activity_names, yticklabels=activity_names)
axes[0].set_xlabel('Predicted', fontweight='bold')
axes[0].set_ylabel('True', fontweight='bold')
axes[0].set_title('Counts')
plt.setp(axes[0].get_xticklabels(), rotation=45, ha='right', fontsize=7)
plt.setp(axes[0].get_yticklabels(), fontsize=7)

sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='RdYlGn', ax=axes[1],
           xticklabels=activity_names, yticklabels=activity_names, vmin=0, vmax=1)
axes[1].set_xlabel('Predicted', fontweight='bold')
axes[1].set_ylabel('True', fontweight='bold')
axes[1].set_title('Normalized')
plt.setp(axes[1].get_xticklabels(), rotation=45, ha='right', fontsize=7)
plt.setp(axes[1].get_yticklabels(), fontsize=7)

plt.tight_layout()
plt.savefig(output_dir / 'ch5_05_confusion_matrix.png', bbox_inches='tight')
plt.close()
print("   ✅ ch5_05_confusion_matrix.png")

# ============== VIZ 2: PER-ACTIVITY PERFORMANCE ==============
print("📊 Creating Per-Activity Performance...")
report = classification_report(y_test, y_pred, target_names=activity_names, 
                               output_dict=True, zero_division=0)

precision = [report[a]['precision'] for a in activity_names]
recall = [report[a]['recall'] for a in activity_names]
f1_scores = [report[a]['f1-score'] for a in activity_names]
support = [report[a]['support'] for a in activity_names]

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle(f'Per-Activity Performance - {best_model_name}', fontsize=14, fontweight='bold')

# Metrics comparison
ax1 = axes[0, 0]
x = np.arange(len(activity_names))
w = 0.25
ax1.barh(x-w, precision, w, label='Precision', color='steelblue', alpha=0.8)
ax1.barh(x, recall, w, label='Recall', color='coral', alpha=0.8)
ax1.barh(x+w, f1_scores, w, label='F1', color='green', alpha=0.8)
ax1.set_yticks(x)
ax1.set_yticklabels(activity_names, fontsize=8)
ax1.set_xlabel('Score')
ax1.set_title('Metrics by Activity')
ax1.legend()
ax1.set_xlim([0, 1.05])
ax1.grid(axis='x', alpha=0.3)

# F1 ranking
ax2 = axes[0, 1]
sorted_idx = np.argsort(f1_scores)
sorted_f1 = [f1_scores[i] for i in sorted_idx]
sorted_names = [activity_names[i] for i in sorted_idx]
colors = ['red' if f<0.7 else 'orange' if f<0.85 else 'green' for f in sorted_f1]
ax2.barh(range(len(sorted_names)), sorted_f1, color=colors, alpha=0.7)
ax2.set_yticks(range(len(sorted_names)))
ax2.set_yticklabels(sorted_names, fontsize=8)
ax2.set_xlabel('F1-Score')
ax2.set_title('F1 Ranking (Red<0.7, Orange<0.85, Green≥0.85)')
ax2.set_xlim([0, 1.05])
ax2.grid(axis='x', alpha=0.3)

# Support
ax3 = axes[1, 0]
ax3.bar(range(len(activity_names)), support, color='purple', alpha=0.7)
ax3.set_xticks(range(len(activity_names)))
ax3.set_xticklabels(activity_names, rotation=45, ha='right', fontsize=7)
ax3.set_ylabel('Test Samples')
ax3.set_title('Support per Activity')
ax3.grid(axis='y', alpha=0.3)

# Summary
ax4 = axes[1, 1]
summary = f"""
PERFORMANCE SUMMARY

Average Metrics:
  F1-Score:  {np.mean(f1_scores):.3f}
  Precision: {np.mean(precision):.3f}
  Recall:    {np.mean(recall):.3f}

Best Activity:
  {activity_names[np.argmax(f1_scores)]}
  F1: {max(f1_scores):.3f}

Worst Activity:
  {activity_names[np.argmin(f1_scores)]}
  F1: {min(f1_scores):.3f}

Activities ≥0.85 F1: {sum(1 for f in f1_scores if f>=0.85)}
Activities <0.70 F1: {sum(1 for f in f1_scores if f<0.70)}
"""
ax4.text(0.05, 0.95, summary, transform=ax4.transAxes, fontsize=10,
         va='top', family='monospace', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
ax4.axis('off')

plt.tight_layout()
plt.savefig(output_dir / 'ch5_06_per_activity_performance.png', bbox_inches='tight')
plt.close()
print("   ✅ ch5_06_per_activity_performance.png")

# ============== VIZ 3: FEATURE IMPORTANCE ==============
print("📊 Creating Feature Importance...")
rf_path = models_dir / 'random_forest_model.pkl'
if rf_path.exists():
    rf_model = joblib.load(rf_path)
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle('Feature Importance - Random Forest', fontsize=14, fontweight='bold')
    
    # Top 20 features
    top_n = 20
    ax1 = axes[0]
    colors = plt.cm.viridis(np.linspace(0, 1, top_n))
    ax1.barh(range(top_n), importances[indices[:top_n]], color=colors, alpha=0.7)
    ax1.set_yticks(range(top_n))
    ax1.set_yticklabels([top_features[i] for i in indices[:top_n]], fontsize=8)
    ax1.set_xlabel('Importance')
    ax1.set_title(f'Top {top_n} Features')
    ax1.invert_yaxis()
    ax1.grid(axis='x', alpha=0.3)
    
    # Cumulative
    ax2 = axes[1]
    cumsum = np.cumsum(np.sort(importances)[::-1])
    ax2.plot(range(1, len(cumsum)+1), cumsum, 'b-', linewidth=2)
    ax2.fill_between(range(1, len(cumsum)+1), cumsum, alpha=0.3)
    ax2.axhline(0.8, color='r', linestyle='--', label='80%')
    ax2.axhline(0.9, color='orange', linestyle='--', label='90%')
    ax2.set_xlabel('Number of Features')
    ax2.set_ylabel('Cumulative Importance')
    ax2.set_title('Cumulative Feature Importance')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'ch5_07_feature_importance.png', bbox_inches='tight')
    plt.close()
    print("   ✅ ch5_07_feature_importance.png")

# ============== VIZ 4: ERROR ANALYSIS ==============
print("📊 Creating Error Analysis...")
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Error Analysis', fontsize=14, fontweight='bold')

correct = (y_test == y_pred).sum()
incorrect = (y_test != y_pred).sum()
acc = correct / len(y_test)

# Error rate by activity
ax1 = axes[0, 0]
error_rates = []
for i in range(n_classes):
    mask = y_test == i
    if mask.sum() > 0:
        err = (y_test[mask] != y_pred[mask]).sum() / mask.sum() * 100
        error_rates.append(err)
    else:
        error_rates.append(0)

colors = ['green' if e<10 else 'orange' if e<20 else 'red' for e in error_rates]
ax1.barh(activity_names, error_rates, color=colors, alpha=0.7)
ax1.set_xlabel('Error Rate (%)')
ax1.set_title('Error Rate by Activity')
ax1.grid(axis='x', alpha=0.3)

# Top confusions
ax2 = axes[0, 1]
pairs = []
for i in range(len(cm)):
    for j in range(len(cm)):
        if i != j and cm[i,j] > 0:
            pairs.append((activity_names[i], activity_names[j], cm[i,j]))
pairs.sort(key=lambda x: x[2], reverse=True)
top_pairs = pairs[:8]

if top_pairs:
    labels = [f"{t[:8]}→{p[:8]}" for t,p,_ in top_pairs]
    counts = [c for _,_,c in top_pairs]
    ax2.barh(range(len(labels)), counts, color='crimson', alpha=0.6)
    ax2.set_yticks(range(len(labels)))
    ax2.set_yticklabels(labels, fontsize=8)
    ax2.set_xlabel('Count')
    ax2.set_title('Top Confusion Pairs')
    ax2.grid(axis='x', alpha=0.3)

# Accuracy pie
ax3 = axes[1, 0]
ax3.pie([correct, incorrect], labels=[f'Correct\n{correct:,}', f'Wrong\n{incorrect:,}'],
       autopct='%1.1f%%', colors=['green','red'], explode=(0.02,0.02))
ax3.set_title(f'Overall: {acc:.2%} Accuracy')

# Summary
ax4 = axes[1, 1]
err_summary = f"""
ERROR SUMMARY

Correct:   {correct:,} ({acc:.1%})
Incorrect: {incorrect:,} ({1-acc:.1%})

Best Activity:
  {activity_names[np.argmin(error_rates)]}
  Error: {min(error_rates):.1f}%

Worst Activity:
  {activity_names[np.argmax(error_rates)]}
  Error: {max(error_rates):.1f}%

Top Confusion:
  {top_pairs[0][0]} → {top_pairs[0][1]}
  ({top_pairs[0][2]} cases)
"""
ax4.text(0.05, 0.95, err_summary, transform=ax4.transAxes, fontsize=10,
         va='top', family='monospace', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
ax4.axis('off')

plt.tight_layout()
plt.savefig(output_dir / 'ch5_08_error_analysis.png', bbox_inches='tight')
plt.close()
print("   ✅ ch5_08_error_analysis.png")

# ============== VIZ 5: ALL MODELS COMPARISON ==============
print("📊 Creating Model Comparison...")
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Model Comparison Summary', fontsize=14, fontweight='bold')

model_names = list(results.keys())
accs = [results[m]['test_accuracy'] for m in model_names]
f1s = [results[m]['test_f1'] for m in model_names]
precs = [results[m]['test_precision'] for m in model_names]
recs = [results[m]['test_recall'] for m in model_names]
times = [results[m]['training_time'] for m in model_names]

# All metrics
ax1 = axes[0, 0]
x = np.arange(len(model_names))
w = 0.2
ax1.bar(x-1.5*w, accs, w, label='Accuracy', color='steelblue')
ax1.bar(x-0.5*w, f1s, w, label='F1', color='coral')
ax1.bar(x+0.5*w, precs, w, label='Precision', color='green')
ax1.bar(x+1.5*w, recs, w, label='Recall', color='purple')
ax1.set_xticks(x)
ax1.set_xticklabels(model_names, fontsize=9)
ax1.set_ylabel('Score')
ax1.set_title('All Metrics')
ax1.legend(fontsize=8)
ax1.set_ylim([0, 1.05])
ax1.grid(axis='y', alpha=0.3)

# Training time
ax2 = axes[0, 1]
colors = ['green' if t<60 else 'orange' if t<300 else 'red' for t in times]
ax2.bar(model_names, times, color=colors, alpha=0.7)
ax2.set_ylabel('Time (seconds)')
ax2.set_title('Training Time')
for i, t in enumerate(times):
    ax2.text(i, t, f'{t:.1f}s', ha='center', va='bottom', fontsize=9)
ax2.grid(axis='y', alpha=0.3)

# Accuracy bar
ax3 = axes[1, 0]
colors = plt.cm.viridis(np.linspace(0, 1, len(model_names)))
bars = ax3.barh(model_names, accs, color=colors, alpha=0.7)
ax3.set_xlabel('Accuracy')
ax3.set_title('Test Accuracy Comparison')
ax3.set_xlim([0, 1.05])
for i, a in enumerate(accs):
    ax3.text(a+0.01, i, f'{a:.4f}', va='center', fontsize=10, fontweight='bold')
ax3.grid(axis='x', alpha=0.3)

# Winner summary
ax4 = axes[1, 1]
best_acc_m = model_names[np.argmax(accs)]
best_f1_m = model_names[np.argmax(f1s)]
fastest_m = model_names[np.argmin(times)]

winner = f"""
🏆 MODEL COMPARISON RESULTS

BEST BY ACCURACY:
  {best_acc_m}
  Accuracy: {max(accs):.4f}

BEST BY F1-SCORE:
  {best_f1_m}
  F1: {max(f1s):.4f}

FASTEST:
  {fastest_m}
  Time: {min(times):.1f}s

━━━━━━━━━━━━━━━━━━━━━━━━━

RECOMMENDATION:
  Use {best_model_name} for deployment
  
  • High accuracy ({results[best_model_name]['test_accuracy']:.4f})
  • Balanced metrics
  • Good generalization
"""
ax4.text(0.05, 0.95, winner, transform=ax4.transAxes, fontsize=10,
         va='top', family='monospace', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
ax4.axis('off')

plt.tight_layout()
plt.savefig(output_dir / 'ch5_09_model_comparison.png', bbox_inches='tight')
plt.close()
print("   ✅ ch5_09_model_comparison.png")

# ============== VIZ 6: FINAL DASHBOARD ==============
print("📊 Creating Final Dashboard...")
fig = plt.figure(figsize=(20, 12))
fig.suptitle('PAMAP2 Activity Recognition - Final Results Dashboard', 
             fontsize=18, fontweight='bold', y=0.98)

# Layout
gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)

# KPIs
kpis = [
    ('Best Model', best_model_name, 'steelblue'),
    ('Accuracy', f'{best_acc:.2%}', 'green'),
    ('F1-Score', f"{results[best_model_name]['test_f1']:.4f}", 'coral'),
    ('Classes', str(n_classes), 'purple')
]

for i, (label, value, color) in enumerate(kpis):
    ax = fig.add_subplot(gs[0, i])
    ax.text(0.5, 0.6, str(value), ha='center', va='center', fontsize=20, fontweight='bold', color=color)
    ax.text(0.5, 0.25, label, ha='center', va='center', fontsize=12, color='gray')
    ax.axis('off')

# Mini confusion matrix
ax_cm = fig.add_subplot(gs[1:, 0:2])
sns.heatmap(cm_norm, cmap='RdYlGn', vmin=0, vmax=1, ax=ax_cm,
           xticklabels=activity_names, yticklabels=activity_names, cbar_kws={'shrink':0.8})
ax_cm.set_title('Confusion Matrix (Normalized)', fontsize=12, fontweight='bold')
plt.setp(ax_cm.get_xticklabels(), rotation=45, ha='right', fontsize=6)
plt.setp(ax_cm.get_yticklabels(), fontsize=6)

# F1 by activity
ax_f1 = fig.add_subplot(gs[1, 2:])
colors = ['red' if f<0.7 else 'orange' if f<0.85 else 'green' for f in f1_scores]
ax_f1.barh(activity_names, f1_scores, color=colors, alpha=0.7)
ax_f1.set_xlabel('F1-Score')
ax_f1.set_title('F1-Score by Activity', fontsize=12, fontweight='bold')
ax_f1.set_xlim([0, 1.05])
plt.setp(ax_f1.get_yticklabels(), fontsize=7)
ax_f1.grid(axis='x', alpha=0.3)

# Model comparison
ax_models = fig.add_subplot(gs[2, 2:])
x = np.arange(len(model_names))
ax_models.bar(x, accs, color=['gold' if m==best_model_name else 'steelblue' for m in model_names], alpha=0.7)
ax_models.set_xticks(x)
ax_models.set_xticklabels(model_names, fontsize=9)
ax_models.set_ylabel('Accuracy')
ax_models.set_title('Model Comparison (★ = Best)', fontsize=12, fontweight='bold')
ax_models.set_ylim([0, 1.05])
for i, a in enumerate(accs):
    marker = '★' if model_names[i] == best_model_name else ''
    ax_models.text(i, a+0.02, f'{a:.3f}{marker}', ha='center', fontsize=9)
ax_models.grid(axis='y', alpha=0.3)

plt.savefig(output_dir / 'ch5_10_final_dashboard.png', bbox_inches='tight')
plt.close()
print("   ✅ ch5_10_final_dashboard.png")

# ============== DONE ==============
print("\n" + "=" * 70)
print("✅ ALL ML VISUALIZATIONS COMPLETE!")
print("=" * 70)
print(f"\n📁 Saved in: {output_dir}/")
print("\nGenerated:")
print("  • ch5_05_confusion_matrix.png")
print("  • ch5_06_per_activity_performance.png")
print("  • ch5_07_feature_importance.png")
print("  • ch5_08_error_analysis.png")
print("  • ch5_09_model_comparison.png")
print("  • ch5_10_final_dashboard.png")
print("\n🎉 Total: 6 new visualizations!")
print("=" * 70)