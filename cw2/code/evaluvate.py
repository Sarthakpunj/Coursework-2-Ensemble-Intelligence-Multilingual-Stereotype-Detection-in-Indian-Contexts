"""
Model Evaluation Script
Evaluates ALBERT, MuRIL, and Ensemble models
Generates accuracy metrics and visualizations
"""

import torch
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BertTokenizer
from tqdm import tqdm

# ═══════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════

# Update these paths to match your repo structure
BASE_DIR = Path(__file__).parent
ALBERT_PATH = '/Users/sarthakpunj/Downloads/albert_final 3'
MURIL_PATH = '/Users/sarthakpunj/Downloads/muril_proper_final'
ENSEMBLE_PATH = '/Users/sarthakpunj/Downloads/bias_detection_COMPLETE/ensemble_meta_model_proper.pkl'
TEST_DATA = '/Users/sarthakpunj/Downloads/bias_detection_COMPLETE/test_ULTIMATE.csv'

# Output directory for results
OUTPUT_DIR = BASE_DIR / 'evaluation_results'
OUTPUT_DIR.mkdir(exist_ok=True)

# ═══════════════════════════════════════════════════════════════
# LOAD MODELS
# ═══════════════════════════════════════════════════════════════

print("="*60)
print("LOADING MODELS")
print("="*60)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ALBERT
print("\n1. Loading ALBERT...")
albert_tokenizer = AutoTokenizer.from_pretrained(str(ALBERT_PATH))
albert_model = AutoModelForSequenceClassification.from_pretrained(str(ALBERT_PATH))
albert_model.to(device).eval()
print(" ALBERT loaded")

# MuRIL
print("\n2. Loading MuRIL...")
muril_tokenizer = BertTokenizer.from_pretrained(str(MURIL_PATH))
muril_model = AutoModelForSequenceClassification.from_pretrained(str(MURIL_PATH))
muril_model.to(device).eval()
print(" MuRIL loaded")

# Ensemble
print("\n3. Loading Ensemble...")
with open(ENSEMBLE_PATH, 'rb') as f:
    ensemble_model = pickle.load(f)
print(" Ensemble loaded")

# ═══════════════════════════════════════════════════════════════
# LOAD TEST DATA
# ═══════════════════════════════════════════════════════════════

print(f"\n4. Loading test data from: {TEST_DATA}")
df_test = pd.read_csv(TEST_DATA)
print(f" Loaded {len(df_test)} test samples")

# Convert labels to integers if they're strings
if df_test['label'].dtype == 'object':
    print("\nConverting string labels to integers...")
    
    # Auto-detect the correct mapping
    unique_labels = df_test['label'].unique()
    print(f"Found labels: {unique_labels}")
    
    # Try different possible formats
    if 'non-stereotype' in unique_labels:
        label_map = {'non-stereotype': 0, 'stereotype': 1}
    elif 'non_stereotype' in unique_labels:
        label_map = {'non_stereotype': 0, 'stereotype': 1}
    elif 'Non-Stereotype' in unique_labels:
        label_map = {'Non-Stereotype': 0, 'Stereotype': 1}
    elif 'Non-stereotype' in unique_labels:
        label_map = {'Non-stereotype': 0, 'Stereotype': 1}
    else:
        # Fallback: assume first unique value is 0, second is 1
        label_map = {unique_labels[0]: 0, unique_labels[1]: 1}
        print(f"  Auto-detected mapping (verify this is correct!):")
    
    print(f"Label mapping: {label_map}")
    df_test['label'] = df_test['label'].map(label_map)
    
    # Verify conversion worked
    print(f" Labels converted. New unique values: {df_test['label'].unique()}")
    print(f"Label counts: {df_test['label'].value_counts().to_dict()}")
else:
    print(f"Labels are already numeric: {df_test['label'].unique()}")

# ═══════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def detect_language(text):
    """Detect language of text"""
    hindi_chars = sum(1 for c in str(text) if '\u0900' <= c <= '\u097F')
    if hindi_chars > len(str(text)) * 0.3:
        return 'Hindi'
    
    hinglish_words = ['hai', 'hain', 'ka', 'ki', 'ke', 'mein', 'se', 'ko']
    text_lower = f' {str(text).lower()} '
    if sum(1 for w in hinglish_words if f' {w} ' in text_lower) >= 2:
        return 'Hinglish'
    
    return 'English'


def get_predictions(text):
    """Get predictions from all models"""
    # ALBERT
    with torch.no_grad():
        inputs = albert_tokenizer(text, return_tensors='pt', truncation=True, 
                                  max_length=128, padding=True).to(device)
        outputs = albert_model(**inputs)
        albert_probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
    
    # MuRIL
    with torch.no_grad():
        inputs = muril_tokenizer(text, return_tensors='pt', truncation=True,
                                max_length=128, padding=True).to(device)
        outputs = muril_model(**inputs)
        muril_probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
    
    # Feature vector (6 features - ALBERT + MuRIL probabilities)
    # This matches your ensemble training
    features = np.array([[
        albert_probs[0],  # ALBERT non-stereotype
        albert_probs[1],  # ALBERT stereotype
        muril_probs[0],   # MuRIL non-stereotype
        muril_probs[1],   # MuRIL stereotype
        albert_probs[1] - muril_probs[1],  # Difference
        albert_probs[1] * muril_probs[1]   # Product
    ]])
    
    # Ensemble prediction
    if hasattr(ensemble_model, 'predict_proba'):
        ensemble_probs = ensemble_model.predict_proba(features)[0]
        ensemble_pred = 1 if ensemble_probs[1] > 0.5 else 0
    else:
        ensemble_pred = ensemble_model.predict(features)[0]
        ensemble_probs = np.array([1-ensemble_pred, ensemble_pred])
    
    return {
        'albert_pred': 1 if albert_probs[1] > 0.5 else 0,
        'albert_prob': albert_probs[1],
        'muril_pred': 1 if muril_probs[1] > 0.5 else 0,
        'muril_prob': muril_probs[1],
        'ensemble_pred': ensemble_pred,
        'ensemble_prob': ensemble_probs[1]
    }

# ═══════════════════════════════════════════════════════════════
# EVALUATE MODELS
# ═══════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("EVALUATING MODELS ON TEST SET")
print("="*60)

y_true = df_test['label'].values
albert_preds = []
muril_preds = []
ensemble_preds = []
albert_probs = []
muril_probs = []
ensemble_probs = []

print("\nProcessing test samples...")
for idx, row in tqdm(df_test.iterrows(), total=len(df_test)):
    text = row['text']
    preds = get_predictions(text)
    
    albert_preds.append(preds['albert_pred'])
    muril_preds.append(preds['muril_pred'])
    ensemble_preds.append(preds['ensemble_pred'])
    albert_probs.append(preds['albert_prob'])
    muril_probs.append(preds['muril_prob'])
    ensemble_probs.append(preds['ensemble_prob'])

# Convert to arrays
albert_preds = np.array(albert_preds)
muril_preds = np.array(muril_preds)
ensemble_preds = np.array(ensemble_preds)
albert_probs = np.array(albert_probs)
muril_probs = np.array(muril_probs)
ensemble_probs = np.array(ensemble_probs)

# ═══════════════════════════════════════════════════════════════
# CALCULATE METRICS
# ═══════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("METRICS")
print("="*60)

def calculate_metrics(y_true, y_pred, y_prob, model_name):
    """Calculate and print metrics"""
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # ROC-AUC
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    print(f"\n{model_name}:")
    print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  ROC-AUC:   {roc_auc:.4f}")
    
    return {
        'model': model_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'fpr': fpr,
        'tpr': tpr
    }

# Calculate metrics for all models
metrics = []
metrics.append(calculate_metrics(y_true, albert_preds, albert_probs, 'ALBERT'))
metrics.append(calculate_metrics(y_true, muril_preds, muril_probs, 'MuRIL'))
metrics.append(calculate_metrics(y_true, ensemble_preds, ensemble_probs, 'Ensemble'))

# Save metrics to CSV
metrics_df = pd.DataFrame([{
    'Model': m['model'],
    'Accuracy': m['accuracy'],
    'Precision': m['precision'],
    'Recall': m['recall'],
    'F1-Score': m['f1'],
    'ROC-AUC': m['roc_auc']
} for m in metrics])

metrics_df.to_csv(OUTPUT_DIR / 'metrics.csv', index=False)
print(f"\n Metrics saved to: {OUTPUT_DIR / 'metrics.csv'}")

# ═══════════════════════════════════════════════════════════════
# GENERATE VISUALIZATIONS
# ═══════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("GENERATING VISUALIZATIONS")
print("="*60)

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)

# 1. Metrics Comparison Bar Chart
print("\n1. Creating metrics comparison chart...")
fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(metrics))
width = 0.15

metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
colors = ['#3B82F6', '#8B5CF6', '#10B981', '#F59E0B', '#EF4444']

for i, metric in enumerate(metrics_to_plot):
    values = [m[metric] for m in metrics]
    ax.bar(x + i*width, values, width, label=metric.upper(), color=colors[i])
    
    # Add value labels
    for j, v in enumerate(values):
        ax.text(j + i*width, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=8)

ax.set_xlabel('Models', fontsize=12, fontweight='bold')
ax.set_ylabel('Score', fontsize=12, fontweight='bold')
ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
ax.set_xticks(x + width * 2)
ax.set_xticklabels([m['model'] for m in metrics])
ax.legend(loc='upper right')
ax.set_ylim(0, 1.1)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '1_metrics_comparison.png', dpi=300, bbox_inches='tight')
print(f" Saved: {OUTPUT_DIR / '1_metrics_comparison.png'}")
plt.close()

# 2. ROC Curves
print("\n2. Creating ROC curves...")
fig, ax = plt.subplots(figsize=(10, 8))

for i, m in enumerate(metrics):
    ax.plot(m['fpr'], m['tpr'], label=f"{m['model']} (AUC = {m['roc_auc']:.3f})", 
            linewidth=2, color=colors[i])

ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random (AUC = 0.500)')
ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
ax.set_title('ROC Curves - Model Comparison', fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=10)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '2_roc_curves.png', dpi=300, bbox_inches='tight')
print(f" Saved: {OUTPUT_DIR / '2_roc_curves.png'}")
plt.close()

# 3. Confusion Matrices
print("\n3. Creating confusion matrices...")
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for idx, (preds, model_name, ax) in enumerate(zip(
    [albert_preds, muril_preds, ensemble_preds],
    ['ALBERT', 'MuRIL', 'Ensemble'],
    axes
)):
    cm = confusion_matrix(y_true, preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False,
                xticklabels=['Non-Stereotype', 'Stereotype'],
                yticklabels=['Non-Stereotype', 'Stereotype'])
    ax.set_title(f'{model_name}', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=10)
    ax.set_xlabel('Predicted Label', fontsize=10)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '3_confusion_matrices.png', dpi=300, bbox_inches='tight')
print(f" Saved: {OUTPUT_DIR / '3_confusion_matrices.png'}")
plt.close()

# 4. Prediction Distribution
print("\n4. Creating prediction distribution...")
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for idx, (probs, model_name, ax) in enumerate(zip(
    [albert_probs, muril_probs, ensemble_probs],
    ['ALBERT', 'MuRIL', 'Ensemble'],
    axes
)):
    ax.hist(probs[y_true == 0], bins=30, alpha=0.6, label='Non-Stereotype', color='green')
    ax.hist(probs[y_true == 1], bins=30, alpha=0.6, label='Stereotype', color='red')
    ax.set_xlabel('Prediction Probability', fontsize=10)
    ax.set_ylabel('Count', fontsize=10)
    ax.set_title(f'{model_name}', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '4_prediction_distribution.png', dpi=300, bbox_inches='tight')
print(f" Saved: {OUTPUT_DIR / '4_prediction_distribution.png'}")
plt.close()

# 5. Model Agreement Analysis
print("\n5. Creating model agreement analysis...")
fig, ax = plt.subplots(figsize=(10, 6))

# Calculate agreement
all_agree_correct = ((albert_preds == muril_preds) & 
                     (muril_preds == ensemble_preds) & 
                     (ensemble_preds == y_true)).sum()
all_agree_wrong = ((albert_preds == muril_preds) & 
                   (muril_preds == ensemble_preds) & 
                   (ensemble_preds != y_true)).sum()
disagree = len(y_true) - all_agree_correct - all_agree_wrong

agreement_data = {
    'All Agree\n(Correct)': all_agree_correct,
    'All Agree\n(Wrong)': all_agree_wrong,
    'Disagree': disagree
}

colors_agree = ['#10B981', '#EF4444', '#F59E0B']
bars = ax.bar(agreement_data.keys(), agreement_data.values(), color=colors_agree)

# Add value labels
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height)}\n({height/len(y_true)*100:.1f}%)',
            ha='center', va='bottom', fontsize=11, fontweight='bold')

ax.set_ylabel('Number of Samples', fontsize=12, fontweight='bold')
ax.set_title('Model Agreement Analysis', fontsize=14, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '5_model_agreement.png', dpi=300, bbox_inches='tight')
print(f" Saved: {OUTPUT_DIR / '5_model_agreement.png'}")
plt.close()

# 6. Per-Class Performance
print("\n6. Creating per-class performance...")
fig, ax = plt.subplots(figsize=(10, 6))

models = ['ALBERT', 'MuRIL', 'Ensemble']
class_0_scores = []
class_1_scores = []

for preds in [albert_preds, muril_preds, ensemble_preds]:
    # Precision for each class
    cm = confusion_matrix(y_true, preds)
    class_0_prec = cm[0, 0] / (cm[0, 0] + cm[1, 0]) if (cm[0, 0] + cm[1, 0]) > 0 else 0
    class_1_prec = cm[1, 1] / (cm[0, 1] + cm[1, 1]) if (cm[0, 1] + cm[1, 1]) > 0 else 0
    class_0_scores.append(class_0_prec)
    class_1_scores.append(class_1_prec)

x = np.arange(len(models))
width = 0.35

ax.bar(x - width/2, class_0_scores, width, label='Non-Stereotype', color='#10B981')
ax.bar(x + width/2, class_1_scores, width, label='Stereotype', color='#EF4444')

# Add value labels
for i, (v0, v1) in enumerate(zip(class_0_scores, class_1_scores)):
    ax.text(i - width/2, v0 + 0.02, f'{v0:.3f}', ha='center', va='bottom', fontsize=9)
    ax.text(i + width/2, v1 + 0.02, f'{v1:.3f}', ha='center', va='bottom', fontsize=9)

ax.set_xlabel('Models', fontsize=12, fontweight='bold')
ax.set_ylabel('Precision', fontsize=12, fontweight='bold')
ax.set_title('Per-Class Precision Comparison', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend()
ax.set_ylim(0, 1.1)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '6_per_class_performance.png', dpi=300, bbox_inches='tight')
print(f" Saved: {OUTPUT_DIR / '6_per_class_performance.png'}")
plt.close()

# ═══════════════════════════════════════════════════════════════
# SUMMARY REPORT
# ═══════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("GENERATING SUMMARY REPORT")
print("="*60)

report = f"""
MODEL EVALUATION REPORT
{'='*60}

Dataset: {TEST_DATA.name}
Total Samples: {len(df_test)}
Non-Stereotypes: {(y_true == 0).sum()} ({(y_true == 0).sum()/len(y_true)*100:.1f}%)
Stereotypes: {(y_true == 1).sum()} ({(y_true == 1).sum()/len(y_true)*100:.1f}%)

{'='*60}
PERFORMANCE METRICS
{'='*60}

{metrics_df.to_string(index=False)}

{'='*60}
MODEL AGREEMENT
{'='*60}

All Models Agree (Correct): {all_agree_correct} ({all_agree_correct/len(y_true)*100:.1f}%)
All Models Agree (Wrong):   {all_agree_wrong} ({all_agree_wrong/len(y_true)*100:.1f}%)
Models Disagree:            {disagree} ({disagree/len(y_true)*100:.1f}%)

{'='*60}
BEST MODEL
{'='*60}

Based on F1-Score: {metrics_df.loc[metrics_df['F1-Score'].idxmax(), 'Model']}
F1-Score: {metrics_df['F1-Score'].max():.4f}

{'='*60}
GENERATED FILES
{'='*60}

1. metrics.csv - Detailed metrics
2. 1_metrics_comparison.png - Bar chart of all metrics
3. 2_roc_curves.png - ROC curves for all models
4. 3_confusion_matrices.png - Confusion matrices
5. 4_prediction_distribution.png - Probability distributions
6. 5_model_agreement.png - Agreement analysis
7. 6_per_class_performance.png - Per-class precision

All files saved in: {OUTPUT_DIR}

{'='*60}
"""

# Save report
with open(OUTPUT_DIR / 'evaluation_report.txt', 'w') as f:
    f.write(report)

print(report)
print(f" Report saved to: {OUTPUT_DIR / 'evaluation_report.txt'}")

print("\n" + "="*60)
print("EVALUATION COMPLETE!")
print("="*60)
print(f"\nAll results saved in: {OUTPUT_DIR}")
print("\nGenerated files:")
print("  • metrics.csv")
print("  • evaluation_report.txt")
print("  • 6 visualization images (.png)")