

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BertTokenizer
import numpy as np
import pickle
from pathlib import Path
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt

# Paths
ALBERT_PATH = "/Users/sarthakpunj/Downloads/albert_final 3"
MURIL_PATH = "/Users/sarthakpunj/Downloads/muril_proper_final"
ENSEMBLE_PATH = "/Users/sarthakpunj/Downloads/bias_detection_COMPLETE/ensemble_meta_model_proper.pkl"

print("="*60)
print("THRESHOLD OPTIMIZATION")
print("="*60)

# Load models
print("\nLoading models...")
device = torch.device('cpu')

albert_tokenizer = AutoTokenizer.from_pretrained(ALBERT_PATH)
albert_model = AutoModelForSequenceClassification.from_pretrained(ALBERT_PATH)
albert_model.to(device).eval()

muril_tokenizer = BertTokenizer.from_pretrained(MURIL_PATH)
muril_model = AutoModelForSequenceClassification.from_pretrained(MURIL_PATH)
muril_model.to(device).eval()

with open(ENSEMBLE_PATH, 'rb') as f:
    ensemble_model = pickle.load(f)

print(" Models loaded")

def get_ensemble_prob(text):
    """Get ensemble probability for a text"""
    # ALBERT
    with torch.no_grad():
        inputs = albert_tokenizer(text, return_tensors='pt', 
                                 padding=True, truncation=True, 
                                 max_length=128).to(device)
        albert_probs = torch.softmax(albert_model(**inputs).logits, dim=1).cpu().numpy()[0]
    
    # MuRIL
    with torch.no_grad():
        inputs = muril_tokenizer(text, return_tensors='pt',
                               padding=True, truncation=True,
                               max_length=128).to(device)
        muril_probs = torch.softmax(muril_model(**inputs).logits, dim=1).cpu().numpy()[0]
    
    # Ensemble
    features = np.array([[
        albert_probs[0], albert_probs[1],
        muril_probs[0], muril_probs[1],
        albert_probs[1] * muril_probs[1],
        np.abs(albert_probs[1] - muril_probs[1])
    ]])
    
    return ensemble_model.predict_proba(features)[0][1]

# Load validation set
print("\nLoading validation set...")
val_df = pd.read_csv('/Users/sarthakpunj/Downloads/datasets/val_ULTIMATE.csv')
print(f"Validation set: {len(val_df)} samples")

# Get predictions (sample for speed - adjust if you want full validation)
print("\nGetting predictions (sampling 500 for speed)...")
sample_df = val_df.sample(n=min(500, len(val_df)), random_state=42)

y_true = sample_df['label_id'].values
y_probs = []

for i, text in enumerate(sample_df['text'].values):
    if i % 100 == 0:
        print(f"  Progress: {i}/{len(sample_df)}")
    prob = get_ensemble_prob(text)
    y_probs.append(prob)

y_probs = np.array(y_probs)

print(" Predictions obtained")

# Test different thresholds
print("\n" + "="*60)
print("THRESHOLD ANALYSIS")
print("="*60)

thresholds = [0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30, 0.40, 0.50]

results = []

print(f"\n{'Threshold':<12} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Accuracy':<12}")
print("-" * 60)

for threshold in thresholds:
    y_pred = (y_probs > threshold).astype(int)
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary', zero_division=0
    )
    
    accuracy = (y_pred == y_true).mean()
    
    results.append({
        'threshold': threshold,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy
    })
    
    print(f"{threshold:<12.2f} {precision:<12.3f} {recall:<12.3f} {f1:<12.3f} {accuracy:<12.3f}")

# Find best threshold by F1
best_f1 = max(results, key=lambda x: x['f1'])
best_acc = max(results, key=lambda x: x['accuracy'])

print("\n" + "="*60)
print("RECOMMENDATIONS")
print("="*60)

print(f"\n Best F1-Score:")
print(f"   Threshold: {best_f1['threshold']:.2f}")
print(f"   Precision: {best_f1['precision']:.3f}")
print(f"   Recall:    {best_f1['recall']:.3f}")
print(f"   F1:        {best_f1['f1']:.3f}")
print(f"   Accuracy:  {best_f1['accuracy']:.3f}")

print(f"\n Best Accuracy:")
print(f"   Threshold: {best_acc['threshold']:.2f}")
print(f"   Precision: {best_acc['precision']:.3f}")
print(f"   Recall:    {best_acc['recall']:.3f}")
print(f"   F1:        {best_acc['f1']:.3f}")
print(f"   Accuracy:  {best_acc['accuracy']:.3f}")

# Plot
print("\n Creating visualization...")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: Precision/Recall/F1 vs Threshold
ax = axes[0, 0]
thresh_vals = [r['threshold'] for r in results]
prec_vals = [r['precision'] for r in results]
rec_vals = [r['recall'] for r in results]
f1_vals = [r['f1'] for r in results]

ax.plot(thresh_vals, prec_vals, 'o-', label='Precision', linewidth=2)
ax.plot(thresh_vals, rec_vals, 's-', label='Recall', linewidth=2)
ax.plot(thresh_vals, f1_vals, '^-', label='F1-Score', linewidth=2)
ax.axvline(x=best_f1['threshold'], color='red', linestyle='--', label=f'Best F1 ({best_f1["threshold"]:.2f})')
ax.set_xlabel('Threshold')
ax.set_ylabel('Score')
ax.set_title('Precision/Recall/F1 vs Threshold')
ax.legend()
ax.grid(alpha=0.3)

# Plot 2: Accuracy vs Threshold
ax = axes[0, 1]
acc_vals = [r['accuracy'] for r in results]
ax.plot(thresh_vals, acc_vals, 'o-', linewidth=2, color='purple')
ax.axvline(x=best_acc['threshold'], color='red', linestyle='--', label=f'Best Acc ({best_acc["threshold"]:.2f})')
ax.set_xlabel('Threshold')
ax.set_ylabel('Accuracy')
ax.set_title('Accuracy vs Threshold')
ax.legend()
ax.grid(alpha=0.3)

# Plot 3: Confusion Matrix at Best Threshold
ax = axes[1, 0]
y_pred_best = (y_probs > best_f1['threshold']).astype(int)
cm = confusion_matrix(y_true, y_pred_best)
im = ax.imshow(cm, cmap='Blues')
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(['Safe', 'Stereotype'])
ax.set_yticklabels(['Safe', 'Stereotype'])
ax.set_xlabel('Predicted')
ax.set_ylabel('True')
ax.set_title(f'Confusion Matrix (Threshold={best_f1["threshold"]:.2f})')

for i in range(2):
    for j in range(2):
        text = ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max()/2 else "black", fontsize=20)

plt.colorbar(im, ax=ax)

# Plot 4: Probability Distribution
ax = axes[1, 1]
stereo_probs = y_probs[y_true == 1]
safe_probs = y_probs[y_true == 0]

ax.hist(safe_probs, bins=20, alpha=0.5, label='Safe', color='green')
ax.hist(stereo_probs, bins=20, alpha=0.5, label='Stereotype', color='red')
ax.axvline(x=best_f1['threshold'], color='blue', linestyle='--', linewidth=2, label=f'Threshold ({best_f1["threshold"]:.2f})')
ax.set_xlabel('Ensemble Probability')
ax.set_ylabel('Count')
ax.set_title('Probability Distribution')
ax.legend()

plt.tight_layout()
plt.savefig('/Users/sarthakpunj/Desktop/cw2/threshold_analysis.png', dpi=300)
print(" Saved: threshold_analysis.png")

plt.show()

print("\n" + "="*60)
print(f"RECOMMENDED THRESHOLD: {best_f1['threshold']:.2f}")
print("="*60)
print(f"\nThis threshold provides the best F1-score balance.")
print(f"Update your code to use: THRESHOLD = {best_f1['threshold']:.2f}")
