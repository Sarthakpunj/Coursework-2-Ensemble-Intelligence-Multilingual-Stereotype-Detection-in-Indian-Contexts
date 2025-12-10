"""
Overfitting Check: Evaluate on Train, Val, and Test
This will show if models are overfitting
"""

import torch
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BertTokenizer
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════

BASE_DIR = Path(__file__).parent

ALBERT_PATH = '/Users/sarthakpunj/Downloads/albert_final 3'
MURIL_PATH = '/Users/sarthakpunj/Downloads/muril_proper_final'
ENSEMBLE_PATH = '/Users/sarthakpunj/Downloads/bias_detection_COMPLETE/ensemble_meta_model_proper.pkl'

# All three datasets
TRAIN_DATA = '/Users/sarthakpunj/Downloads/bias_detection_COMPLETE/train_ULTIMATE.csv'
VAL_DATA = '/Users/sarthakpunj/Downloads/bias_detection_COMPLETE/val_ULTIMATE.csv'
TEST_DATA = '/Users/sarthakpunj/Downloads/bias_detection_COMPLETE/test_ULTIMATE.csv'

# ═══════════════════════════════════════════════════════════════
# LOAD MODELS
# ═══════════════════════════════════════════════════════════════

print("="*60)
print("LOADING MODELS")
print("="*60)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

print("\n1. Loading ALBERT...")
albert_tokenizer = AutoTokenizer.from_pretrained(str(ALBERT_PATH))
albert_model = AutoModelForSequenceClassification.from_pretrained(str(ALBERT_PATH))
albert_model.to(device).eval()
print(" ALBERT loaded")

print("\n2. Loading MuRIL...")
muril_tokenizer = BertTokenizer.from_pretrained(str(MURIL_PATH))
muril_model = AutoModelForSequenceClassification.from_pretrained(str(MURIL_PATH))
muril_model.to(device).eval()
print(" MuRIL loaded")

print("\n3. Loading Ensemble...")
with open(ENSEMBLE_PATH, 'rb') as f:
    ensemble_model = pickle.load(f)
print(" Ensemble loaded")

# ═══════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════

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
    
    # Feature vector (6 features)
    features = np.array([[
        albert_probs[0], albert_probs[1],
        muril_probs[0], muril_probs[1],
        albert_probs[1] - muril_probs[1],
        albert_probs[1] * muril_probs[1]
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


def evaluate_dataset(csv_path, dataset_name):
    """Evaluate models on a dataset"""
    print(f"\n{'='*60}")
    print(f"EVALUATING: {dataset_name}")
    print(f"{'='*60}")
    
    # Load data
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} samples")
    
    # Convert labels
    if df['label'].dtype == 'object':
        label_map = {'non-stereotype': 0, 'stereotype': 1}
        df['label'] = df['label'].map(label_map)
    
    y_true = df['label'].values
    
    # Get predictions
    albert_preds = []
    muril_preds = []
    ensemble_preds = []
    albert_probs = []
    muril_probs = []
    ensemble_probs = []
    
    print(f"\nProcessing {dataset_name} samples...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
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
    
    # Calculate metrics
    results = {}
    
    for model_name, preds, probs in [
        ('ALBERT', albert_preds, albert_probs),
        ('MuRIL', muril_preds, muril_probs),
        ('Ensemble', ensemble_preds, ensemble_probs)
    ]:
        acc = accuracy_score(y_true, preds)
        prec = precision_score(y_true, preds, zero_division=0)
        rec = recall_score(y_true, preds, zero_division=0)
        f1 = f1_score(y_true, preds, zero_division=0)
        
        try:
            auc = roc_auc_score(y_true, probs)
        except:
            auc = 0.0
        
        results[model_name] = {
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'auc': auc
        }
        
        print(f"\n{model_name}:")
        print(f"  Accuracy:  {acc:.4f} ({acc*100:.2f}%)")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall:    {rec:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        print(f"  ROC-AUC:   {auc:.4f}")
    
    return results

# ═══════════════════════════════════════════════════════════════
# EVALUATE ALL DATASETS
# ═══════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("OVERFITTING CHECK")
print("="*60)

all_results = {}

# Evaluate on all three sets
all_results['Train'] = evaluate_dataset(TRAIN_DATA, 'TRAIN SET')
all_results['Val'] = evaluate_dataset(VAL_DATA, 'VALIDATION SET')
all_results['Test'] = evaluate_dataset(TEST_DATA, 'TEST SET')

# ═══════════════════════════════════════════════════════════════
# OVERFITTING ANALYSIS
# ═══════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("OVERFITTING ANALYSIS")
print("="*60)

for model_name in ['ALBERT', 'MuRIL', 'Ensemble']:
    print(f"\n{'='*60}")
    print(f"{model_name}")
    print(f"{'='*60}")
    
    train_acc = all_results['Train'][model_name]['accuracy']
    val_acc = all_results['Val'][model_name]['accuracy']
    test_acc = all_results['Test'][model_name]['accuracy']
    
    train_f1 = all_results['Train'][model_name]['f1']
    val_f1 = all_results['Val'][model_name]['f1']
    test_f1 = all_results['Test'][model_name]['f1']
    
    print(f"\nAccuracy:")
    print(f"  Train: {train_acc:.4f} ({train_acc*100:.2f}%)")
    print(f"  Val:   {val_acc:.4f} ({val_acc*100:.2f}%)")
    print(f"  Test:  {test_acc:.4f} ({test_acc*100:.2f}%)")
    
    print(f"\nF1-Score:")
    print(f"  Train: {train_f1:.4f}")
    print(f"  Val:   {val_f1:.4f}")
    print(f"  Test:  {test_f1:.4f}")
    
    # Calculate gaps
    train_val_gap = abs(train_acc - val_acc) * 100
    val_test_gap = abs(val_acc - test_acc) * 100
    train_test_gap = abs(train_acc - test_acc) * 100
    
    print(f"\nGaps (Accuracy):")
    print(f"  Train - Val:  {train_val_gap:.2f}%")
    print(f"  Val - Test:   {val_test_gap:.2f}%")
    print(f"  Train - Test: {train_test_gap:.2f}%")
    
    # Overfitting verdict
    print(f"\nVerdict:")
    if train_test_gap < 2:
        print("   NO OVERFITTING - Excellent generalization!")
    elif train_test_gap < 5:
        print("  MINIMAL OVERFITTING - Good generalization")
    elif train_test_gap < 10:
        print("    MODERATE OVERFITTING - Some memorization")
    else:
        print("   SIGNIFICANT OVERFITTING - Poor generalization")
    
    if train_acc > test_acc:
        print(f"  Model performs {(train_acc - test_acc)*100:.2f}% better on training data")
    else:
        print(f"  Model generalizes well (test ≥ train)")

# ═══════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("SUMMARY")
print("="*60)

# Create comparison table
comparison_df = pd.DataFrame({
    'Model': ['ALBERT', 'MuRIL', 'Ensemble'] * 3,
    'Dataset': ['Train']*3 + ['Val']*3 + ['Test']*3,
    'Accuracy': [
        all_results['Train']['ALBERT']['accuracy'],
        all_results['Train']['MuRIL']['accuracy'],
        all_results['Train']['Ensemble']['accuracy'],
        all_results['Val']['ALBERT']['accuracy'],
        all_results['Val']['MuRIL']['accuracy'],
        all_results['Val']['Ensemble']['accuracy'],
        all_results['Test']['ALBERT']['accuracy'],
        all_results['Test']['MuRIL']['accuracy'],
        all_results['Test']['Ensemble']['accuracy']
    ],
    'F1-Score': [
        all_results['Train']['ALBERT']['f1'],
        all_results['Train']['MuRIL']['f1'],
        all_results['Train']['Ensemble']['f1'],
        all_results['Val']['ALBERT']['f1'],
        all_results['Val']['MuRIL']['f1'],
        all_results['Val']['Ensemble']['f1'],
        all_results['Test']['ALBERT']['f1'],
        all_results['Test']['MuRIL']['f1'],
        all_results['Test']['Ensemble']['f1']
    ]
})

print("\n" + comparison_df.to_string(index=False))

# Save results
OUTPUT_DIR = BASE_DIR / 'evaluation_results'
comparison_df.to_csv(OUTPUT_DIR / 'overfitting_check.csv', index=False)
print(f"\n Results saved to: {OUTPUT_DIR / 'overfitting_check.csv'}")

print("\n" + "="*60)
print("OVERFITTING CHECK COMPLETE!")
print("="*60)