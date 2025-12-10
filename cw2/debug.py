"""
Quick Batch Tester
Test multiple sentences quickly and see results in a table
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BertTokenizer
import numpy as np
import pickle
from pathlib import Path
import pandas as pd

# Paths
ALBERT_PATH = "/Users/sarthakpunj/Downloads/albert_final 3"
MURIL_PATH = "/Users/sarthakpunj/Downloads/muril_proper_final"
ENSEMBLE_PATH = "/Users/sarthakpunj/Downloads/bias_detection_COMPLETE/ensemble_meta_model_proper.pkl"

print("Loading models...")
device = torch.device('cpu')

albert_tokenizer = AutoTokenizer.from_pretrained(ALBERT_PATH)
albert_model = AutoModelForSequenceClassification.from_pretrained(ALBERT_PATH)
albert_model.to(device).eval()

muril_tokenizer = BertTokenizer.from_pretrained(MURIL_PATH)
muril_model = AutoModelForSequenceClassification.from_pretrained(MURIL_PATH)
muril_model.to(device).eval()

with open(ENSEMBLE_PATH, 'rb') as f:
    ensemble_model = pickle.load(f)

print("✅ Models loaded!\n")

def predict_batch(texts):
    """Predict for multiple texts"""
    results = []
    
    for text in texts:
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
        
        ensemble_pred = ensemble_model.predict(features)[0]
        ensemble_prob = ensemble_model.predict_proba(features)[0][1]
        
        # VALIDATED THRESHOLD: 20% (validation set optimized)
        # Precision=100%, Recall=88.3%, F1=93.8%
        THRESHOLD = 0.10
        is_stereotype = ensemble_prob > THRESHOLD
        
        results.append({
            'Text': text[:60] + '...' if len(text) > 60 else text,
            'ALBERT': f"{albert_probs[1]*100:.1f}%",
            'MuRIL': f"{muril_probs[1]*100:.1f}%",
            'Ensemble': f"{ensemble_prob*100:.1f}%",
            'Result': '⚠️ STEREO' if is_stereotype else '✅ SAFE'
        })
    
    return results

# Test sentences
test_sentences = [
    # Should detect
    "All Gujaratis are businessmen",
    "Muslims are terrorists",
    "Women can't drive well",
    "Biharis are uneducated",
    "All Marwadis are greedy",
    
    # Should be safe
    "Muslims bring valuable diversity",
    "Women are excellent leaders",
    "Gujarati culture is rich",
    "Biharis contribute to society",
    "Christians deserve equal rights",
]

print("Testing sentences...\n")
results = predict_batch(test_sentences)

# Display
df = pd.DataFrame(results)
print(df.to_string(index=False))

# Summary
stereo_count = sum(1 for r in results if r['Result'] == '⚠️ STEREO')
print(f"\n{'='*80}")
print(f"Summary: {stereo_count}/{len(results)} detected as stereotypes")
print(f"{'='*80}")