#  Ensemble Intelligence: Multilingual Stereotype Detection in Indian Contexts  
### **ALBERT + MuRIL + Meta-Ensemble | 95.8% Accuracy | 100% Precision | Hindi + English + Hinglish**

A multilingual, culturally-grounded stereotype detection system designed for **Indian social contexts**.  
Powered by an ensemble of **fine-tuned ALBERT**, **Google’s MuRIL**, and a **meta-classifier** that fuses their predictions for highly reliable bias detection.

---

## **Overview**

Large Language Models (LLMs) often generate or reinforce harmful stereotypes—especially in underrepresented regions like India.  
This project builds a **production-ready**, **explainable**, and **multilingual** system that detects stereotypes across:

- **English**
- **Hindi**
- **Hinglish/code-mixed text**

✔ 95.8% Accuracy  
✔ 100% Precision (zero false positives)  
✔ +15.8% improvement over original ALBERT baseline  
✔ Culturally grounded detection for **44+ Indian social groups**

---

## **Model Architecture**

The final system is an **ensemble**, combining:

1. **ALBERT-v2**  
   - Strong performance for English & Hinglish  
   - Lightweight & carbon-efficient  

2. **MuRIL (Multilingual Representations for Indian Languages)**  
   - Powerful for Hindi + mixed-language content  

3. **Meta-Ensemble Classifier**  
   - Uses 6 engineered features including:  
     - ALBERT prob  
     - MuRIL prob  
     - Product (agreement)  
     - Absolute difference (disagreement)  
   - Learns when to trust each model  
   - Achieves **100% precision**
  
   <img width="1031" height="826" alt="image" src="https://github.com/user-attachments/assets/3f4bff2e-901b-4248-9ffc-173164070856" />



---

##  **Datasets**

A unified dataset of **50,000 samples**, balanced:

- **25,000 stereotypes**
- **25,000 safe statements**

### Sources:
| Dataset | Share |
|--------|-------|
| SPICE | 42% |
| Template-generated | 28% |
| IndiBias (Gender, Religion, Caste) | 29% |
| Others | 1% |

**Languages:** Hindi, English, Hinglish  
**Split:** 70% train / 15% validation / 15% test  
**Difficulty:** 56% medium, 29% hard, 16% easy  

---

##  **Performance**

| Model | Accuracy | Precision | Notes |
|-------|----------|-----------|-------|
| ALBERT | 95.6% | ~99% | Efficient & stable |
| MuRIL | 95.6% | ~97% | Best for Hindi |
| **Ensemble** | **95.8%** | **100%** | No false positives |

### Improvements
- +15.8% over ALBERT baseline (80% macro-F1 from original paper)
- <0.01% overfitting gap
- Highly carbon-efficient (0.05g CO₂/inference)

---

##  **Flask Web Application**

The repository includes a **full interactive web UI** built with Flask:

### ✔ Features:
- Real-time predictions from ALBERT, MuRIL & Ensemble  
- Trigger word detection  
- **LIME Explainability** (word-level importance)  
- Bias heatmaps  
- Gauge charts for model confidence  
- Carbon emission tracker  
- Groq LLaMA-3.3 integration to generate test sentences  
- Clean HTML/CSS/JS interface  

---

##  **Installation**

```bash
git clone https://github.com/Sarthakpunj/Coursework-2-Ensemble-Intelligence-Multilingual-Stereotype-Detection
cd cw2

pip install -r requirements.txt
export GROQ_API_KEY="your_groq_key_here"

python3 app.py
