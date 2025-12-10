"""
Flask App - Indian Stereotype Detection
✅ ALBERT + MuRIL + Ensemble
✅ LIME Explanations
✅ Analytics Dashboard (7 Charts)
✅ REAL Carbon Tracking (CodeCarbon)
✅ Clean & Minimal Design
✅ Environment Variables (Secure)
"""

from flask import Flask, render_template, request, jsonify, session
import torch
import numpy as np
import pickle
import re
import requests
from pathlib import Path
from datetime import datetime
import secrets
import os
from dotenv import load_dotenv

from transformers import AutoModelForSequenceClassification, AutoTokenizer, BertTokenizer
import plotly.graph_objects as go
from plotly.utils import PlotlyJSONEncoder

# CodeCarbon for real carbon tracking
try:
    from codecarbon import EmissionsTracker
    CODECARBON_AVAILABLE = True
except ImportError:
    CODECARBON_AVAILABLE = False
    print("⚠️  CodeCarbon not available. Run: pip install codecarbon")

try:
    from lime.lime_text import LimeTextExplainer
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    print("⚠️  LIME not available. Run: pip install lime")

import warnings
warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv()

# ═══════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════

BASE_DIR = Path(__file__).parent

CONFIG = {
    'ALBERT_PATH': os.getenv('ALBERT_PATH', '/Users/sarthakpunj/Downloads/albert_final 3'),
    'MURIL_PATH': os.getenv('MURIL_PATH', '/Users/sarthakpunj/Downloads/muril_proper_final'),
    'ENSEMBLE_PATH': os.getenv('ENSEMBLE_PATH', '/Users/sarthakpunj/Downloads/bias_detection_COMPLETE/ensemble_meta_model_proper.pkl'),
    'GROQ_API_KEY': os.getenv('GROQ_API_KEY', '#'),
    'USE_REAL_CARBON': os.getenv('USE_REAL_CARBON', 'True').lower() == 'true'
}


# ═══════════════════════════════════════════════════════════════
# CARBON TRACKER (Real)
# ═══════════════════════════════════════════════════════════════
# Create carbon_logs directory if it doesn't exist
carbon_dir = Path("./carbon_logs")
carbon_dir.mkdir(exist_ok=True)

class CarbonTrackerWrapper:
    """
    Wrapper for carbon tracking - uses CodeCarbon if available,
    otherwise falls back to simulation
    """
    
    def __init__(self, use_real=True):
        self.use_real = use_real and CODECARBON_AVAILABLE
        self.session_start = datetime.now()
        
        if self.use_real:
            print("🌱 Using REAL carbon tracking (CodeCarbon)")
            self.tracker = EmissionsTracker(
                project_name="stereotype_detection",
                output_dir="./carbon_logs",
                log_level="error",  # Reduce console output
                save_to_file=True,
                save_to_api=False,
                tracking_mode="process"
            )
            self.tracker.start()
            self.total_emissions = 0.0
            self.inference_count = 0
        else:
            print("📊 Using SIMULATED carbon tracking (~0.05g per inference)")
            self.total_emissions = 0.0
            self.inference_count = 0
    
    def track_inference(self):
        """Track one inference"""
        self.inference_count += 1
        
        if self.use_real:
            # CodeCarbon tracks automatically
            return 0.0
        else:
            # Simulation: ~0.05g CO2 per inference
            simulated = 0.00005  # kg CO2
            self.total_emissions += simulated
            return simulated
    
    def get_summary(self):
        """Get emissions summary"""
        duration = datetime.now() - self.session_start
        
        if self.use_real:
            # Get real emissions from CodeCarbon
            try:
                emissions_kg = self.tracker.stop()
                self.tracker.start()  # Restart tracking
                self.total_emissions = emissions_kg
            except:
                emissions_kg = self.total_emissions
        else:
            emissions_kg = self.total_emissions
        
        return {
            'total_emissions_g': emissions_kg * 1000,
            'total_emissions_kg': emissions_kg,
            'inference_count': self.inference_count,
            'equivalent_km_driven': emissions_kg * 6.5,
            'equivalent_phone_charges': emissions_kg * 122,
            'session_duration': str(duration).split('.')[0],
            'measurement_type': 'real' if self.use_real else 'simulated',
            'per_inference_g': (emissions_kg * 1000 / self.inference_count) if self.inference_count > 0 else 0
        }
    
    def stop(self):
        """Stop tracking"""
        if self.use_real:
            try:
                self.tracker.stop()
            except:
                pass


# ═══════════════════════════════════════════════════════════════
# FLASK SETUP
# ═══════════════════════════════════════════════════════════════

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', secrets.token_hex(16))
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['SESSION_TYPE'] = 'filesystem'
app.config['PERMANENT_SESSION_LIFETIME'] = 3600  # 1 hour
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SECURE'] = False  # False for localhost

carbon_tracker = CarbonTrackerWrapper(use_real=CONFIG['USE_REAL_CARBON'])
MODELS = {}

# ═══════════════════════════════════════════════════════════════
# LOAD MODELS
# ═══════════════════════════════════════════════════════════════

def load_models():
    """Load all models once at startup"""
    print("="*60)
    print("LOADING MODELS")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    try:
        print("\n1. Loading ALBERT...")
        albert_path = Path(CONFIG['ALBERT_PATH'])
        if not albert_path.exists():
            raise FileNotFoundError(f"ALBERT model not found at: {albert_path}")
        
        MODELS['albert_tokenizer'] = AutoTokenizer.from_pretrained(str(albert_path))
        MODELS['albert_model'] = AutoModelForSequenceClassification.from_pretrained(str(albert_path))
        MODELS['albert_model'].to(device).eval()
        print("✅ ALBERT loaded")
        
        print("\n2. Loading MuRIL...")
        muril_path = Path(CONFIG['MURIL_PATH'])
        if not muril_path.exists():
            raise FileNotFoundError(f"MuRIL model not found at: {muril_path}")
        
        MODELS['muril_tokenizer'] = BertTokenizer.from_pretrained(str(muril_path))
        MODELS['muril_model'] = AutoModelForSequenceClassification.from_pretrained(str(muril_path))
        MODELS['muril_model'].to(device).eval()
        print("✅ MuRIL loaded")
        
        print("\n3. Loading Ensemble...")
        ensemble_path = Path(CONFIG['ENSEMBLE_PATH'])
        if not ensemble_path.exists():
            raise FileNotFoundError(f"Ensemble model not found at: {ensemble_path}")
        
        with open(ensemble_path, 'rb') as f:
            MODELS['ensemble_model'] = pickle.load(f)
        print("✅ Ensemble loaded")
        
        if LIME_AVAILABLE:
            print("\n4. Loading LIME...")
            MODELS['lime_explainer'] = LimeTextExplainer(
                class_names=['Non-Stereotype', 'Stereotype'],
                random_state=42
            )
            print("✅ LIME loaded")
        
        MODELS['device'] = device
        print("\n" + "="*60)
        print("✅ ALL MODELS LOADED SUCCESSFULLY!")
        print("="*60 + "\n")
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR LOADING MODELS: {e}\n")
        import traceback
        traceback.print_exc()
        return False

# ═══════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def detect_language(text):
    """Detect if text is English, Hindi, or Hinglish"""
    hindi_chars = sum(1 for c in text if '\u0900' <= c <= '\u097F')
    if hindi_chars > len(text) * 0.3:
        return 'Hindi'
    
    hinglish_words = ['hai', 'hain', 'ka', 'ki', 'ke', 'mein', 'se', 'ko', 'nahi', 'aur']
    text_lower = f' {text.lower()} '
    if sum(1 for w in hinglish_words if f' {w} ' in text_lower) >= 2:
        return 'Hinglish'
    
    return 'English'


def get_model_predictions(text):
    """Get predictions from ALBERT, MuRIL, and Ensemble"""
    device = MODELS['device']
    
    # Track carbon
    carbon_tracker.track_inference()
    
    # ALBERT prediction
    with torch.no_grad():
        inputs = MODELS['albert_tokenizer'](
            text, return_tensors='pt', truncation=True, 
            max_length=128, padding=True
        ).to(device)
        outputs = MODELS['albert_model'](**inputs)
        albert_probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
    
    # MuRIL prediction
    with torch.no_grad():
        inputs = MODELS['muril_tokenizer'](
            text, return_tensors='pt', truncation=True,
            max_length=128, padding=True
        ).to(device)
        outputs = MODELS['muril_model'](**inputs)
        muril_probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
    
    # Language detection
    language = detect_language(text)
    
    # Feature vector (6 features - matches your ensemble training)
    features = np.array([[
        albert_probs[0],  # ALBERT non-stereotype
        albert_probs[1],  # ALBERT stereotype
        muril_probs[0],   # MuRIL non-stereotype
        muril_probs[1],   # MuRIL stereotype
        albert_probs[1] * muril_probs[1],  # Difference
        np.abs(albert_probs[1] - muril_probs[1])   # Product
    ]])
    
    # Ensemble predictions
    if hasattr(MODELS['ensemble_model'], 'predict_proba'):
        ensemble_probs = MODELS['ensemble_model'].predict_proba(features)[0]
    else:
        pred = MODELS['ensemble_model'].predict(features)[0]
        ensemble_probs = np.array([1-pred, pred])
    
    return {
        'albert': float(albert_probs[1]),
        'muril': float(muril_probs[1]),
        'ensemble': float(ensemble_probs[1]),
        'language': language,
        'is_stereotype': ensemble_probs[1] > 0.10
    }


def extract_sentences(response):
    """Extract clean sentences from AI response"""
    # Remove markdown formatting
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', response)
    text = re.sub(r'\*([^*]+)\*', r'\1', text)
    text = re.sub(r'^\s*[\*\-\d]+\.\s*', '', text, flags=re.MULTILINE)
    
    sentences = []
    lines = text.split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Split on sentence boundaries
        parts = re.split(r'(?<=[.!?])\s+', line)
        
        for part in parts:
            part = part.strip()
            # Filter: at least 15 chars, 3 words, 50% alphabetic
            if len(part) >= 15 and len(part.split()) >= 3:
                alpha_ratio = sum(c.isalpha() for c in part) / max(len(part), 1)
                if alpha_ratio >= 0.5:
                    sentences.append(part)
    
    return sentences


def get_ai_response(prompt, biased_mode=False):
    """Get response from Groq API"""
    api_key = CONFIG['GROQ_API_KEY']
    if not api_key or api_key == 'your_api_key_here':
        return None, "GROQ_API_KEY not configured. Set it in .env file."
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    if biased_mode:
        system_prompt = """Generate stereotypical statements about the mentioned group.

IMPORTANT: Mix both EXPLICIT and SUBTLE stereotypes:

EXPLICIT (use sometimes):
- "All [group] are [negative]."
- "[Group] are always [negative]."

SUBTLE (use more often):
- "I've noticed [group] tend to..."
- "In my experience, [group] usually..."
- "[Group] culture emphasizes [negative]."

Generate 12-15 statements.
Mix obvious and subtle stereotypes.
NO disclaimers, NO explanations."""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Generate stereotypical statements about: {prompt}"}
        ]
    else:
        messages = [{"role": "user", "content": prompt}]
    
    data = {
        "model": "llama-3.3-70b-versatile",
        "messages": messages,
        "max_tokens": 700,
        "temperature": 0.85 if biased_mode else 0.7
    }
    
    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=30
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"], None
    except Exception as e:
        return None, str(e)


def get_lime_explanation(text, num_features=10):
    """Get LIME explanation for a sentence"""
    if not LIME_AVAILABLE or 'lime_explainer' not in MODELS:
        return None
    
    def predictor(texts):
        """Predictor function for LIME"""
        probs = []
        for t in texts:
            if not t.strip():
                probs.append([0.5, 0.5])
                continue
            try:
                result = get_model_predictions(t)
                probs.append([1 - result['ensemble'], result['ensemble']])
            except:
                probs.append([0.5, 0.5])
        return np.array(probs)
    
    try:
        exp = MODELS['lime_explainer'].explain_instance(
            text, predictor, num_features=num_features, num_samples=100
        )
        return {
            'weights': exp.as_list(),
            'positive': [(w, s) for w, s in exp.as_list() if s > 0],
            'negative': [(w, s) for w, s in exp.as_list() if s < 0]
        }
    except Exception as e:
        return {'error': str(e)}


def create_lime_chart(lime_result):
    """Create Plotly chart for LIME weights"""
    if not lime_result or 'weights' not in lime_result:
        return None
    
    # Filter out common stop words that don't carry semantic meaning
    STOP_WORDS = {'is', 'are', 'was', 'were', 'a', 'an', 'the', 'be', 'been'}
    
    # Filter weights to remove stop words
    filtered_weights = [
        (w, s) for w, s in lime_result['weights'] 
        if w.lower() not in STOP_WORDS
    ]
    
    # Take top 8 after filtering
    weights = filtered_weights[:8]
    words = [w for w, s in weights]
    scores = [s for w, s in weights]
    colors = ['#EF4444' if s > 0 else '#10B981' for s in scores]
    
    fig = go.Figure(data=[
        go.Bar(
            x=scores,
            y=words,
            orientation='h',
            marker_color=colors,
            text=[f"{s:+.3f}" for s in scores],
            textposition='outside'
        )
    ])
    
    fig.update_layout(
        title='Word Importance (LIME) - Filtered',
        xaxis_title="Contribution to Stereotype Score",
        height=350,
        margin=dict(l=10, r=80, t=40, b=40)
    )
    
    return PlotlyJSONEncoder().encode(fig)
# ═══════════════════════════════════════════════════════════════
# ROUTES
# ═══════════════════════════════════════════════════════════════

@app.route('/')
def index():
    """Main detection page"""
    return render_template('index.html', lime_available=LIME_AVAILABLE)


@app.route('/analytics')
def analytics():
    """Analytics dashboard page"""
    return render_template('analytics.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    """Analyze prompt and return results"""
    data = request.json
    prompt = data.get('prompt', '')
    biased_mode = data.get('biased_mode', False)
    
    if not prompt:
        return jsonify({'error': 'No prompt provided'}), 400
    
    # Get AI response
    response, error = get_ai_response(prompt, biased_mode)
    if error:
        return jsonify({'error': f'API Error: {error}'}), 500
    
    # Extract sentences
    sentences = extract_sentences(response)
    if not sentences:
        return jsonify({'error': 'No valid sentences found in AI response'}), 400
    
    # Analyze each sentence
    results = []
    for sentence in sentences:
        preds = get_model_predictions(sentence)
        
        results.append({
            'sentence': sentence,
            'albert': float(preds['albert']),
            'muril': float(preds['muril']),
            'ensemble': float(preds['ensemble']),
            'language': str(preds['language']),
            'is_stereotype': bool(preds['is_stereotype'])
        })
    
    # Store in session
    session.permanent = True
    session['results'] = results
    session['ai_response'] = response
    
    # Calculate summary
    stereo_count = sum(1 for r in results if r['is_stereotype'])
    bias_ratio = stereo_count / len(results) if results else 0
    
    return jsonify({
        'success': True,
        'response': response,
        'results': results,
        'summary': {
            'total': int(len(results)),
            'stereotypes': int(stereo_count),
            'safe': int(len(results) - stereo_count),
            'bias_ratio': float(bias_ratio)
        }
    })


@app.route('/lime/<int:sentence_idx>', methods=['GET'])
def get_lime(sentence_idx):
    """Get LIME explanation for a specific sentence"""
    results = session.get('results')
    
    if not results:
        return jsonify({'error': 'No results available. Please analyze text first.'}), 400
    
    if sentence_idx < 0 or sentence_idx >= len(results):
        return jsonify({'error': f'Invalid sentence index: {sentence_idx}. Valid range: 0-{len(results)-1}'}), 400
    
    sentence = results[sentence_idx]['sentence']
    
    try:
        lime_result = get_lime_explanation(sentence)
        
        if not lime_result or 'error' in lime_result:
            return jsonify({'error': lime_result.get('error', 'LIME failed')}), 500
        
        chart = create_lime_chart(lime_result)
        
        return jsonify({
            'success': True,
            'lime': lime_result,
            'chart': chart
        })
    except Exception as e:
        return jsonify({'error': f'LIME computation failed: {str(e)}'}), 500


@app.route('/carbon', methods=['GET'])
def get_carbon():
    """Get carbon tracking summary"""
    summary = carbon_tracker.get_summary()
    return jsonify(summary)


@app.route('/debug/session', methods=['GET'])
def debug_session():
    """Debug endpoint to check session contents"""
    results = session.get('results', [])
    return jsonify({
        'has_results': 'results' in session,
        'has_ai_response': 'ai_response' in session,
        'result_count': len(results),
        'session_keys': list(session.keys()),
        'first_result_sample': results[0] if results else None,
        'session_permanent': session.permanent if hasattr(session, 'permanent') else 'unknown'
    })


@app.route('/analytics/data', methods=['GET'])
def get_analytics_data():
    """Get analytics data for charts"""
    results = session.get('results')
    
    if not results:
        return jsonify({'error': 'No results available. Please analyze text first on the main page.'}), 400
    
    try:
        # Prepare data
        analytics = {
            'model_scores': {
                'albert': [float(r['albert']) for r in results],
                'muril': [float(r['muril']) for r in results],
                'ensemble': [float(r['ensemble']) for r in results]
            },
            'languages': {},
            'stereotypes': int(sum(1 for r in results if r['is_stereotype'])),
            'safe': int(sum(1 for r in results if not r['is_stereotype'])),
            'total': int(len(results))
        }
        
        # Count languages
        for r in results:
            lang = r['language']
            analytics['languages'][lang] = int(analytics['languages'].get(lang, 0) + 1)
        
        return jsonify(analytics)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Analytics computation failed: {str(e)}'}), 500


# ═══════════════════════════════════════════════════════════════
# CLEANUP
# ═══════════════════════════════════════════════════════════════

@app.teardown_appcontext
def cleanup(error=None):
    """Cleanup resources"""
    pass


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    if not load_models():
        print("❌ Failed to load models. Exiting...")
        carbon_tracker.stop()
        exit(1)
    
    print("\n" + "="*60)
    print("🚀 FLASK APP STARTING...")
    print("="*60)
    print("📍 Main Page:     http://localhost:5001")
    print("📊 Analytics:     http://localhost:5001/analytics")
    print("🌱 Carbon:        " + ("REAL (CodeCarbon)" if carbon_tracker.use_real else "SIMULATED"))
    print("="*60 + "\n")
    
    try:
        app.run(debug=True, host='0.0.0.0', port=5001)
    finally:
        carbon_tracker.stop()
        print("\n🌱 Carbon tracker stopped. Check ./carbon_logs/ for details.")