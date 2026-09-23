import os
import re
import joblib
import pandas as pd
import numpy as np
import streamlit as st

# ==========================================
# Page Configuration & Styling
# ==========================================
st.set_page_config(
    page_title="Phishing URL Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for seamless Light Mode and Dark Mode support
st.markdown("""
<style>
    /* Adaptive typography using CSS variables */
    .main-title {
        font-size: 2.4rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        margin-bottom: 0.2rem;
        color: var(--text-color, #0F172A);
    }
    .sub-title {
        font-size: 1.05rem;
        color: var(--text-color, #475569);
        opacity: 0.85;
        margin-bottom: 1.4rem;
    }
    
    /* Result card default (Light Mode) */
    .result-card-safe {
        background-color: rgba(16, 185, 129, 0.12);
        border: 2px solid #10B981;
        border-radius: 12px;
        padding: 22px;
        margin-top: 15px;
        margin-bottom: 20px;
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.15);
    }
    .result-card-safe .card-title {
        color: #047857;
        font-size: 1.5rem;
        font-weight: 700;
        margin-top: 10px;
        margin-bottom: 6px;
    }
    .result-card-safe .card-desc {
        color: #065F46;
        font-size: 1.05rem;
        margin-bottom: 10px;
    }
    .result-card-safe .card-score {
        color: #047857;
        font-size: 1.15rem;
        font-weight: 700;
    }

    .result-card-danger {
        background-color: rgba(239, 68, 68, 0.12);
        border: 2px solid #EF4444;
        border-radius: 12px;
        padding: 22px;
        margin-top: 15px;
        margin-bottom: 20px;
        box-shadow: 0 4px 12px rgba(239, 68, 68, 0.15);
    }
    .result-card-danger .card-title {
        color: #B91C1C;
        font-size: 1.5rem;
        font-weight: 700;
        margin-top: 10px;
        margin-bottom: 6px;
    }
    .result-card-danger .card-desc {
        color: #991B1B;
        font-size: 1.05rem;
        margin-bottom: 10px;
    }
    .result-card-danger .card-score {
        color: #B91C1C;
        font-size: 1.15rem;
        font-weight: 700;
    }

    /* Badges */
    .badge-safe {
        background-color: #10B981;
        color: #FFFFFF !important;
        padding: 5px 12px;
        border-radius: 16px;
        font-weight: 600;
        font-size: 0.9rem;
        display: inline-block;
    }
    .badge-danger {
        background-color: #EF4444;
        color: #FFFFFF !important;
        padding: 5px 12px;
        border-radius: 16px;
        font-weight: 600;
        font-size: 0.9rem;
        display: inline-block;
    }

    /* Dark Mode Styling via OS preference or Streamlit theme setting */
    @media (prefers-color-scheme: dark) {
        .main-title {
            color: #F8FAFC !important;
        }
        .sub-title {
            color: #94A3B8 !important;
        }
        .result-card-safe {
            background-color: rgba(16, 185, 129, 0.18) !important;
            border-color: #34D399 !important;
        }
        .result-card-safe .card-title {
            color: #34D399 !important;
        }
        .result-card-safe .card-desc {
            color: #E2E8F0 !important;
        }
        .result-card-safe .card-score {
            color: #6EE7B7 !important;
        }

        .result-card-danger {
            background-color: rgba(239, 68, 68, 0.18) !important;
            border-color: #F87171 !important;
        }
        .result-card-danger .card-title {
            color: #F87171 !important;
        }
        .result-card-danger .card-desc {
            color: #E2E8F0 !important;
        }
        .result-card-danger .card-score {
            color: #FCA5A5 !important;
        }
    }

    /* Streamlit explicit dark theme overrides */
    [data-theme="dark"] .main-title, .stApp[data-theme="dark"] .main-title {
        color: #F8FAFC !important;
    }
    [data-theme="dark"] .sub-title, .stApp[data-theme="dark"] .sub-title {
        color: #94A3B8 !important;
    }
    [data-theme="dark"] .result-card-safe {
        background-color: rgba(16, 185, 129, 0.18) !important;
        border-color: #34D399 !important;
    }
    [data-theme="dark"] .result-card-safe .card-title {
        color: #34D399 !important;
    }
    [data-theme="dark"] .result-card-safe .card-desc {
        color: #E2E8F0 !important;
    }
    [data-theme="dark"] .result-card-safe .card-score {
        color: #6EE7B7 !important;
    }

    [data-theme="dark"] .result-card-danger {
        background-color: rgba(239, 68, 68, 0.18) !important;
        border-color: #F87171 !important;
    }
    [data-theme="dark"] .result-card-danger .card-title {
        color: #F87171 !important;
    }
    [data-theme="dark"] .result-card-danger .card-desc {
        color: #E2E8F0 !important;
    }
    [data-theme="dark"] .result-card-danger .card-score {
        color: #FCA5A5 !important;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# Feature Extraction Logic
# ==========================================
FEATURE_COLUMNS = [
    'url_length', 'domain_length', 'count_dots', 'count_hyphens',
    'count_at', 'count_double_slash', 'count_question', 'count_slash',
    'count_equal', 'count_http', 'has_ip'
]

IP_REGEX = re.compile(
    r'(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)'
)
DOMAIN_REGEX = re.compile(r'^(?:https?://)?([^/?#]+)')

def extract_lexical_features(raw_url: str) -> pd.DataFrame:
    """
    Extracts the exact 11 lexical features used during model training:
    1. url_length
    2. domain_length
    3. count_dots
    4. count_hyphens
    5. count_at
    6. count_double_slash
    7. count_question
    8. count_slash
    9. count_equal
    10. count_http
    11. has_ip
    """
    url_str = str(raw_url).strip()
    
    # 1. Total URL length
    url_len = len(url_str)
    
    # 2. Domain length
    match_domain = DOMAIN_REGEX.search(url_str)
    domain_str = match_domain.group(1) if match_domain else ""
    domain_len = len(domain_str)
    
    # 3-10. Special character counts
    dots = url_str.count('.')
    hyphens = url_str.count('-')
    ats = url_str.count('@')
    double_slashes = url_str.count('//')
    questions = url_str.count('?')
    slashes = url_str.count('/')
    equals = url_str.count('=')
    https = url_str.count('http')
    
    # 11. Presence of IP address
    has_ip = 1 if IP_REGEX.search(url_str) else 0
    
    feature_dict = {
        'url_length': url_len,
        'domain_length': domain_len,
        'count_dots': dots,
        'count_hyphens': hyphens,
        'count_at': ats,
        'count_double_slash': double_slashes,
        'count_question': questions,
        'count_slash': slashes,
        'count_equal': equals,
        'count_http': https,
        'has_ip': has_ip
    }
    
    return pd.DataFrame([feature_dict], columns=FEATURE_COLUMNS)

# ==========================================
# Load Saved Artifacts (Cached)
# ==========================================
@st.cache_resource
def load_artifacts(model_choice: str = "Random Forest (High Accuracy: 87.5%)"):
    model_paths = {
        "Random Forest (High Accuracy: 87.5%)": "models/random_forest.joblib",
        "XGBoost (High Precision: 90.7%)": "models/xgboost.joblib",
        "Decision Tree (F1-Score: 80.2%)": "models/decision_tree.joblib",
        "Logistic Regression (Linear Baseline)": "models/logistic_regression.joblib"
    }
    
    model_path = model_paths.get(model_choice, "models/random_forest.joblib")
    scaler_path = "models/scaler.joblib"
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler file not found: {scaler_path}")
        
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

# ==========================================
# Sidebar: Model Config & Context
# ==========================================
with st.sidebar:
    st.header("⚙️ Settings & Info")
    
    model_option = st.selectbox(
        "Select Model for Inference:",
        [
            "Random Forest (High Accuracy: 87.5%)",
            "XGBoost (High Precision: 90.7%)",
            "Decision Tree (F1-Score: 80.2%)",
            "Logistic Regression (Linear Baseline)"
        ],
        index=0,
        help="Choose between ensemble tree models, gradient boosting, and baseline linear classifier."
    )
    
    st.markdown("---")
    st.subheader("📊 Benchmark Specs")
    if "Random Forest" in model_option:
        st.markdown("""
        - **Model:** Random Forest (100 Trees)
        - **Accuracy:** 87.52%
        - **Precision:** 92.56%
        - **F1-Score:** 0.8102
        - **Status:** Recommended for Production
        """)
    elif "XGBoost" in model_option:
        st.markdown("""
        - **Model:** XGBoost Classifier
        - **Accuracy:** 86.53%
        - **Precision:** 90.69%
        - **F1-Score:** 0.7980
        - **Speed:** Ultra-fast gradient boosting
        """)
    elif "Decision Tree" in model_option:
        st.markdown("""
        - **Model:** Decision Tree Classifier
        - **Max Depth:** 15
        - **Accuracy:** 86.48%
        - **Recall:** 73.37%
        - **F1-Score:** 0.8019
        """)
    else:
        st.markdown("""
        - **Model:** Logistic Regression (Linear)
        - **Accuracy:** 72.08%
        - **Precision:** 72.48%
        - **F1-Score:** 0.5221
        - **Role:** Academic Baseline Classifier
        """)
        
    st.markdown("---")
    st.markdown("""
    **Dataset Overview:**
    - Integrated: 707,473 unique URLs
    - Cleaned & deduplicated
    - 80/20 Stratified train-test split
    - Deep Learning: Char 1D-CNN (92.98%) & Transformer (93.25%)
    """)

# Load artifacts
try:
    model, scaler = load_artifacts(model_option)
except Exception as e:
    st.error(f"Failed to load machine learning artifacts: {e}")
    st.stop()

# ==========================================
# Main Header
# ==========================================
st.markdown('<div class="main-title">🛡️ Phishing Website Detection System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Real-time machine learning detection of malicious and phishing URLs via lexical feature analysis.</div>', unsafe_allow_html=True)

# Initialize session state for URL input
if "url_input" not in st.session_state:
    st.session_state["url_input"] = ""

def set_sample_url(url_val: str):
    st.session_state["url_input"] = url_val

# Quick sample buttons
st.write("**Quick Test Samples:**")
col_s1, col_s2, col_s3 = st.columns(3)
with col_s1:
    st.button("✅ Safe: Wikipedia URL", on_click=set_sample_url, args=("https://en.wikipedia.org/wiki/Phishing",), use_container_width=True)
with col_s2:
    st.button("⚠️ Phishing: Fake PayPal Login", on_click=set_sample_url, args=("www.dghjdgf.com/paypal.co.uk/cycgi-bin/webscrcmd=_home-customer&nav=1/loading.php",), use_container_width=True)
with col_s3:
    st.button("⚠️ Phishing: Bare IP Address", on_click=set_sample_url, args=("http://192.168.1.100/secure/center/update/limit/login.html",), use_container_width=True)

# Input Box
user_url = st.text_input(
    "Enter Website URL to Inspect:",
    key="url_input",
    placeholder="e.g. https://www.google.com or suspicious-site.net/account/login.php?id=382",
    help="You can enter raw URLs with or without http/https prefixes."
)

col_check, col_fill = st.columns([1.2, 4])
check_clicked = col_check.button("🔍 Check URL", type="primary", use_container_width=True)

# ==========================================
# Prediction Execution & Results
# ==========================================
if check_clicked or (user_url.strip() != ""):
    cleaned_url = user_url.strip()
    if not cleaned_url:
        st.warning("Please type or paste a URL to inspect.")
    else:
        with st.spinner("Extracting lexical features and evaluating URL..."):
            # 1. Feature Extraction
            features_raw_df = extract_lexical_features(cleaned_url)
            
            # 2. Standardization
            features_scaled_arr = scaler.transform(features_raw_df)
            features_scaled_df = pd.DataFrame(features_scaled_arr, columns=FEATURE_COLUMNS)
            
            # 3. Model Prediction
            pred_class = int(model.predict(features_scaled_df)[0])
            
            # 4. Probability Score
            if hasattr(model, "predict_proba"):
                probas = model.predict_proba(features_scaled_df)[0]
                prob_legit = float(probas[0])
                prob_phish = float(probas[1])
            else:
                prob_phish = 1.0 if pred_class == 1 else 0.0
                prob_legit = 1.0 - prob_phish
                
        # Result Presentation
        if pred_class == 1:
            st.markdown(f"""
            <div class="result-card-danger">
                <span class="badge-danger">🚨 Malicious URL Detected</span>
                <div class="card-title">Phishing / Fraudulent Website</div>
                <div class="card-desc">
                    This link exhibits high-risk lexical indicators commonly associated with credential-harvesting or scam portals.
                </div>
                <div class="card-score">
                    Phishing Confidence Score: {prob_phish * 100:.1f}%
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.progress(prob_phish)
        else:
            st.markdown(f"""
            <div class="result-card-safe">
                <span class="badge-safe">✅ Legitimate URL</span>
                <div class="card-title">Safe / Legitimate Website</div>
                <div class="card-desc">
                    No abnormal lexical anomalies or suspicious pattern tokens were found in this URL.
                </div>
                <div class="card-score">
                    Legitimacy Confidence Score: {prob_legit * 100:.1f}%
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.progress(prob_legit)

        # Metrics Overview Cards
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Predicted Label", "Phishing (1)" if pred_class == 1 else "Legitimate (0)")
        m2.metric("Phishing Score", f"{prob_phish:.1%}")
        m3.metric("Legitimate Score", f"{prob_legit:.1%}")
        m4.metric("URL Length", f"{len(cleaned_url)} chars")

        # Extracted Features Inspection
        with st.expander("📊 Extracted Lexical Features Breakdown", expanded=True):
            st.markdown("The 11 lexical features extracted from the input URL and supplied to the model:")
            
            display_df = features_raw_df.copy()
            display_df.rename(columns={
                'url_length': 'URL Length',
                'domain_length': 'Domain Length',
                'count_dots': 'Dots (.)',
                'count_hyphens': 'Hyphens (-)',
                'count_at': 'At (@)',
                'count_double_slash': 'Double Slash (//)',
                'count_question': 'Question Mark (?)',
                'count_slash': 'Slash (/)',
                'count_equal': 'Equal (=)',
                'count_http': 'HTTP Count',
                'has_ip': 'Has IP Address'
            }, inplace=True)
            
            st.dataframe(display_df, use_container_width=True)
            
            st.markdown("""
            > **Lexical Risk Factors:**
            > - **Domain & URL Length:** Long obscure URLs often conceal subdomains or redirect parameters.
            > - **Special Characters (`@`, `//`, `=`, `?`):** Often used in phishing schemes to mimic trusted brand URLs or embed target redirect paths.
            > - **IP Address Usage:** Phishing pages hosted on compromised machines often use raw numerical IP addresses rather than registered domain names.
            """)
