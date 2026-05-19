"""
CarPrice AI — versi Streamlit
Jalankan lokal: streamlit run streamlit_app.py
"""
import os
import warnings

import joblib
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import InconsistentVersionWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

import sklearn.compose._column_transformer as _ct_module
if not hasattr(_ct_module, "_RemainderColsList"):
    class _RemainderColsList(list):
        pass
    _ct_module._RemainderColsList = _RemainderColsList


# ============================================================================
st.set_page_config(
    page_title="CarPrice AI — Prediksi Harga Mobil",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================================================================
# SESSION STATE
# ============================================================================
if "theme" not in st.session_state:
    st.session_state.theme = "light"
if "last_price" not in st.session_state:
    st.session_state.last_price = None
if "predictions_count" not in st.session_state:
    st.session_state.predictions_count = 0
if "show_confetti" not in st.session_state:
    st.session_state.show_confetti = False


def is_dark():
    return st.session_state.theme == "dark"


# ============================================================================
# CSS
# ============================================================================
def inject_css():
    dark = is_dark()
    if dark:
        bg = "#070b1f"; bg_soft = "#0f1638"; surface = "#141b40"
        text = "#f1f4ff"; text_muted = "#c1c8ee"; border = "#2a3470"
        indigo_50 = "#1f2858"; indigo_100 = "#2a3470"; indigo_200 = "#4756a6"
        danger_bg = "#3b1418"; danger_fg = "#fca5a5"
        credit_bg = "linear-gradient(135deg, #1c2554, #2a3470)"
    else:
        bg = "#f5f7ff"; bg_soft = "#eef1fb"; surface = "#ffffff"
        text = "#0f1531"; text_muted = "#5b6391"; border = "#e1e5f5"
        indigo_50 = "#e8eaf6"; indigo_100 = "#c5cae9"; indigo_200 = "#9fa8da"
        danger_bg = "#fef2f2"; danger_fg = "#b91c1c"
        credit_bg = "linear-gradient(135deg, #e3f2fd, #f0f4ff)"

    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Space+Grotesk:wght@500;600;700&display=swap');

        :root {{
            --navy-950: #060d2c; --navy-900: #0a1545; --navy-800: #131f5e;
            --navy-700: #1a237e; --navy-600: #283593; --navy-500: #3949ab;
            --navy-400: #5c6bc0; --indigo-300: #7986cb;
            --indigo-200: {indigo_200}; --indigo-100: {indigo_100}; --indigo-50: {indigo_50};
            --bg: {bg}; --bg-soft: {bg_soft}; --surface: {surface};
            --text: {text}; --text-muted: {text_muted}; --border: {border};
            --danger-bg: {danger_bg}; --danger-fg: {danger_fg};
        }}

        html, body, .stApp, [data-testid="stAppViewContainer"] {{
            background: var(--bg) !important;
            color: var(--text) !important;
            font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
        }}
        header[data-testid="stHeader"] {{ background: transparent !important; }}
        #MainMenu, footer, [data-testid="stToolbar"] {{ visibility: hidden; }}
        .block-container {{ padding-top: 1.2rem !important; padding-bottom: 3rem !important; max-width: 1240px !important; }}

        h1, h2, h3, h4, p, span, label, div {{ color: var(--text); }}
        h1, h2, h3, h4 {{
            font-family: 'Space Grotesk', 'Plus Jakarta Sans', sans-serif !important;
            letter-spacing: -0.015em !important;
        }}

        /* ===== NAVBAR ===== */
        .navbar {{
            display: flex; align-items: center; justify-content: space-between;
            padding: 6px 0 14px;
        }}
        .logo {{
            display: inline-flex; align-items: center; gap: 10px;
            font-family: 'Space Grotesk', sans-serif;
            font-weight: 700; font-size: 16px;
        }}
        .logo-mark {{
            width: 32px; height: 32px; border-radius: 9px;
            background: linear-gradient(135deg, var(--navy-700), var(--navy-500));
            display: grid; place-items: center;
            color: white; font-size: 16px;
            box-shadow: 0 6px 16px rgba(26,35,126,0.35);
        }}
        .logo .ai-tag {{ color: var(--text-muted); font-weight: 500; }}

        /* ===== HERO ===== */
        .hero {{
            position: relative;
            background:
                radial-gradient(1200px 500px at 15% -10%, rgba(121,134,203,0.55), transparent 60%),
                radial-gradient(900px 500px at 95% 10%, rgba(57,73,171,0.7), transparent 60%),
                linear-gradient(135deg, var(--navy-950) 0%, var(--navy-800) 60%, var(--navy-600) 100%);
            color: white !important;
            padding: 56px 36px 56px;
            border-radius: 22px;
            overflow: hidden;
            margin-bottom: 24px;
            box-shadow: 0 30px 60px rgba(10,21,69,0.35);
        }}
        .hero::after {{
            content: ""; position: absolute; inset: 0;
            background-image:
                linear-gradient(rgba(255,255,255,0.05) 1px, transparent 1px),
                linear-gradient(90deg, rgba(255,255,255,0.05) 1px, transparent 1px);
            background-size: 44px 44px;
            mask-image: radial-gradient(circle at 50% 30%, black, transparent 70%);
            -webkit-mask-image: radial-gradient(circle at 50% 30%, black, transparent 70%);
            pointer-events: none;
        }}
        .hero > * {{ position: relative; z-index: 2; }}
        .brand {{
            display: inline-flex; align-items: center; gap: 8px;
            padding: 7px 14px;
            background: rgba(255,255,255,0.1);
            border: 1px solid rgba(255,255,255,0.2);
            backdrop-filter: blur(10px);
            border-radius: 999px;
            font-size: 12.5px; font-weight: 600;
            margin-bottom: 18px;
            color: white !important;
        }}
        .brand-dot {{
            width: 7px; height: 7px;
            background: #4ade80; border-radius: 50%;
            animation: pulse 2s infinite;
        }}
        .hero h1 {{
            font-size: clamp(28px, 4.4vw, 44px) !important;
            font-weight: 700 !important;
            line-height: 1.12 !important;
            color: white !important;
            margin: 0 0 14px !important;
        }}
        .hero .accent {{
            background: linear-gradient(90deg, #c7d2fe, #ffffff 55%, #a5b4fc);
            -webkit-background-clip: text; background-clip: text;
            color: transparent;
        }}
        .hero p.subtitle {{
            margin: 0 0 18px;
            max-width: 700px;
            font-size: 16px;
            color: rgba(255,255,255,0.82) !important;
        }}
        .stat-pill {{
            display: inline-flex; align-items: center; gap: 8px;
            padding: 9px 15px;
            background: rgba(255,255,255,0.08);
            border: 1px solid rgba(255,255,255,0.16);
            border-radius: 999px;
            font-size: 12.5px; font-weight: 500;
            color: white !important;
            margin: 4px 6px 0 0;
        }}

        /* ===== CARDS via st.container(border=True) =====
           Streamlit native border container -> styled as our card */
        [data-testid="stVerticalBlockBorderWrapper"] {{
            background: var(--surface) !important;
            border-radius: 20px !important;
            box-shadow: 0 30px 70px rgba(10,21,69,0.18) !important;
            border: 1px solid var(--border) !important;
            padding: 28px !important;
        }}
        /* Reduce inner gap inside cards a bit */
        [data-testid="stVerticalBlockBorderWrapper"] [data-testid="stVerticalBlock"] {{ gap: 0.6rem; }}

        /* Card header */
        .card-header {{
            display: flex; align-items: center; gap: 12px;
            margin-bottom: 4px;
        }}
        .card-icon {{
            width: 42px; height: 42px; border-radius: 12px;
            display: grid; place-items: center;
            background: linear-gradient(135deg, var(--navy-700), var(--navy-500));
            color: white !important; font-size: 20px;
            box-shadow: 0 8px 20px rgba(26,35,126,0.35);
            flex-shrink: 0;
        }}
        .card-title {{
            font-family: 'Space Grotesk', sans-serif;
            font-size: 22px; font-weight: 700;
            color: var(--text) !important;
            display: inline-flex; align-items: center; gap: 8px;
        }}
        .card-sub {{
            color: var(--text-muted) !important;
            font-size: 14px;
            margin: 4px 0 14px 54px;
        }}
        .live-badge {{
            display: inline-flex; align-items: center; gap: 6px;
            font-size: 10px; font-weight: 700; letter-spacing: 1.4px;
            padding: 3px 10px;
            background: linear-gradient(135deg, #16a34a, #22c55e);
            color: white !important;
            border-radius: 999px;
        }}
        .live-dot {{
            width: 6px; height: 6px;
            background: white; border-radius: 50%;
            animation: pulse 1.4s infinite;
        }}

        /* ===== PROGRESS ===== */
        .progress-row {{
            display: flex; align-items: center; gap: 12px;
            padding: 12px 14px;
            background: linear-gradient(135deg, var(--bg-soft), var(--bg));
            border: 1px solid var(--border);
            border-radius: 12px;
            margin: 10px 0 6px;
        }}
        .progress-bar {{
            flex: 1; height: 8px;
            background: var(--indigo-50);
            border-radius: 999px; overflow: hidden;
        }}
        .progress-fill {{
            height: 100%;
            background: linear-gradient(90deg, var(--navy-500), #5c6bc0, #16a34a);
            background-size: 200% 100%;
            animation: shimmer 3s linear infinite;
            border-radius: 999px;
        }}
        .progress-label {{
            font-size: 12px; font-weight: 700;
            color: var(--text-muted) !important;
            letter-spacing: 0.5px; white-space: nowrap;
        }}

        /* ===== STREAMLIT WIDGETS ===== */
        [data-testid="stNumberInput"] label,
        [data-testid="stSelectbox"] label {{
            color: var(--text) !important;
            font-weight: 600 !important;
            font-size: 13px !important;
        }}
        [data-testid="stNumberInput"] input,
        [data-testid="stTextInput"] input {{
            background: var(--bg) !important;
            color: var(--text) !important;
            border: 1.5px solid var(--border) !important;
            border-radius: 11px !important;
            font-weight: 500 !important;
        }}
        [data-testid="stNumberInput"] input:focus,
        [data-testid="stTextInput"] input:focus {{
            border-color: var(--navy-500) !important;
            box-shadow: 0 0 0 4px rgba(57, 73, 171, 0.18) !important;
        }}
        [data-testid="stNumberInput"] button {{
            background: var(--bg-soft) !important;
            color: var(--text-muted) !important;
            border-color: var(--border) !important;
        }}
        [data-testid="stNumberInput"] button:hover {{
            background: var(--indigo-50) !important;
            color: var(--navy-700) !important;
        }}
        [data-baseweb="select"] > div {{
            background: var(--bg) !important;
            border: 1.5px solid var(--border) !important;
            border-radius: 11px !important;
            color: var(--text) !important;
        }}
        [data-baseweb="popover"] {{ background: var(--surface) !important; }}
        [data-baseweb="select"] svg {{ color: var(--text-muted) !important; }}

        /* Buttons */
        .stButton > button {{
            border-radius: 12px !important;
            font-weight: 700 !important;
            font-family: 'Plus Jakarta Sans', sans-serif !important;
            padding: 10px 16px !important;
            transition: all 0.25s ease !important;
            border: none !important;
            white-space: nowrap !important;
        }}
        .stButton > button[kind="primary"] {{
            background: linear-gradient(135deg, var(--navy-700), var(--navy-500)) !important;
            color: white !important;
            box-shadow: 0 12px 28px rgba(26,35,126,0.34) !important;
        }}
        .stButton > button[kind="primary"]:hover {{
            transform: translateY(-2px);
            box-shadow: 0 16px 36px rgba(26,35,126,0.45) !important;
        }}
        .stButton > button[kind="secondary"] {{
            background: var(--bg-soft) !important;
            color: var(--text-muted) !important;
            border: 1px solid var(--border) !important;
        }}
        .stButton > button[kind="secondary"]:hover {{
            background: var(--indigo-50) !important;
            color: var(--navy-700) !important;
            transform: translateY(-2px);
        }}
        /* Theme toggle (icon-only) */
        .theme-toggle button {{
            width: 42px !important;
            height: 42px !important;
            padding: 0 !important;
            font-size: 18px !important;
            border-radius: 12px !important;
            min-width: 42px !important;
        }}

        /* ===== PRICE CARD ===== */
        .price-card {{
            position: relative;
            background:
                radial-gradient(800px 240px at 0% 0%, rgba(121,134,203,0.4), transparent 60%),
                linear-gradient(135deg, var(--navy-900), var(--navy-600));
            color: white !important;
            padding: 28px 26px;
            border-radius: 18px;
            margin: 14px 0 22px;
            overflow: hidden;
        }}
        .price-card::before {{
            content: "";
            position: absolute; top: -40%; right: -20%;
            width: 280px; height: 280px;
            background: radial-gradient(circle, rgba(255,255,255,0.18), transparent 60%);
            border-radius: 50%;
        }}
        .price-card > * {{ position: relative; z-index: 2; }}
        .price-label {{
            display: inline-flex; align-items: center; gap: 6px;
            font-size: 11px; font-weight: 700;
            letter-spacing: 1.5px; color: rgba(255,255,255,0.85) !important;
            text-transform: uppercase;
            background: rgba(255,255,255,0.12);
            padding: 5px 12px; border-radius: 999px;
        }}
        .price-value {{
            font-family: 'Space Grotesk', sans-serif;
            font-size: clamp(32px, 5vw, 44px);
            font-weight: 700;
            color: white !important;
            letter-spacing: -0.025em;
            margin-top: 12px; line-height: 1.1;
            font-variant-numeric: tabular-nums;
            animation: priceIn 0.6s cubic-bezier(0.16, 1, 0.3, 1);
        }}
        .price-note {{
            margin-top: 10px;
            font-size: 13px;
            color: rgba(255,255,255,0.72) !important;
        }}
        .price-trend {{
            display: inline-flex; align-items: center; gap: 4px;
            font-size: 12px; font-weight: 700;
            padding: 3px 10px;
            border-radius: 999px;
            margin-top: 10px;
        }}
        .price-trend.up   {{ background: rgba(74,222,128,0.22); color: #86efac !important; }}
        .price-trend.down {{ background: rgba(248,113,113,0.22); color: #fca5a5 !important; }}
        .price-trend.flat {{ background: rgba(255,255,255,0.16); color: white !important; }}

        /* ===== SUMMARY TABLE ===== */
        .summary-title {{
            font-family: 'Space Grotesk', sans-serif;
            font-size: 12px; font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 1.4px;
            color: var(--text-muted) !important;
            margin: 4px 0 10px;
        }}
        .summary-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 13.5px;
            border: 1px solid var(--border);
            border-radius: 12px;
            overflow: hidden;
            background: var(--surface);
        }}
        .summary-table tr {{ transition: background 0.2s; }}
        .summary-table tr:hover {{ background: var(--bg-soft); }}
        .summary-table td {{
            padding: 10px 14px;
            border-bottom: 1px solid var(--border);
        }}
        .summary-table tr:last-child td {{ border-bottom: none; }}
        .summary-table td:first-child {{
            color: var(--text-muted) !important;
            font-weight: 500;
        }}
        .summary-table td:last-child {{
            text-align: right;
            color: var(--text) !important;
            font-weight: 700;
            font-variant-numeric: tabular-nums;
        }}

        /* ===== CREDIT BOX ===== */
        .credit {{
            margin-top: 22px;
            padding: 18px 20px;
            background: {credit_bg};
            border-radius: 14px;
            border-left: 4px solid var(--navy-700);
            display: flex; gap: 14px; align-items: center;
        }}
        .credit-avatar {{
            width: 46px; height: 46px;
            border-radius: 50%;
            background: linear-gradient(135deg, var(--navy-700), var(--navy-500));
            color: white !important;
            display: grid; place-items: center;
            font-weight: 800;
            font-family: 'Space Grotesk', sans-serif;
            font-size: 16px;
            box-shadow: 0 6px 14px rgba(26,35,126,0.3);
            flex-shrink: 0;
        }}
        .credit-meta {{ font-size: 13px; line-height: 1.55; }}
        .credit-title {{
            font-size: 11px; text-transform: uppercase;
            letter-spacing: 1.4px; font-weight: 700;
            color: var(--text-muted) !important;
            margin-bottom: 3px;
        }}
        .credit-name {{ font-weight: 700; color: var(--text) !important; font-size: 14px; }}
        .credit-nim {{ color: var(--text-muted) !important; font-size: 13px; }}

        /* ===== ALERT ===== */
        [data-testid="stAlert"] {{
            background: var(--danger-bg) !important;
            color: var(--danger-fg) !important;
            border-left: 4px solid var(--danger-fg) !important;
            border-radius: 10px !important;
        }}
        [data-testid="stAlert"] p {{ color: var(--danger-fg) !important; }}

        /* ===== FOOTER ===== */
        .app-footer {{
            text-align: center;
            padding: 24px 0 8px;
            color: var(--text-muted) !important;
            font-size: 13px;
        }}

        /* ===== ANIMATIONS ===== */
        @keyframes pulse {{
            0%   {{ box-shadow: 0 0 0 0 rgba(74,222,128,0.6); }}
            70%  {{ box-shadow: 0 0 0 10px rgba(74,222,128,0); }}
            100% {{ box-shadow: 0 0 0 0 rgba(74,222,128,0); }}
        }}
        @keyframes shimmer {{
            0% {{ background-position: 0% 50%; }}
            100% {{ background-position: 200% 50%; }}
        }}
        @keyframes priceIn {{
            from {{ opacity: 0; transform: scale(0.96); }}
            to   {{ opacity: 1; transform: scale(1); }}
        }}

        @media (max-width: 980px) {{
            .hero {{ padding: 44px 22px; }}
            [data-testid="stVerticalBlockBorderWrapper"] {{ padding: 22px !important; }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ============================================================================
# LOAD MODEL
# ============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model_prediksi_harga_mobil.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "feature_columns.pkl")


def _find_first(estimator, target_cls):
    if isinstance(estimator, target_cls):
        return estimator
    if isinstance(estimator, Pipeline):
        for _, step in estimator.steps:
            found = _find_first(step, target_cls)
            if found is not None:
                return found
    return None


@st.cache_resource(show_spinner="Memuat model machine learning...")
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    features = list(joblib.load(FEATURES_PATH))
    if not isinstance(model, Pipeline):
        raise RuntimeError("Model bukan sklearn.Pipeline.")

    preprocessor = regressor = None
    for _, step in model.steps:
        if isinstance(step, ColumnTransformer) and preprocessor is None:
            preprocessor = step
        elif isinstance(step, LinearRegression) and regressor is None:
            regressor = step

    num_cols, cat_cols = [], []
    imputer = scaler = encoder = None
    transformers_iter = (
        getattr(preprocessor, "transformers_", None) or preprocessor.transformers
    )
    for _, trans, cols in transformers_iter:
        if trans in ("drop", "passthrough"):
            continue
        cols_list = list(cols) if not isinstance(cols, str) else [cols]
        enc = _find_first(trans, OneHotEncoder)
        if enc is not None:
            cat_cols = cols_list
            encoder = enc
        else:
            num_cols = cols_list
            imputer = _find_first(trans, SimpleImputer)
            scaler = _find_first(trans, StandardScaler)

    imp_stats = np.asarray(imputer.statistics_, dtype=float) if imputer is not None else None
    s_mean = np.asarray(scaler.mean_, dtype=float) if scaler is not None else None
    s_scale = np.asarray(scaler.scale_, dtype=float) if scaler is not None else None
    ohe_categories = [np.asarray(c) for c in encoder.categories_]
    ohe_drop = getattr(encoder, "drop_idx_", None)
    ohe_active = []
    for i, cats in enumerate(ohe_categories):
        drop = None
        if ohe_drop is not None:
            d = ohe_drop[i]
            if d is not None and not (isinstance(d, float) and np.isnan(d)):
                drop = int(d)
        ohe_active.append([j for j in range(len(cats)) if j != drop])

    coef = np.asarray(regressor.coef_, dtype=float).ravel()
    intercept = float(np.asarray(regressor.intercept_).ravel()[0])

    return {
        "features": features, "num_cols": num_cols, "cat_cols": cat_cols,
        "imp_stats": imp_stats, "s_mean": s_mean, "s_scale": s_scale,
        "ohe_categories": ohe_categories, "ohe_active": ohe_active,
        "coef": coef, "intercept": intercept,
    }


def manual_predict(raw: dict, art: dict) -> float:
    nums = np.array([float(raw[c]) for c in art["num_cols"]], dtype=float)
    if art["imp_stats"] is not None:
        mask = np.isnan(nums)
        if mask.any():
            nums[mask] = art["imp_stats"][mask]
    if art["s_mean"] is not None and art["s_scale"] is not None:
        nums = (nums - art["s_mean"]) / art["s_scale"]
    cat_parts = []
    for i, col in enumerate(art["cat_cols"]):
        cats = art["ohe_categories"][i]
        active = art["ohe_active"][i]
        vec = np.zeros(len(active), dtype=float)
        match = np.where(cats == raw[col])[0]
        if len(match) > 0 and int(match[0]) in active:
            vec[active.index(int(match[0]))] = 1.0
        cat_parts.append(vec)
    cat_vec = np.concatenate(cat_parts) if cat_parts else np.array([], dtype=float)
    x = np.concatenate([nums, cat_vec])
    return float(np.dot(x, art["coef"]) + art["intercept"])


# ============================================================================
# RENDER
# ============================================================================
inject_css()

try:
    ART = load_artifacts()
except Exception as e:
    st.error(f"Gagal memuat model: {e}")
    st.stop()


LABEL_MAP = {
    "Sales_in_thousands": "Jumlah Penjualan (ribuan unit)",
    "Engine_size": "Ukuran Mesin (Liter)",
    "Horsepower": "Tenaga Kuda / Horsepower (HP)",
    "Wheelbase": "Wheelbase (inch)",
    "Width": "Lebar Kendaraan (inch)",
    "Length": "Panjang Kendaraan (inch)",
    "Curb_weight": "Berat Kendaraan (ton)",
    "Fuel_capacity": "Kapasitas Tangki (gallon)",
    "Fuel_efficiency": "Efisiensi BBM (MPG)",
    "Power_perf_factor": "Power Performance Factor",
    "Vehicle_type": "Jenis Kendaraan",
}

FIELDS = [
    ("Sales_in_thousands", "Jumlah Penjualan",   "ribuan unit", "📈", 0.01, 0.11, 540.56, 53.0,    "Volume penjualan tahunan dalam ribuan unit."),
    ("Engine_size",        "Ukuran Mesin",       "Liter",       "⚙️", 0.1,  1.0,  8.0,    3.06,   "Kapasitas mesin dalam liter."),
    ("Horsepower",         "Tenaga Kuda",        "HP",          "🐎", 1.0,  55.0, 450.0,  185.95, "Daya keluaran mesin (horsepower)."),
    ("Wheelbase",          "Wheelbase",          "inch",        "📏", 0.1,  92.6, 138.7,  107.49, "Jarak antara sumbu roda depan dan belakang."),
    ("Width",              "Lebar Kendaraan",    "inch",        "↔️", 0.1,  62.6, 79.9,   71.15,  "Lebar kendaraan dalam inci."),
    ("Length",             "Panjang Kendaraan",  "inch",        "📐", 0.1,  149.4,224.5,  187.34, "Panjang total kendaraan dalam inci."),
    ("Curb_weight",        "Berat Kendaraan",    "ton",         "⚖️", 0.01, 1.9,  5.57,   3.38,   "Berat kosong kendaraan (ton metrik)."),
    ("Fuel_capacity",      "Kapasitas Tangki",   "gallon",      "⛽", 0.1,  10.3, 32.0,   17.95,  "Kapasitas tangki bahan bakar (US gallon)."),
    ("Fuel_efficiency",    "Efisiensi BBM",      "MPG",         "🌿", 0.1,  15.0, 45.0,   23.84,  "Konsumsi BBM (Miles Per Gallon)."),
    ("Power_perf_factor",  "Power Perf. Factor", "index",       "⚡", 0.01, 23.28,188.14, 77.04,  "Indeks komposit performa daya kendaraan."),
]


# ===== NAVBAR =====
nav_l, nav_r = st.columns([6, 1])
with nav_l:
    st.markdown(
        """
        <div class="navbar">
            <div class="logo">
                <span class="logo-mark">🚗</span>
                <span>CarPrice <span class="ai-tag">AI</span></span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with nav_r:
    st.markdown('<div class="theme-toggle" style="display:flex; justify-content:flex-end; padding-top:6px;">', unsafe_allow_html=True)
    icon = "☀️" if is_dark() else "🌙"
    if st.button(icon, key="theme_btn", help="Toggle dark mode"):
        st.session_state.theme = "light" if is_dark() else "dark"
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)


# ===== HERO =====
st.markdown(
    """
    <div class="hero">
        <div class="brand">
            <span class="brand-dot"></span>
            CarPrice AI · Live Machine Learning Prediction
        </div>
        <h1>Estimasi <span class="accent">harga mobil</span> secara<br/>instan dan akurat dengan AI.</h1>
        <p class="subtitle">
            Masukkan spesifikasi kendaraan dan dapatkan perkiraan harga pasar real-time.
            Didukung pipeline scikit-learn yang ter-validasi secara saintifik.
        </p>
        <div>
            <span class="stat-pill">⚡ Real-time Inference</span>
            <span class="stat-pill">🧠 Linear Regression Pipeline</span>
            <span class="stat-pill">📊 11 Fitur Input</span>
            <span class="stat-pill">💵 Output dalam USD</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ===== MAIN GRID =====
col_form, col_result = st.columns([1.05, 1], gap="large")

# ---------- LEFT: FORM CARD ----------
with col_form:
    with st.container(border=True):
        st.markdown(
            """
            <div class="card-header">
                <div class="card-icon">🚗</div>
                <div class="card-title">Spesifikasi Kendaraan</div>
            </div>
            <div class="card-sub">Hasil prediksi diperbarui otomatis saat Anda mengubah input.</div>
            """,
            unsafe_allow_html=True,
        )

        bcol1, bcol2, _ = st.columns([1.3, 1, 2])
        with bcol1:
            if st.button("✨ Isi rata-rata", key="fill_avg", type="secondary", use_container_width=True):
                for fid, *_, default, _ in FIELDS:
                    st.session_state[f"in_{fid}"] = float(default)
                st.session_state["in_Vehicle_type"] = (
                    list(ART["ohe_categories"][0])[0] if ART["ohe_categories"] else "Car"
                )
                st.rerun()
        with bcol2:
            if st.button("↺ Reset", key="reset_btn", type="secondary", use_container_width=True):
                for fid, *_, default, _ in FIELDS:
                    st.session_state[f"in_{fid}"] = float(default)
                st.session_state["in_Vehicle_type"] = (
                    list(ART["ohe_categories"][0])[0] if ART["ohe_categories"] else "Car"
                )
                st.session_state.last_price = None
                st.session_state.predictions_count = 0
                st.session_state.show_confetti = False
                st.rerun()

        inputs = {}
        grid = st.columns(2)
        for i, (fid, label, unit, icon, step, fmin, fmax, default, hint) in enumerate(FIELDS):
            with grid[i % 2]:
                key_in = f"in_{fid}"
                if key_in not in st.session_state:
                    st.session_state[key_in] = float(default)
                inputs[fid] = st.number_input(
                    f"{icon}  {label}  ·  {unit}",
                    min_value=float(fmin),
                    max_value=float(fmax),
                    step=float(step),
                    key=key_in,
                    help=f"{hint}\n\nRange: {fmin} – {fmax} · rata-rata: {default}",
                )

        inputs["Vehicle_type"] = st.selectbox(
            "🚙  Jenis Kendaraan  ·  kategori",
            options=list(ART["ohe_categories"][0]) if ART["ohe_categories"] else ["Car", "Passenger"],
            key="in_Vehicle_type",
            help="Pilih kategori kendaraan: Car atau Passenger.",
        )

        filled = sum(1 for fid, *_ in FIELDS if st.session_state.get(f"in_{fid}") is not None)
        if st.session_state.get("in_Vehicle_type"):
            filled += 1
        pct = int((filled / 11) * 100)
        st.markdown(
            f"""
            <div class="progress-row">
                <div class="progress-bar"><div class="progress-fill" style="width: {pct}%;"></div></div>
                <div class="progress-label">{filled} / 11</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ---------- RIGHT: RESULT CARD ----------
with col_result:
    with st.container(border=True):
        st.markdown(
            """
            <div class="card-header">
                <div class="card-icon">💰</div>
                <div class="card-title">
                    Hasil Prediksi
                    <span class="live-badge"><span class="live-dot"></span>LIVE</span>
                </div>
            </div>
            <div class="card-sub">Estimasi harga otomatis diperbarui saat Anda mengubah input.</div>
            """,
            unsafe_allow_html=True,
        )

        try:
            harga = manual_predict(inputs, ART) * 1000

            last = st.session_state.last_price
            trend_html = ""
            if last is not None and last != 0:
                diff = harga - last
                pct_change = (diff / abs(last)) * 100
                if abs(pct_change) >= 0.05:
                    if diff > 0:
                        trend_html = f'<div class="price-trend up">▲ +{pct_change:.2f}% vs sebelumnya</div>'
                    else:
                        trend_html = f'<div class="price-trend down">▼ {pct_change:.2f}% vs sebelumnya</div>'
                else:
                    trend_html = '<div class="price-trend flat">─ tidak berubah</div>'

            st.markdown(
                f"""
                <div class="price-card">
                    <span class="price-label">💎 Estimasi Harga</span>
                    <div class="price-value">${harga:,.2f}</div>
                    <div class="price-note">Nilai dalam US Dollar (USD) berdasarkan model regresi.</div>
                    {trend_html}
                </div>
                """,
                unsafe_allow_html=True,
            )

            if st.session_state.predictions_count == 0:
                st.session_state.show_confetti = True
            st.session_state.predictions_count += 1
            st.session_state.last_price = harga

            if harga < 0:
                st.warning(
                    "⚠️ Hasil prediksi negatif. Ini ekstrapolasi model linear di luar range training. "
                    "Coba sesuaikan input ke rentang yang lebih realistis."
                )

            rows_html = "".join(
                f"<tr><td>{LABEL_MAP.get(k, k)}</td><td>{v}</td></tr>"
                for k, v in inputs.items()
            )
            st.markdown(
                f"""
                <div class="summary-title">📋 Ringkasan Input</div>
                <table class="summary-table">
                    <tbody>{rows_html}</tbody>
                </table>
                """,
                unsafe_allow_html=True,
            )
        except Exception as e:
            st.error(f"Terjadi kesalahan: {e}")

        st.markdown(
            """
            <div class="credit">
                <div class="credit-avatar">JA</div>
                <div class="credit-meta">
                    <div class="credit-title">SISTEM INI DIBUAT OLEH</div>
                    <div class="credit-name">Jeni Adi Hidayat</div>
                    <div class="credit-nim">NIM · 237006158</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ===== CONFETTI =====
if st.session_state.show_confetti:
    st.session_state.show_confetti = False
    components.html(
        """
        <script src="https://cdn.jsdelivr.net/npm/canvas-confetti@1.9.3/dist/confetti.browser.min.js"></script>
        <script>
        const colors = ['#1a237e', '#3949ab', '#7986cb', '#c5cae9', '#ffffff'];
        confetti({ particleCount: 90, spread: 70, origin: { y: 0.4 }, colors });
        setTimeout(() => confetti({ particleCount: 50, spread: 100, origin: { x: 0.2, y: 0.45 }, colors }), 200);
        setTimeout(() => confetti({ particleCount: 50, spread: 100, origin: { x: 0.8, y: 0.45 }, colors }), 350);
        </script>
        """,
        height=0,
    )


# ===== FOOTER =====
st.markdown(
    """
    <div class="app-footer">
        © 2026 CarPrice AI · Built with Streamlit & scikit-learn
    </div>
    """,
    unsafe_allow_html=True,
)
