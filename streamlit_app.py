"""
CarPrice AI — versi Streamlit
Jalankan lokal: streamlit run streamlit_app.py
"""
import os
import warnings

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import InconsistentVersionWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

# Shim agar pickle dari sklearn 1.6 bisa dibuka di sklearn versi lebih baru
import sklearn.compose._column_transformer as _ct_module
if not hasattr(_ct_module, "_RemainderColsList"):
    class _RemainderColsList(list):
        pass
    _ct_module._RemainderColsList = _RemainderColsList


# ---------------------------------------------------------------------------
# Page config & styling
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="CarPrice AI — Prediksi Harga Mobil",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
        .main .block-container { padding-top: 2rem; padding-bottom: 3rem; }
        h1, h2, h3 { font-family: 'Space Grotesk', 'Inter', sans-serif; letter-spacing: -0.01em; }
        .hero {
            background: linear-gradient(135deg, #0a1545 0%, #1a237e 60%, #3949ab 100%);
            color: white;
            padding: 36px 32px;
            border-radius: 20px;
            margin-bottom: 24px;
            box-shadow: 0 24px 60px rgba(10, 21, 69, 0.28);
        }
        .hero h1 { color: white; font-size: 2.2rem; margin: 0; }
        .hero p  { color: rgba(255,255,255,0.85); margin-top: 8px; max-width: 720px; }
        .pill {
            display: inline-block; padding: 6px 14px; margin: 6px 6px 0 0;
            background: rgba(255,255,255,0.12); border: 1px solid rgba(255,255,255,0.2);
            border-radius: 999px; color: white; font-size: 12.5px; font-weight: 600;
        }
        .price-card {
            background: linear-gradient(135deg, #131f5e, #283593);
            color: white;
            padding: 30px 28px;
            border-radius: 18px;
            box-shadow: 0 14px 32px rgba(26,35,126,0.34);
        }
        .price-label {
            display: inline-block; padding: 4px 12px;
            background: rgba(255,255,255,0.14);
            border-radius: 999px;
            font-size: 11px; letter-spacing: 1.4px; font-weight: 700;
            text-transform: uppercase;
        }
        .price-value {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 2.6rem; font-weight: 800;
            letter-spacing: -0.02em; margin-top: 10px;
        }
        .price-note { color: rgba(255,255,255,0.75); font-size: 13px; margin-top: 8px; }
        .credit {
            margin-top: 18px; padding: 16px 18px;
            background: linear-gradient(135deg, #e3f2fd, #f0f4ff);
            border-radius: 14px; border-left: 4px solid #1a237e;
        }
        .credit b { color: #0a1545; }
        section[data-testid="stSidebar"] { display: none; }
        footer { visibility: hidden; }
        #MainMenu { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Load model & extract parameters (manual prediction = bebas masalah versi)
# ---------------------------------------------------------------------------
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
        "features": features,
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "imp_stats": imp_stats,
        "s_mean": s_mean,
        "s_scale": s_scale,
        "ohe_categories": ohe_categories,
        "ohe_active": ohe_active,
        "coef": coef,
        "intercept": intercept,
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


try:
    ART = load_artifacts()
except Exception as e:
    st.error(f"Gagal memuat model: {e}")
    st.stop()


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <h1>🚗 CarPrice AI</h1>
        <p>Estimasi harga mobil berbasis machine learning. Masukkan spesifikasi kendaraan
        dan dapatkan perkiraan harga pasar secara real-time.</p>
        <div>
            <span class="pill">⚡ Real-time Inference</span>
            <span class="pill">🧠 Linear Regression Pipeline</span>
            <span class="pill">📊 11 Fitur Input</span>
            <span class="pill">💵 Output dalam USD</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

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

# (id, label, satuan, ikon, step, min, max, default)
FIELDS = [
    ("Sales_in_thousands", "Jumlah Penjualan",   "ribuan unit", "📈", 0.01, 0.11, 540.56, 53.0),
    ("Engine_size",        "Ukuran Mesin",       "Liter",       "⚙️", 0.1,  1.0,  8.0,    3.06),
    ("Horsepower",         "Tenaga Kuda",        "HP",          "🐎", 1.0,  55.0, 450.0,  185.95),
    ("Wheelbase",          "Wheelbase",          "inch",        "📏", 0.1,  92.6, 138.7,  107.49),
    ("Width",              "Lebar Kendaraan",    "inch",        "↔️", 0.1,  62.6, 79.9,   71.15),
    ("Length",             "Panjang Kendaraan",  "inch",        "📐", 0.1,  149.4,224.5,  187.34),
    ("Curb_weight",        "Berat Kendaraan",    "ton",         "⚖️", 0.01, 1.9,  5.57,   3.38),
    ("Fuel_capacity",      "Kapasitas Tangki",   "gallon",      "⛽", 0.1,  10.3, 32.0,   17.95),
    ("Fuel_efficiency",    "Efisiensi BBM",      "MPG",         "🌿", 0.1,  15.0, 45.0,   23.84),
    ("Power_perf_factor",  "Power Perf. Factor", "index",       "⚡", 0.01, 23.28,188.14, 77.04),
]

col_form, col_result = st.columns([1.05, 1], gap="large")

with col_form:
    st.subheader("🚗 Spesifikasi Kendaraan")
    st.caption("Hasil prediksi diperbarui otomatis saat Anda mengubah input.")

    inputs = {}
    grid = st.columns(2)
    for i, (fid, label, unit, icon, step, fmin, fmax, default) in enumerate(FIELDS):
        with grid[i % 2]:
            inputs[fid] = st.number_input(
                f"{icon} {label} ({unit})",
                min_value=float(fmin),
                max_value=float(fmax),
                value=float(default),
                step=float(step),
                help=f"Range: {fmin} – {fmax} · rata-rata: {default}",
                key=fid,
            )

    inputs["Vehicle_type"] = st.selectbox(
        "🚙 Jenis Kendaraan",
        options=list(ART["ohe_categories"][0]) if ART["ohe_categories"] else ["Car", "Passenger"],
        index=0,
        help="Pilih kategori kendaraan.",
    )

with col_result:
    st.subheader("💰 Hasil Prediksi")
    st.caption("Estimasi harga pasar berdasarkan spesifikasi yang Anda masukkan.")

    try:
        harga = manual_predict(inputs, ART) * 1000
        st.markdown(
            f"""
            <div class="price-card">
                <span class="price-label">💎 Estimasi Harga</span>
                <div class="price-value">${harga:,.2f}</div>
                <div class="price-note">Nilai dalam US Dollar (USD) — model Linear Regression.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if harga < 0:
            st.warning(
                "⚠️ Hasil prediksi negatif. Ini adalah ekstrapolasi model linear di luar range data training. "
                "Coba sesuaikan nilai input ke rentang yang lebih realistis."
            )

        st.markdown("##### 📋 Ringkasan Input")
        df_summary = pd.DataFrame(
            [(LABEL_MAP.get(k, k), v) for k, v in inputs.items()],
            columns=["Spesifikasi", "Nilai"],
        )
        st.dataframe(df_summary, hide_index=True, use_container_width=True)

    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")

    st.markdown(
        """
        <div class="credit">
            <div style="font-size:11px; letter-spacing:1.4px; color:#5b6391; font-weight:700;">
                SISTEM INI DIBUAT OLEH
            </div>
            <div style="margin-top:4px;"><b>Jeni Adi Hidayat</b></div>
            <div style="color:#5b6391;">NIM · 237006158</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    "<div style='text-align:center; color:#8c93b8; padding:24px 0; font-size:13px;'>"
    "© 2026 CarPrice AI · Built with Streamlit & scikit-learn"
    "</div>",
    unsafe_allow_html=True,
)
