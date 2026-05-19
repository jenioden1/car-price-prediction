"""
CarPrice AI — versi Streamlit
Tampilan 1:1 dengan app.py (Flask). Halaman index.html yang sama persis di-embed
sebagai iframe; prediksi live dihitung di JavaScript memakai parameter model
(coef, mean, scale, dst) yang diekstrak dari pipeline scikit-learn.

Jalankan lokal: streamlit run streamlit_app.py
"""
import json
import os
import warnings

import joblib
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from jinja2 import Environment, FileSystemLoader, select_autoescape
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import InconsistentVersionWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

# Shim agar pickle dari sklearn 1.6 bisa dibuka di sklearn versi baru
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

# Streamlit chrome dihilangkan supaya iframe-nya yang full
st.markdown(
    """
    <style>
        [data-testid="stHeader"] { display: none; }
        [data-testid="stToolbar"] { display: none; }
        #MainMenu, footer { visibility: hidden; }
        .block-container {
            padding: 0 !important;
            max-width: 100% !important;
        }
        .stApp { background: #f5f7ff; }
        iframe { border: none !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================================
# LOAD MODEL & EXTRACT PARAMS
# ============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
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

    imp_stats = (
        np.asarray(imputer.statistics_, dtype=float).tolist() if imputer is not None else None
    )
    s_mean = np.asarray(scaler.mean_, dtype=float).tolist() if scaler is not None else None
    s_scale = np.asarray(scaler.scale_, dtype=float).tolist() if scaler is not None else None

    ohe_categories = [np.asarray(c).tolist() for c in encoder.categories_]
    ohe_drop = getattr(encoder, "drop_idx_", None)
    ohe_active = []
    for i, cats in enumerate(ohe_categories):
        drop = None
        if ohe_drop is not None:
            d = ohe_drop[i]
            if d is not None and not (isinstance(d, float) and np.isnan(d)):
                drop = int(d)
        ohe_active.append([j for j in range(len(cats)) if j != drop])

    coef = np.asarray(regressor.coef_, dtype=float).ravel().tolist()
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


# ============================================================================
# RENDER index.html via Jinja2
# ============================================================================
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


@st.cache_data(show_spinner=False)
def render_template_html() -> str:
    env = Environment(
        loader=FileSystemLoader(TEMPLATES_DIR),
        autoescape=select_autoescape(["html", "xml"]),
    )
    tpl = env.get_template("index.html")
    return tpl.render(
        harga_prediksi=None,
        input_data=None,
        error=None,
        label_map=LABEL_MAP,
    )


def patch_for_streamlit(html: str, art: dict) -> str:
    """
    Ganti semua hal yang bergantung pada Flask backend dengan logika sisi-klien:
    - <form action="/predict"> → action="javascript:void(0)" (no-op)
    - fetch('/api/predict')    → panggilan ke fungsi predictLocal() di JS
    Kemudian sisipkan parameter model (coef, mean, scale, ...) sebagai JSON
    dan fungsi predictLocal yang melakukan perhitungan numpy-equivalent.
    """
    # 1) Form tidak boleh submit ke endpoint
    html = html.replace(
        'action="/predict" method="POST"',
        'action="javascript:void(0)" method="POST"',
    )

    # 2) Inject script: model params + override fetch
    inject = """
<script>
window.__CP_MODEL__ = __MODEL_JSON__;

(function() {
    const M = window.__CP_MODEL__;

    function predictLocal(payload) {
        // Numeric pipeline: impute -> standard scale
        const nums = M.num_cols.map(c => parseFloat(payload[c]));
        for (let i = 0; i < nums.length; i++) {
            if (Number.isNaN(nums[i]) && M.imp_stats) nums[i] = M.imp_stats[i];
        }
        if (M.s_mean && M.s_scale) {
            for (let i = 0; i < nums.length; i++) {
                nums[i] = (nums[i] - M.s_mean[i]) / M.s_scale[i];
            }
        }
        // Categorical pipeline: one-hot dengan respect drop_idx
        let catVec = [];
        for (let i = 0; i < M.cat_cols.length; i++) {
            const cats = M.ohe_categories[i];
            const active = M.ohe_active[i];
            const v = new Array(active.length).fill(0);
            const idx = cats.indexOf(payload[M.cat_cols[i]]);
            if (idx >= 0) {
                const pos = active.indexOf(idx);
                if (pos >= 0) v[pos] = 1;
            }
            catVec = catVec.concat(v);
        }
        const x = nums.concat(catVec);
        let y = M.intercept;
        for (let i = 0; i < x.length; i++) y += x[i] * M.coef[i];
        return y * 1000; // model output dalam ribuan USD
    }

    // Override window.fetch untuk endpoint /api/predict saja
    const origFetch = window.fetch.bind(window);
    window.fetch = function(input, init) {
        const url = typeof input === 'string' ? input : input.url;
        if (url && url.endsWith('/api/predict')) {
            try {
                const body = init && init.body ? JSON.parse(init.body) : {};
                const harga = predictLocal(body);
                const json = {
                    success: true,
                    harga_prediksi: harga,
                    input_data: body,
                    label_map: __LABEL_MAP_JSON__,
                };
                return Promise.resolve(new Response(JSON.stringify(json), {
                    status: 200,
                    headers: { 'Content-Type': 'application/json' }
                }));
            } catch (e) {
                return Promise.resolve(new Response(JSON.stringify({
                    success: false, error: String(e)
                }), { status: 400, headers: { 'Content-Type': 'application/json' }}));
            }
        }
        return origFetch(input, init);
    };

    // Submit form juga diarahkan ke prediksi lokal (fallback non-JS tidak ada di Streamlit)
    document.addEventListener('DOMContentLoaded', function() {
        const form = document.getElementById('predictForm');
        if (form) {
            form.addEventListener('submit', function(e) {
                e.preventDefault();
                if (typeof scheduleLivePredict === 'function') scheduleLivePredict();
            }, true);
        }
    });
})();
</script>
"""
    inject = inject.replace("__MODEL_JSON__", json.dumps(art))
    inject = inject.replace("__LABEL_MAP_JSON__", json.dumps(LABEL_MAP))

    # Sisipkan tepat sebelum </head> agar variabel tersedia sebelum script lain jalan
    if "</head>" in html:
        html = html.replace("</head>", inject + "\n</head>")
    else:
        html = inject + html
    return html


# ============================================================================
# RENDER
# ============================================================================
try:
    ART = load_artifacts()
except Exception as e:
    st.error(f"Gagal memuat model: {e}")
    st.stop()

try:
    base_html = render_template_html()
except Exception as e:
    st.error(f"Gagal me-render template index.html: {e}")
    st.stop()

final_html = patch_for_streamlit(base_html, ART)

# Iframe tinggi cukup besar agar seluruh konten tampil tanpa scroll dalam frame
components.html(final_html, height=2400, scrolling=True)
