# =====================================================
# APLIKASI PREDIKSI HARGA MOBIL
# =====================================================
# CARA MENJALANKAN:
# 1. pip install -r requirements.txt
# 2. Taruh model_prediksi_harga_mobil.pkl dan feature_columns.pkl di folder yang sama
# 3. python app.py
# 4. Buka browser: http://localhost:5000
# =====================================================

from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import numpy as np
import os
import warnings
from sklearn.exceptions import InconsistentVersionWarning
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression

warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

# ---------------------------------------------------------------------------
# Shim agar pickle dari sklearn 1.6 bisa di-unpickle di sklearn 1.8
# ---------------------------------------------------------------------------
import sklearn.compose._column_transformer as _ct_module
if not hasattr(_ct_module, "_RemainderColsList"):
    class _RemainderColsList(list):
        """Stub agar pickle lama tetap bisa dimuat di sklearn versi baru."""
        pass
    _ct_module._RemainderColsList = _RemainderColsList

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model_prediksi_harga_mobil.pkl')
FEATURES_PATH = os.path.join(BASE_DIR, 'feature_columns.pkl')

model = joblib.load(MODEL_PATH)
features = list(joblib.load(FEATURES_PATH))


# ---------------------------------------------------------------------------
# Ekstraksi parameter pipeline secara MANUAL
# ---------------------------------------------------------------------------
# Karena pipeline dari sklearn 1.6 tidak fully kompatibel saat di-transform di
# sklearn 1.8 (menyebabkan output prediksi konstan), kita keluarkan semua
# parameter yang sudah ter-fit (mean, scale, categories, coef, intercept) lalu
# hitung prediksi sendiri pakai numpy. Hasilnya identik dengan model.predict()
# di sklearn 1.6, tapi tidak bergantung versi.
# ---------------------------------------------------------------------------
def _find_first(estimator, target_cls):
    if isinstance(estimator, target_cls):
        return estimator
    if isinstance(estimator, Pipeline):
        for _, step in estimator.steps:
            found = _find_first(step, target_cls)
            if found is not None:
                return found
    return None


if not isinstance(model, Pipeline):
    raise RuntimeError("Model yang dimuat bukan sklearn.Pipeline.")

preprocessor = None
regressor = None
for _, step in model.steps:
    if isinstance(step, ColumnTransformer) and preprocessor is None:
        preprocessor = step
    elif isinstance(step, LinearRegression) and regressor is None:
        regressor = step

if preprocessor is None or regressor is None:
    raise RuntimeError("Tidak menemukan ColumnTransformer / LinearRegression di pipeline.")

NUM_COLS, CAT_COLS = [], []
imputer = scaler = encoder = None

transformers_iter = getattr(preprocessor, 'transformers_', None) or preprocessor.transformers
for name, trans, cols in transformers_iter:
    if trans == 'drop' or trans == 'passthrough':
        continue
    cols_list = list(cols) if not isinstance(cols, str) else [cols]
    enc = _find_first(trans, OneHotEncoder)
    if enc is not None:
        CAT_COLS = cols_list
        encoder = enc
    else:
        NUM_COLS = cols_list
        imputer = _find_first(trans, SimpleImputer)
        scaler = _find_first(trans, StandardScaler)

# Numeric params
IMP_STATS = np.asarray(imputer.statistics_, dtype=float) if imputer is not None else None
SCALER_MEAN = np.asarray(scaler.mean_, dtype=float) if scaler is not None else None
SCALER_SCALE = np.asarray(scaler.scale_, dtype=float) if scaler is not None else None

# Categorical params
OHE_CATEGORIES = [np.asarray(c) for c in encoder.categories_]
OHE_DROP_IDX = getattr(encoder, 'drop_idx_', None)

OHE_ACTIVE_INDICES = []
for i, cats in enumerate(OHE_CATEGORIES):
    drop = None
    if OHE_DROP_IDX is not None:
        d = OHE_DROP_IDX[i]
        if d is not None and not (isinstance(d, float) and np.isnan(d)):
            drop = int(d)
    OHE_ACTIVE_INDICES.append([j for j in range(len(cats)) if j != drop])

# Regressor params
COEF = np.asarray(regressor.coef_, dtype=float).ravel()
INTERCEPT = float(np.asarray(regressor.intercept_).ravel()[0])


def manual_predict(raw_input: dict) -> float:
    """Reproduksi pipeline (impute -> scale -> onehot -> linreg) pakai numpy."""
    # ----- Numeric branch -----
    nums = np.array([float(raw_input[c]) for c in NUM_COLS], dtype=float)
    if IMP_STATS is not None:
        nan_mask = np.isnan(nums)
        if nan_mask.any():
            nums[nan_mask] = IMP_STATS[nan_mask]
    if SCALER_MEAN is not None and SCALER_SCALE is not None:
        nums = (nums - SCALER_MEAN) / SCALER_SCALE

    # ----- Categorical branch -----
    cat_parts = []
    for i, col in enumerate(CAT_COLS):
        cats = OHE_CATEGORIES[i]
        active = OHE_ACTIVE_INDICES[i]
        vec = np.zeros(len(active), dtype=float)
        match = np.where(cats == raw_input[col])[0]
        if len(match) > 0:
            cat_idx = int(match[0])
            if cat_idx in active:
                vec[active.index(cat_idx)] = 1.0
        cat_parts.append(vec)
    cat_vec = np.concatenate(cat_parts) if cat_parts else np.array([], dtype=float)

    x = np.concatenate([nums, cat_vec])
    if x.shape[0] != COEF.shape[0]:
        raise RuntimeError(
            f"Dimensi fitur tidak cocok dengan koefisien model "
            f"(got {x.shape[0]}, expected {COEF.shape[0]})."
        )
    return float(np.dot(x, COEF) + INTERCEPT)


# ---------------------------------------------------------------------------
# Konstanta UI
# ---------------------------------------------------------------------------
LABEL_MAP = {
    'Sales_in_thousands': 'Jumlah Penjualan (ribuan unit)',
    'Engine_size': 'Ukuran Mesin (Liter)',
    'Horsepower': 'Tenaga Kuda / Horsepower (HP)',
    'Wheelbase': 'Wheelbase (inch)',
    'Width': 'Lebar Kendaraan (inch)',
    'Length': 'Panjang Kendaraan (inch)',
    'Curb_weight': 'Berat Kendaraan (ton)',
    'Fuel_capacity': 'Kapasitas Tangki (gallon)',
    'Fuel_efficiency': 'Efisiensi BBM (MPG)',
    'Power_perf_factor': 'Power Performance Factor',
    'Vehicle_type': 'Jenis Kendaraan',
}

VALID_VEHICLE_TYPES = list(OHE_CATEGORIES[0]) if OHE_CATEGORIES else ['Car', 'Passenger']


def _parse_payload(getter):
    """getter: callable(key, default='') -> str. Validasi & cast input."""
    raw = {}
    for col in NUM_COLS:
        v = (getter(col, '') or '').strip() if isinstance(getter(col, ''), str) else getter(col, '')
        if v == '' or v is None:
            raise ValueError(f"Field '{LABEL_MAP.get(col, col)}' tidak boleh kosong.")
        try:
            raw[col] = float(v)
        except (ValueError, TypeError):
            raise ValueError(f"Field '{LABEL_MAP.get(col, col)}' harus berupa angka.")
    for col in CAT_COLS:
        v = getter(col, '')
        if isinstance(v, str):
            v = v.strip()
        if not v:
            raise ValueError(f"Field '{LABEL_MAP.get(col, col)}' tidak boleh kosong.")
        if col == 'Vehicle_type' and v not in VALID_VEHICLE_TYPES:
            raise ValueError(f"Jenis Kendaraan harus salah satu dari: {VALID_VEHICLE_TYPES}.")
        raw[col] = v
    return raw


@app.route('/', methods=['GET'])
def index():
    return render_template(
        'index.html',
        harga_prediksi=None,
        input_data=None,
        error=None,
        label_map=LABEL_MAP,
    )


@app.route('/predict', methods=['POST'])
def predict():
    try:
        raw = _parse_payload(lambda k, d='': request.form.get(k, d))
        harga = manual_predict(raw) * 1000
        return render_template(
            'index.html',
            harga_prediksi=harga,
            input_data=raw,
            error=None,
            label_map=LABEL_MAP,
        )
    except Exception as e:
        return render_template(
            'index.html',
            harga_prediksi=None,
            input_data=None,
            error=f"Terjadi kesalahan: {str(e)}",
            label_map=LABEL_MAP,
        )


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """Endpoint JSON untuk live prediction (auto-update tanpa refresh)."""
    try:
        data = request.get_json(silent=True)
        if data is None:
            data = request.form.to_dict()
        raw = _parse_payload(lambda k, d='': data.get(k, d))
        harga = manual_predict(raw) * 1000
        return jsonify({
            'success': True,
            'harga_prediksi': harga,
            'input_data': raw,
            'label_map': LABEL_MAP,
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
