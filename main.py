

import io
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")

# ─────────────────────────── Page Config ────────────────────────────
st.set_page_config(
    page_title="ATM Shipment Forecasting Dashboard",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────── Custom CSS ─────────────────────────────
st.markdown("""
<style>
/* ── Root & Background ── */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0d1b2a 0%, #1b2d45 50%, #0d1b2a 100%);
    color: #e0e9f5;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a1628 0%, #112240 100%);
    border-right: 1px solid #1e3a5f;
}
[data-testid="stSidebar"] * { color: #ccd6f6 !important; }
[data-testid="stSidebar"] .stRadio label { font-size: 0.95rem; }

/* ── Main title ── */
.main-title {
    font-size: 2.4rem; font-weight: 800;
    background: linear-gradient(90deg, #64b5f6, #42a5f5, #1e88e5);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 0.2rem;
}
.sub-title { color: #8bafd4; font-size: 1rem; margin-bottom: 1.5rem; }

/* ── KPI Cards ── */
.kpi-card {
    background: linear-gradient(135deg, #112240 0%, #1b3a6b 100%);
    border: 1px solid #1e4d82;
    border-radius: 16px; padding: 20px 24px;
    box-shadow: 0 4px 20px rgba(30,100,200,0.25);
    text-align: center; margin-bottom: 12px;
    transition: transform 0.2s;
}
.kpi-card:hover { transform: translateY(-3px); }
.kpi-icon { font-size: 2rem; margin-bottom: 6px; }
.kpi-label { color: #8bafd4; font-size: 0.82rem; text-transform: uppercase;
             letter-spacing: 0.08em; margin-bottom: 4px; }
.kpi-value { color: #64b5f6; font-size: 1.9rem; font-weight: 800; margin-bottom: 4px; }
.kpi-delta { font-size: 0.8rem; font-weight: 600; }
.kpi-positive { color: #43d98c; }
.kpi-negative { color: #ff6b6b; }

/* ── Section headers ── */
.section-header {
    color: #64b5f6; font-size: 1.3rem; font-weight: 700;
    border-left: 4px solid #1e88e5; padding-left: 12px;
    margin: 1.5rem 0 1rem 0;
}

/* ── Model cards ── */
.model-card {
    background: #112240; border-radius: 14px;
    border: 1px solid #1e4d82; padding: 16px;
    margin-bottom: 10px;
}
.best-model-card {
    background: linear-gradient(135deg, #0d3b5e, #1a5276);
    border: 2px solid #43d98c !important;
    box-shadow: 0 0 18px rgba(67, 217, 140, 0.3);
}
.model-name { font-size: 1.05rem; font-weight: 700; color: #e0e9f5; }
.metric-row { display: flex; gap: 18px; margin-top: 8px; }
.metric-item { flex: 1; text-align: center; background: #0d1b2a;
               border-radius: 8px; padding: 8px; }
.metric-label { color: #8bafd4; font-size: 0.75rem; }
.metric-val { color: #64b5f6; font-weight: 700; font-size: 1rem; }

/* ── Forecast table ── */
.forecast-table { width: 100%; border-collapse: collapse; margin-top: 10px; }
.forecast-table th {
    background: #1e3a6e; color: #90caf9;
    padding: 10px 14px; font-size: 0.85rem;
    text-transform: uppercase; letter-spacing: 0.05em;
}
.forecast-table td {
    padding: 10px 14px; text-align: center;
    border-bottom: 1px solid #1e3a5f; color: #e0e9f5;
}
.forecast-table tr:hover td { background: #1b3a6b; }
.best-badge {
    background: #43d98c; color: #0d1b2a;
    padding: 3px 10px; border-radius: 20px;
    font-size: 0.78rem; font-weight: 700;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(90deg, #1565c0, #1e88e5);
    color: white; border: none; border-radius: 8px;
    padding: 0.5rem 1.4rem; font-weight: 600;
    transition: all 0.2s;
}
.stButton > button:hover {
    background: linear-gradient(90deg, #1976d2, #42a5f5);
    box-shadow: 0 4px 15px rgba(30,136,229,0.5);
}

/* ── Upload area ── */
[data-testid="stFileUploader"] {
    background: #112240; border: 2px dashed #1e4d82;
    border-radius: 12px; padding: 20px;
}

/* ── Divider ── */
hr { border-color: #1e3a5f; }

/* ── Sliders ── */
[data-testid="stSlider"] > div > div { color: #64b5f6; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
#  HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════

MONTH_MAP = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4,
    "jun": 6, "jul": 7, "aug": 8,
    "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}

FUTURE_MONTHS = ["September", "October", "November", "December"]
FUTURE_NUMS   = [9, 10, 11, 12]


@st.cache_data(show_spinner=False)
def load_and_process(file_bytes: bytes) -> pd.DataFrame:
    df = pd.read_excel(io.BytesIO(file_bytes))
    df.columns = [c.strip() for c in df.columns]

    # Fill missing numeric values
    for col in df.select_dtypes(include=np.number).columns:
        df[col] = df[col].fillna(df[col].median())

    # Convert Month to numeric
    if "Month_Num" not in df.columns:
        df["Month_Num"] = (
            df["Month"].astype(str).str.lower().str.strip().map(MONTH_MAP)
        )
        if df["Month_Num"].isna().all():
            df["Month_Num"] = range(1, len(df) + 1)

    df.sort_values("Month_Num", inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Lag feature
    df["Lag1"] = df["Shipment_Volume"].shift(1).fillna(method="bfill")
    return df


def build_features(df: pd.DataFrame) -> tuple:
    X = df[["Month_Num", "Backlog", "Lag1"]].values
    y = df["Shipment_Volume"].values
    return X, y


def train_random_forest(X, y):
    param_grid = {
        "n_estimators": [50, 100, 200, 300],
        "max_depth": [3, 5, 7, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    }
    rf = RandomForestRegressor(random_state=42)
    search = RandomizedSearchCV(
        rf, param_grid, n_iter=20, cv=min(3, len(y) - 1),
        scoring="neg_mean_squared_error", random_state=42, n_jobs=-1
    )
    search.fit(X, y)
    best = search.best_estimator_
    preds = best.predict(X)
    mae  = mean_absolute_error(y, preds)
    rmse = np.sqrt(mean_squared_error(y, preds))
    return best, mae, rmse


def train_xgboost_or_gb(X, y):
    """Use XGBoost if available, else GradientBoosting."""
    try:
        from xgboost import XGBRegressor
        model = XGBRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, verbosity=0
        )
        model_name = "XGBoost"
    except ImportError:
        from sklearn.ensemble import GradientBoostingRegressor
        model = GradientBoostingRegressor(n_estimators=200, max_depth=4,
                                          learning_rate=0.1, random_state=42)
        model_name = "Gradient Boosting"

    model.fit(X, y)
    preds = model.predict(X)
    mae  = mean_absolute_error(y, preds)
    rmse = np.sqrt(mean_squared_error(y, preds))
    return model, mae, rmse, model_name


def train_arima(y_series: np.ndarray):
    """ARIMA(2,1,1) via statsmodels or linear trend fallback."""
    try:
        from statsmodels.tsa.arima.model import ARIMA
        model = ARIMA(y_series, order=(2, 1, 1))
        result = model.fit()
        fitted = result.fittedvalues
        # fittedvalues may be shorter due to differencing
        fitted_aligned = np.concatenate([[y_series[0]], fitted])[:len(y_series)]
        mae  = mean_absolute_error(y_series, fitted_aligned)
        rmse = np.sqrt(mean_squared_error(y_series, fitted_aligned))
        return result, mae, rmse, "ARIMA"
    except Exception:
        # Fallback: simple linear regression as "ARIMA fallback"
        from sklearn.linear_model import LinearRegression
        x = np.arange(len(y_series)).reshape(-1, 1)
        model = LinearRegression().fit(x, y_series)
        preds = model.predict(x)
        mae  = mean_absolute_error(y_series, preds)
        rmse = np.sqrt(mean_squared_error(y_series, preds))
        return model, mae, rmse, "Linear Trend (ARIMA fallback)"


def forecast_ml(model, last_vol, backlog_vals, last_lag):
    """Predict Sep–Dec using ML model (RF / XGB)."""
    preds = []
    prev_vol = last_vol
    for i, (mn, bl) in enumerate(zip(FUTURE_NUMS, backlog_vals)):
        lag = prev_vol if i == 0 else preds[-1]
        x = np.array([[mn, bl, lag]])
        p = float(model.predict(x)[0])
        preds.append(p)
        prev_vol = p
    return preds


def forecast_arima(model, steps=4):
    """Forecast 4 steps ahead from ARIMA result."""
    try:
        fc = model.forecast(steps=steps)
        return list(fc)
    except Exception:
        # Linear fallback
        x_fc = np.arange(8, 12).reshape(-1, 1)
        return list(model.predict(x_fc))


def best_model_name(metrics: dict) -> str:
    return min(metrics, key=lambda k: metrics[k]["rmse"])


def export_excel(forecast_df: pd.DataFrame, historical_df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        forecast_df.to_excel(writer, sheet_name="Forecast", index=False)
        historical_df.to_excel(writer, sheet_name="Historical", index=False)
    return buf.getvalue()


# ═══════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("""
        <div style='text-align:center; padding: 20px 0 10px 0;'>
            <div style='font-size:3rem;'>📦</div>
            <div style='font-size:1.1rem; font-weight:700; color:#64b5f6;'>ATM Shipment</div>
            <div style='font-size:0.8rem; color:#8bafd4;'>Forecasting Dashboard</div>
        </div>
        <hr style='border-color:#1e3a5f; margin: 10px 0 20px 0;'>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        ["📤  Upload Data", "🤖  Model Training", "📈  Forecast", "📊  Visualization"],
        label_visibility="collapsed"
    )

    st.markdown("<hr style='border-color:#1e3a5f; margin:20px 0;'>", unsafe_allow_html=True)
    st.markdown("""
        <div style='font-size:0.75rem; color:#4a6fa5; text-align:center;'>
            Models: Random Forest · XGBoost · ARIMA<br>
            <span style='color:#2c4a70;'>© 2025 ATM Logistics Analytics</span>
        </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
#  MAIN HEADER
# ═══════════════════════════════════════════════════════════════

st.markdown("""
    <div class='main-title'>📦 ATM Shipment Forecasting Dashboard</div>
    <div class='sub-title'>AI-powered demand forecasting for ATM machine logistics · Jan–Aug historical · Sep–Dec predictions</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
#  SESSION STATE
# ═══════════════════════════════════════════════════════════════

for key in ["df", "rf_model", "xgb_model", "arima_model",
            "rf_mae", "rf_rmse", "xgb_mae", "xgb_rmse",
            "arima_mae", "arima_rmse", "xgb_name", "arima_name",
            "trained", "backlog_vals"]:
    if key not in st.session_state:
        st.session_state[key] = None

if "trained" not in st.session_state:
    st.session_state.trained = False


# ═══════════════════════════════════════════════════════════════
#  PAGE: UPLOAD DATA
# ═══════════════════════════════════════════════════════════════

if "Upload" in page:
    st.markdown("<div class='section-header'>📤 Upload Shipment Data</div>", unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Upload your Excel file (shipment_data.xlsx)",
        type=["xlsx", "xls"],
        help="File must contain columns: Month, Shipment_Volume, Backlog"
    )

    if uploaded:
        with st.spinner("Processing data..."):
            df = load_and_process(uploaded.read())
        st.session_state.df = df
        st.success("✅ Data loaded and processed successfully!")

        c1, c2, c3 = st.columns(3)
        c1.metric("📅 Months Loaded", len(df))
        c2.metric("📦 Avg Monthly Shipments", f"{df['Shipment_Volume'].mean():,.0f}")
        c3.metric("📋 Avg Backlog", f"{df['Backlog'].mean():,.0f}")

        st.markdown("<div class='section-header'>📋 Dataset Preview</div>", unsafe_allow_html=True)
        display_df = df[["Month", "Month_Num", "Shipment_Volume", "Backlog", "Lag1"]].copy()
        display_df.columns = ["Month", "Month #", "Shipment Volume", "Backlog", "Lag-1 Volume"]
        st.dataframe(
            display_df.style
            .format({"Shipment Volume": "{:,.0f}", "Backlog": "{:,.0f}", "Lag-1 Volume": "{:,.0f}"})
            .background_gradient(subset=["Shipment Volume"], cmap="Blues")
            .background_gradient(subset=["Backlog"], cmap="Oranges"),
            use_container_width=True, height=340
        )

        st.markdown("<div class='section-header'>📊 Quick Statistics</div>", unsafe_allow_html=True)
        stats = df[["Shipment_Volume", "Backlog"]].describe().round(1)
        stats.columns = ["Shipment Volume", "Backlog"]
        st.dataframe(stats, use_container_width=True)

    else:
        st.info("👆 Please upload the **shipment_data.xlsx** file to begin.")
        st.markdown("""
        <div class='model-card'>
            <div class='model-name'>📌 Expected File Format</div>
            <br>
            <table class='forecast-table'>
                <tr><th>Month</th><th>Shipment_Volume</th><th>Backlog</th></tr>
                <tr><td>January</td><td>1,420</td><td>210</td></tr>
                <tr><td>February</td><td>1,385</td><td>195</td></tr>
                <tr><td>…</td><td>…</td><td>…</td></tr>
                <tr><td>August</td><td>2,050</td><td>375</td></tr>
            </table>
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
#  PAGE: MODEL TRAINING
# ═══════════════════════════════════════════════════════════════

elif "Training" in page:
    st.markdown("<div class='section-header'>🤖 Model Training & Evaluation</div>", unsafe_allow_html=True)

    if st.session_state.df is None:
        st.warning("⚠️ Please upload data first in the **Upload Data** tab.")
    else:
        df = st.session_state.df
        X, y = build_features(df)

        if st.button("🚀 Train All Models", use_container_width=True):
            with st.spinner("Training Random Forest with hyperparameter tuning…"):
                rf, rf_mae, rf_rmse = train_random_forest(X, y)
                st.session_state.rf_model, st.session_state.rf_mae, st.session_state.rf_rmse = rf, rf_mae, rf_rmse

            with st.spinner("Training XGBoost / Gradient Boosting…"):
                xgb, xgb_mae, xgb_rmse, xgb_name = train_xgboost_or_gb(X, y)
                st.session_state.xgb_model = xgb
                st.session_state.xgb_mae, st.session_state.xgb_rmse = xgb_mae, xgb_rmse
                st.session_state.xgb_name = xgb_name

            with st.spinner("Training ARIMA time-series model…"):
                arima, arima_mae, arima_rmse, arima_name = train_arima(y)
                st.session_state.arima_model = arima
                st.session_state.arima_mae, st.session_state.arima_rmse = arima_mae, arima_rmse
                st.session_state.arima_name = arima_name

            st.session_state.trained = True
            st.success("✅ All models trained successfully!")

        if st.session_state.trained:
            metrics = {
                "Random Forest": {"mae": st.session_state.rf_mae, "rmse": st.session_state.rf_rmse},
                st.session_state.xgb_name: {"mae": st.session_state.xgb_mae, "rmse": st.session_state.xgb_rmse},
                st.session_state.arima_name: {"mae": st.session_state.arima_mae, "rmse": st.session_state.arima_rmse},
            }
            best = best_model_name(metrics)

            st.markdown("<div class='section-header'>📊 Model Performance Comparison</div>", unsafe_allow_html=True)
            for name, m in metrics.items():
                is_best = name == best
                card_cls = "model-card best-model-card" if is_best else "model-card"
                badge = "<span class='best-badge'>🏆 BEST MODEL</span>" if is_best else ""
                st.markdown(f"""
                <div class='{card_cls}'>
                    <div class='model-name'>{name} {badge}</div>
                    <div class='metric-row'>
                        <div class='metric-item'>
                            <div class='metric-label'>MAE</div>
                            <div class='metric-val'>{m['mae']:,.1f}</div>
                        </div>
                        <div class='metric-item'>
                            <div class='metric-label'>RMSE</div>
                            <div class='metric-val'>{m['rmse']:,.1f}</div>
                        </div>
                        <div class='metric-item'>
                            <div class='metric-label'>RMSE Rank</div>
                            <div class='metric-val'>{"🥇" if is_best else "📊"}</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # Comparison bar chart
            names = list(metrics.keys())
            maes  = [metrics[n]["mae"]  for n in names]
            rmses = [metrics[n]["rmse"] for n in names]
            colors_bar = ["#43d98c" if n == best else "#1e88e5" for n in names]

            fig = make_subplots(rows=1, cols=2, subplot_titles=["MAE Comparison", "RMSE Comparison"])
            fig.add_trace(go.Bar(x=names, y=maes, marker_color=colors_bar, name="MAE",
                                  text=[f"{v:.1f}" for v in maes], textposition="outside"), row=1, col=1)
            fig.add_trace(go.Bar(x=names, y=rmses, marker_color=colors_bar, name="RMSE",
                                  text=[f"{v:.1f}" for v in rmses], textposition="outside"), row=1, col=2)
            fig.update_layout(
                template="plotly_dark", paper_bgcolor="#0d1b2a", plot_bgcolor="#112240",
                font=dict(color="#ccd6f6"), showlegend=False, height=380,
                title_text="Model Evaluation Metrics", title_font_size=16,
            )
            st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════
#  PAGE: FORECAST
# ═══════════════════════════════════════════════════════════════

elif "Forecast" in page:
    st.markdown("<div class='section-header'>📈 Shipment Volume Forecast (Sep–Dec)</div>", unsafe_allow_html=True)

    if not st.session_state.trained:
        st.warning("⚠️ Please train models first in the **Model Training** tab.")
    else:
        df = st.session_state.df

        st.markdown("#### 🎛️ Configure Backlog Values for Forecast Months")
        col1, col2, col3, col4 = st.columns(4)
        aug_bl = int(df["Backlog"].iloc[-1])
        sep_bl = col1.slider("Sep Backlog", 300, 600, aug_bl + 40,  step=5)
        oct_bl = col2.slider("Oct Backlog", 300, 600, aug_bl + 70,  step=5)
        nov_bl = col3.slider("Nov Backlog", 300, 700, aug_bl + 100, step=5)
        dec_bl = col4.slider("Dec Backlog", 350, 800, aug_bl + 150, step=5)
        backlog_vals = [sep_bl, oct_bl, nov_bl, dec_bl]
        st.session_state.backlog_vals = backlog_vals

        last_vol = float(df["Shipment_Volume"].iloc[-1])
        last_lag = float(df["Lag1"].iloc[-1])

        rf_preds    = forecast_ml(st.session_state.rf_model, last_vol, backlog_vals, last_lag)
        xgb_preds   = forecast_ml(st.session_state.xgb_model, last_vol, backlog_vals, last_lag)
        arima_preds = forecast_arima(st.session_state.arima_model)

        metrics = {
            "Random Forest": st.session_state.rf_rmse,
            st.session_state.xgb_name: st.session_state.xgb_rmse,
            st.session_state.arima_name: st.session_state.arima_rmse,
        }
        best = min(metrics, key=metrics.get)

        pred_map = {
            "Random Forest": rf_preds,
            st.session_state.xgb_name: xgb_preds,
            st.session_state.arima_name: arima_preds,
        }
        best_preds = pred_map[best]

        # KPI cards
        total_hist = int(df["Shipment_Volume"].sum())
        total_pred = int(sum(best_preds))
        aug_vol    = int(df["Shipment_Volume"].iloc[-1])
        dec_pred   = int(best_preds[-1])
        growth_pct = (dec_pred - aug_vol) / aug_vol * 100

        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.markdown(f"""
            <div class='kpi-card'>
                <div class='kpi-icon'>📦</div>
                <div class='kpi-label'>Total Predicted (Sep–Dec)</div>
                <div class='kpi-value'>{total_pred:,}</div>
                <div class='kpi-delta kpi-positive'>Best model: {best}</div>
            </div>""", unsafe_allow_html=True)
        with k2:
            st.markdown(f"""
            <div class='kpi-card'>
                <div class='kpi-icon'>📊</div>
                <div class='kpi-label'>Historical Total (Jan–Aug)</div>
                <div class='kpi-value'>{total_hist:,}</div>
                <div class='kpi-delta' style='color:#8bafd4;'>Actual shipments</div>
            </div>""", unsafe_allow_html=True)
        with k3:
            sign = "+" if growth_pct >= 0 else ""
            color_cls = "kpi-positive" if growth_pct >= 0 else "kpi-negative"
            st.markdown(f"""
            <div class='kpi-card'>
                <div class='kpi-icon'>📈</div>
                <div class='kpi-label'>Aug → Dec Growth</div>
                <div class='kpi-value'>{sign}{growth_pct:.1f}%</div>
                <div class='kpi-delta {color_cls}'>{aug_vol:,} → {dec_pred:,}</div>
            </div>""", unsafe_allow_html=True)
        with k4:
            avg_q4 = int(sum(best_preds) / 4)
            st.markdown(f"""
            <div class='kpi-card'>
                <div class='kpi-icon'>📋</div>
                <div class='kpi-label'>Avg Monthly Q4 Forecast</div>
                <div class='kpi-value'>{avg_q4:,}</div>
                <div class='kpi-delta' style='color:#8bafd4;'>Per month (Sep–Dec)</div>
            </div>""", unsafe_allow_html=True)

        # Forecast table
        st.markdown("<div class='section-header'>📋 Detailed Forecast Table</div>", unsafe_allow_html=True)
        forecast_df = pd.DataFrame({
            "Month": FUTURE_MONTHS,
            "Month_Num": FUTURE_NUMS,
            "Backlog": backlog_vals,
            "Random Forest": [round(p) for p in rf_preds],
            st.session_state.xgb_name: [round(p) for p in xgb_preds],
            st.session_state.arima_name: [round(p) for p in arima_preds],
            f"Best Model ({best})": [round(p) for p in best_preds],
        })

        styled = forecast_df.style.format({
            "Random Forest": "{:,}",
            st.session_state.xgb_name: "{:,}",
            st.session_state.arima_name: "{:,}",
            f"Best Model ({best})": "{:,}",
            "Backlog": "{:,}",
        }).background_gradient(subset=[f"Best Model ({best})"], cmap="Blues")
        st.dataframe(styled, use_container_width=True, height=240)

        # Export buttons
        st.markdown("<div class='section-header'>💾 Export Forecasts</div>", unsafe_allow_html=True)
        e1, e2 = st.columns(2)
        with e1:
            excel_bytes = export_excel(forecast_df, df[["Month", "Month_Num", "Shipment_Volume", "Backlog"]])
            st.download_button("📥 Download as Excel (.xlsx)", excel_bytes,
                               file_name="atm_shipment_forecast.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                               use_container_width=True)
        with e2:
            csv = forecast_df.to_csv(index=False)
            st.download_button("📄 Download as CSV", csv,
                               file_name="atm_shipment_forecast.csv",
                               mime="text/csv", use_container_width=True)


# ═══════════════════════════════════════════════════════════════
#  PAGE: VISUALIZATION
# ═══════════════════════════════════════════════════════════════

elif "Visualiz" in page:
    st.markdown("<div class='section-header'>📊 Interactive Visualizations</div>", unsafe_allow_html=True)

    if not st.session_state.trained:
        st.warning("⚠️ Please train models first in the **Model Training** tab.")
    else:
        df  = st.session_state.df
        bvs = st.session_state.backlog_vals or [415, 445, 475, 525]

        last_vol   = float(df["Shipment_Volume"].iloc[-1])
        last_lag   = float(df["Lag1"].iloc[-1])
        rf_preds   = forecast_ml(st.session_state.rf_model, last_vol, bvs, last_lag)
        xgb_preds  = forecast_ml(st.session_state.xgb_model, last_vol, bvs, last_lag)
        arima_preds= forecast_arima(st.session_state.arima_model)

        metrics = {
            "Random Forest": st.session_state.rf_rmse,
            st.session_state.xgb_name: st.session_state.xgb_rmse,
            st.session_state.arima_name: st.session_state.arima_rmse,
        }
        best = min(metrics, key=metrics.get)
        pred_map = {
            "Random Forest": rf_preds,
            st.session_state.xgb_name: xgb_preds,
            st.session_state.arima_name: arima_preds,
        }
        best_preds = pred_map[best]

        months_all = list(df["Month"].astype(str)) + FUTURE_MONTHS
        actual_all = list(df["Shipment_Volume"]) + [None] * 4
        pred_rf_all   = [None] * 8 + rf_preds
        pred_xgb_all  = [None] * 8 + xgb_preds
        pred_arima_all= [None] * 8 + arima_preds
        pred_best_all = [None] * 8 + best_preds

        # ── Chart 1: Main Forecast Line Chart ──
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(
            x=months_all, y=actual_all, mode="lines+markers+text",
            name="Actual (Jan–Aug)", line=dict(color="#64b5f6", width=3),
            marker=dict(size=9, symbol="circle"),
            text=[f"{v:,}" if v else "" for v in actual_all],
            textposition="top center", textfont=dict(size=9, color="#90caf9"),
        ))
        fig1.add_trace(go.Scatter(
            x=months_all, y=pred_best_all, mode="lines+markers+text",
            name=f"Forecast – {best}", line=dict(color="#43d98c", width=3, dash="dash"),
            marker=dict(size=10, symbol="diamond"),
            text=[f"{int(v):,}" if v else "" for v in pred_best_all],
            textposition="top center", textfont=dict(size=9, color="#43d98c"),
        ))
        fig1.add_vrect(x0="September", x1="December",
                       fillcolor="rgba(67,217,140,0.05)", line_width=0,
                       annotation_text="Forecast Zone", annotation_position="top left",
                       annotation_font_color="#43d98c")
        fig1.add_vline(x=7.5, line_dash="dot", line_color="#8bafd4", line_width=1.5)
        fig1.update_layout(
            template="plotly_dark", paper_bgcolor="#0d1b2a", plot_bgcolor="#112240",
            font=dict(color="#ccd6f6"), height=420,
            title=dict(text="📦 ATM Shipment Volume: Actual vs Forecast", font=dict(size=16)),
            xaxis=dict(title="Month", gridcolor="#1e3a5f", showgrid=True),
            yaxis=dict(title="Shipment Volume (Units)", gridcolor="#1e3a5f", showgrid=True),
            legend=dict(orientation="h", y=1.12, x=0.5, xanchor="center",
                        bgcolor="rgba(0,0,0,0)", bordercolor="#1e3a5f"),
            hovermode="x unified",
        )
        st.plotly_chart(fig1, use_container_width=True)

        # ── Chart 2: All-Model Comparison Forecast ──
        fig2 = go.Figure()
        colors = {"Random Forest": "#64b5f6", "XGBoost": "#ffb74d",
                  "Gradient Boosting": "#ffb74d", "ARIMA": "#f06292",
                  "Linear Trend (ARIMA fallback)": "#ce93d8"}
        for name, preds in pred_map.items():
            fig2.add_trace(go.Scatter(
                x=FUTURE_MONTHS, y=preds, mode="lines+markers+text",
                name=name, line=dict(width=2.5, color=colors.get(name, "#aaa")),
                marker=dict(size=10),
                text=[f"{int(v):,}" for v in preds],
                textposition="top center", textfont=dict(size=9),
            ))
        fig2.update_layout(
            template="plotly_dark", paper_bgcolor="#0d1b2a", plot_bgcolor="#112240",
            font=dict(color="#ccd6f6"), height=380,
            title=dict(text="🤖 Model Comparison – Forecast Sep–Dec", font=dict(size=16)),
            xaxis=dict(title="Month", gridcolor="#1e3a5f"),
            yaxis=dict(title="Predicted Shipment Volume", gridcolor="#1e3a5f"),
            legend=dict(orientation="h", y=1.12, x=0.5, xanchor="center",
                        bgcolor="rgba(0,0,0,0)"),
            hovermode="x unified",
        )
        st.plotly_chart(fig2, use_container_width=True)

        # ── Chart 3: Monthly Trend Area ──
        c1, c2 = st.columns(2)
        with c1:
            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(
                x=list(df["Month"]), y=list(df["Shipment_Volume"]),
                fill="tozeroy", mode="lines+markers",
                name="Shipment Volume",
                line=dict(color="#1e88e5", width=2.5),
                fillcolor="rgba(30,136,229,0.15)",
            ))
            fig3.update_layout(
                template="plotly_dark", paper_bgcolor="#0d1b2a", plot_bgcolor="#112240",
                font=dict(color="#ccd6f6"), height=320,
                title=dict(text="📈 Historical Shipment Trend", font=dict(size=14)),
                xaxis=dict(gridcolor="#1e3a5f"), yaxis=dict(gridcolor="#1e3a5f"),
                showlegend=False, margin=dict(t=50, b=30),
            )
            st.plotly_chart(fig3, use_container_width=True)

        with c2:
            fig4 = go.Figure()
            fig4.add_trace(go.Bar(
                x=FUTURE_MONTHS,
                y=[int(p) for p in best_preds],
                marker=dict(
                    color=[int(p) for p in best_preds],
                    colorscale="Blues", showscale=False,
                    line=dict(color="#1e88e5", width=1.5)
                ),
                text=[f"{int(p):,}" for p in best_preds],
                textposition="outside", textfont=dict(color="#90caf9"),
                name="Forecast",
            ))
            fig4.update_layout(
                template="plotly_dark", paper_bgcolor="#0d1b2a", plot_bgcolor="#112240",
                font=dict(color="#ccd6f6"), height=320,
                title=dict(text=f"🏆 Best Model Forecast ({best})", font=dict(size=14)),
                xaxis=dict(gridcolor="#1e3a5f"), yaxis=dict(gridcolor="#1e3a5f"),
                showlegend=False, margin=dict(t=50, b=30),
            )
            st.plotly_chart(fig4, use_container_width=True)

        # ── Chart 4: Backlog vs Volume ──
        fig5 = make_subplots(specs=[[{"secondary_y": True}]])
        fig5.add_trace(go.Bar(
            x=list(df["Month"]), y=list(df["Shipment_Volume"]),
            name="Shipment Volume", marker_color="rgba(30,136,229,0.7)",
        ), secondary_y=False)
        fig5.add_trace(go.Scatter(
            x=list(df["Month"]), y=list(df["Backlog"]),
            name="Backlog", mode="lines+markers",
            line=dict(color="#ffb74d", width=2.5), marker=dict(size=8),
        ), secondary_y=True)
        fig5.update_layout(
            template="plotly_dark", paper_bgcolor="#0d1b2a", plot_bgcolor="#112240",
            font=dict(color="#ccd6f6"), height=360,
            title=dict(text="📊 Shipment Volume vs Backlog (Jan–Aug)", font=dict(size=14)),
            xaxis=dict(gridcolor="#1e3a5f"),
            legend=dict(orientation="h", y=1.1, bgcolor="rgba(0,0,0,0)"),
            hovermode="x unified",
        )
        fig5.update_yaxes(title_text="Shipment Volume", gridcolor="#1e3a5f", secondary_y=False)
        fig5.update_yaxes(title_text="Backlog", gridcolor="#1e3a5f", secondary_y=True)
        st.plotly_chart(fig5, use_container_width=True)

