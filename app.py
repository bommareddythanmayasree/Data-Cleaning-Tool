import io
import json
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Optional profiling (wrapped in try/except so app still runs if unavailable)
try:
    from ydata_profiling import ProfileReport
    PROFILING_AVAILABLE = True
except Exception:
    PROFILING_AVAILABLE = False

# -----------------------------
# App Config & Utilities
# -----------------------------
st.set_page_config(page_title="🧹 AI-Powered Data Cleaning Tool", layout="wide")

def init_state():
    if "raw_df" not in st.session_state:
        st.session_state.raw_df = None
    if "df" not in st.session_state:
        st.session_state.df = None

@st.cache_data(show_spinner=False)
def read_file(upload):
    if upload.name.lower().endswith(".csv"):
        return pd.read_csv(upload)
    return pd.read_excel(upload)

def percent_missing(s: pd.Series) -> float:
    return float(s.isna().mean() * 100)

def detect_inconsistent_categories(df: pd.DataFrame, max_show=6):
    out = {}
    for c in df.select_dtypes(include=["object"]).columns:
        vals = df[c].dropna().astype(str)
        sample = vals.unique()[:max_show]
        out[c] = list(sample)
    return out

def suggest_actions(df: pd.DataFrame):
    suggestions = []
    # Missing value suggestions
    for col in df.columns:
        miss = percent_missing(df[col])
        if miss == 0:
            continue
        if miss > 40:
            suggestions.append({
                "column": col, "issue": "High missing ratio",
                "detail": f"{miss:.1f}% missing", "recommendation": "Consider dropping the column"
            })
        else:
            if pd.api.types.is_numeric_dtype(df[col]):
                suggestions.append({
                    "column": col, "issue": "Missing values",
                    "detail": f"{miss:.1f}% missing", "recommendation": "Impute with Median"
                })
            else:
                suggestions.append({
                    "column": col, "issue": "Missing values",
                    "detail": f"{miss:.1f}% missing", "recommendation": "Impute with Mode"
                })
    # Text standardization suggestions
    for col in df.select_dtypes(include=["object"]).columns:
        if df[col].dropna().str.contains(r"[A-Z]").any():
            suggestions.append({
                "column": col, "issue": "Inconsistent casing",
                "detail": "Mixed upper/lower case detected",
                "recommendation": "Standardize (lowercase + strip)"
            })
    # Duplicate suggestion
    dups = int(df.duplicated().sum())
    if dups > 0:
        suggestions.append({
            "column": "(rows)", "issue": "Duplicates",
            "detail": f"{dups} duplicate rows", "recommendation": "Drop duplicates (keep first)"
        })
    return suggestions

def zscore_outliers(s: pd.Series, thresh=3.0):
    if s.std(skipna=True) == 0 or s.dropna().empty:
        return pd.Series(False, index=s.index)
    z = (s - s.mean(skipna=True)) / s.std(skipna=True)
    return z.abs() > thresh

def iqr_outliers(s: pd.Series, factor=1.5):
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - factor * iqr, q3 + factor * iqr
    return (s < lower) | (s > upper)

def download_bytes_from_html(html_str: str) -> bytes:
    return html_str.encode("utf-8")

# -----------------------------
# Sidebar: Load & Save
# -----------------------------
init_state()
st.sidebar.header("⚙️ Controls")

uploaded = st.sidebar.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])
if uploaded:
    try:
        df = read_file(uploaded)
        st.session_state.raw_df = df.copy()
        st.session_state.df = df.copy()
    except Exception as e:
        st.sidebar.error(f"Failed to read file: {e}")

st.sidebar.write("---")
if st.session_state.df is not None:
    if st.sidebar.button("♻️ Reset to Original"):
        st.session_state.df = st.session_state.raw_df.copy()

# -----------------------------
# Header
# -----------------------------
st.title("🧹 AI-Powered Data Cleaning Tool")
st.caption("Clean, analyze, transform, and export datasets. Perfect for ML preprocessing & portfolio demos.")

if st.session_state.df is None:
    st.info("Upload a dataset from the left sidebar to get started.")
    st.stop()

df = st.session_state.df

# -----------------------------
# Tabs Layout
# -----------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Overview", "🧠 Smart Suggestions", "🧼 Cleaning", "🧪 Transformations", "📤 Export & Report"
])

# -----------------------------
# Tab 1: Overview / EDA
# -----------------------------
with tab1:
    st.subheader("Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{df.shape[0]:,}")
    c2.metric("Columns", f"{df.shape[1]:,}")
    c3.metric("Numeric", f"{df.select_dtypes(include=np.number).shape[1]}")
    c4.metric("Categorical", f"{df.select_dtypes(include='object').shape[1]}")

    st.markdown("#### Missing Values by Column")
    miss = df.isna().sum()
    miss = miss[miss > 0].sort_values(ascending=False)
    if miss.empty:
        st.success("No missing values detected ✅")
    else:
        st.dataframe(miss.to_frame("Missing Count"))

    st.markdown("#### Descriptive Statistics (Numeric)")
    if not df.select_dtypes(include=np.number).empty:
        st.dataframe(df.describe().T)

    st.markdown("#### Quick Visuals")
    num_cols = list(df.select_dtypes(include=np.number).columns)
    cat_cols = list(df.select_dtypes(exclude=np.number).columns)
    viz_c1, viz_c2 = st.columns(2)

    with viz_c1:
        if num_cols:
            col_num = st.selectbox("Histogram (numeric)", num_cols, key="hist_num")
            fig, ax = plt.subplots()
            df[col_num].dropna().plot(kind="hist", bins=30, ax=ax)
            ax.set_xlabel(col_num)
            ax.set_title(f"Histogram: {col_num}")
            st.pyplot(fig)
    with viz_c2:
        if cat_cols:
            col_cat = st.selectbox("Bar Plot (categorical)", cat_cols, key="bar_cat")
            fig2, ax2 = plt.subplots()
            df[col_cat].astype(str).value_counts().head(20).plot(kind="bar", ax=ax2)
            ax2.set_title(f"Top categories: {col_cat}")
            st.pyplot(fig2)

    if len(num_cols) >= 2:
        st.markdown("#### Correlation Heatmap")
        fig3, ax3 = plt.subplots()
        corr = df[num_cols].corr(numeric_only=True)
        im = ax3.imshow(corr, interpolation='nearest')
        ax3.set_xticks(range(len(num_cols)))
        ax3.set_yticks(range(len(num_cols)))
        ax3.set_xticklabels(num_cols, rotation=45, ha='right')
        ax3.set_yticklabels(num_cols)
        fig3.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
        st.pyplot(fig3)

# -----------------------------
# Tab 2: Smart Suggestions
# -----------------------------
with tab2:
    st.subheader("🧠 Suggested Fixes")
    sug = suggest_actions(df)
    if not sug:
        st.success("No obvious issues found. Your data looks great! ✅")
    else:
        st.write("These are automatic recommendations based on your dataset:")
        st.table(pd.DataFrame(sug))

    st.markdown("#### Sample Categories in Text Columns")
    st.json(detect_inconsistent_categories(df))

# -----------------------------
# Tab 3: Cleaning
# -----------------------------
with tab3:
    st.subheader("Missing Values")
    mv_strategy = st.selectbox(
        "Global strategy",
        ["Do Nothing", "Drop Rows with Any NA", "Fill Median (numeric) & Mode (categorical)", "Fill Mean (numeric) & Mode (categorical)"]
    )

    st.subheader("Duplicates")
    dup_action = st.selectbox("Duplicate rows", ["Do Nothing", "Drop duplicates (keep first)"])

    st.subheader("Text Standardization (categorical columns)")
    text_std = st.checkbox("Lowercase + Trim spaces", value=True)

    st.subheader("Outliers (numeric)")
    outlier_method = st.selectbox("Method", ["Do Nothing", "Z-score (>3)", "IQR (1.5×IQR)"])
    outlier_action = st.selectbox("Action", ["Mark only", "Remove rows with outliers", "Cap to bounds (IQR only)"])

    if st.button("Apply Cleaning", type="primary"):
        work = df.copy()

        # Missing values
        if mv_strategy == "Drop Rows with Any NA":
            work = work.dropna()
        elif mv_strategy.startswith("Fill"):
            num_cols = work.select_dtypes(include=np.number).columns
            cat_cols = work.select_dtypes(exclude=np.number).columns
            if "Median" in mv_strategy:
                work[num_cols] = work[num_cols].apply(lambda s: s.fillna(s.median()))
            else:
                work[num_cols] = work[num_cols].apply(lambda s: s.fillna(s.mean()))
            for c in cat_cols:
                if work[c].isna().any():
                    mode_val = work[c].mode(dropna=True)
                    if not mode_val.empty:
                        work[c] = work[c].fillna(mode_val[0])

        # Duplicates
        if dup_action.startswith("Drop"):
            before = work.shape[0]
            work = work.drop_duplicates()
            st.info(f"Dropped {before - work.shape[0]} duplicate rows.")

        # Text standardization
        if text_std:
            for c in work.select_dtypes(include="object").columns:
                work[c] = work[c].astype(str).str.strip().str.lower()

        # Outliers
        if outlier_method != "Do Nothing":
            num_cols = work.select_dtypes(include=np.number).columns
            total_flags = 0
            if outlier_method == "Z-score (>3)":
                for c in num_cols:
                    flags = zscore_outliers(work[c])
                    total_flags += int(flags.sum())
                    if outlier_action == "Remove rows with outliers":
                        work = work.loc[~flags]
                st.info(f"Z-score outliers flagged: {total_flags}")
            else:
                # IQR
                for c in num_cols:
                    flags = iqr_outliers(work[c])
                    total_flags += int(flags.sum())
                    if outlier_action == "Remove rows with outliers":
                        work = work.loc[~flags]
                    elif outlier_action == "Cap to bounds (IQR only)":
                        q1, q3 = work[c].quantile(0.25), work[c].quantile(0.75)
                        iqr = q3 - q1
                        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                        work[c] = work[c].clip(lower, upper)
                st.info(f"IQR outliers flagged: {total_flags}")

        st.session_state.df = work
        st.success("Cleaning applied ✔️")
        st.dataframe(work.head(10), use_container_width=True)

# -----------------------------
# Tab 4: Transformations
# -----------------------------
with tab4:
    st.subheader("Encoding")
    enc_choice = st.selectbox("Categorical encoding", ["None", "One-Hot encode (new columns)", "Label encode (overwrite)"])
    st.caption("One-Hot creates extra columns; Label Encoding overwrites string categories with integer codes.")

    st.subheader("Scaling (numeric)")
    scale_choice = st.selectbox("Scaler", ["None", "StandardScaler (z-score)", "MinMaxScaler (0-1)"])

    if st.button("Apply Transformations"):
        work = df.copy()

        cat_cols = list(work.select_dtypes(exclude=np.number).columns)
        num_cols = list(work.select_dtypes(include=np.number).columns)

        # Encoding
        if enc_choice == "One-Hot encode (new columns)" and cat_cols:
            ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            arr = ohe.fit_transform(work[cat_cols])
            ohe_cols = ohe.get_feature_names_out(cat_cols)
            work_ohe = pd.DataFrame(arr, columns=ohe_cols, index=work.index)
            work = pd.concat([work.drop(columns=cat_cols), work_ohe], axis=1)

        elif enc_choice == "Label encode (overwrite)" and cat_cols:
            for c in cat_cols:
                le = LabelEncoder()
                work[c] = le.fit_transform(work[c].astype(str))

        # Scaling
        if scale_choice != "None" and num_cols:
            if scale_choice.startswith("Standard"):
                scaler = StandardScaler()
            else:
                scaler = MinMaxScaler()
            # Recompute num_cols in case encoding changed structure
            num_cols_after = list(work.select_dtypes(include=np.number).columns)
            work[num_cols_after] = scaler.fit_transform(work[num_cols_after])

        st.session_state.df = work
        st.success("Transformations applied ✔️")
        st.dataframe(work.head(10), use_container_width=True)

# -----------------------------
# Tab 5: Export & Report
# -----------------------------
with tab5:
    st.subheader("Download Cleaned Dataset")
    csv_bytes = st.session_state.df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download CSV", data=csv_bytes, file_name="cleaned_dataset.csv", mime="text/csv")

    st.markdown("---")
    st.subheader("Auto EDA Report (HTML)")

    if PROFILING_AVAILABLE:
        st.caption("Generates a detailed HTML profiling report you can attach in your submission/README.")
        if st.button("Generate Report"):
            with st.spinner("Building report..."):
                profile = ProfileReport(
                    st.session_state.df,
                    title="Data Profile Report",
                    explorative=True,
                    minimal=True
                )
                buf = io.StringIO()
                profile.to_file(buf)  # writes HTML string when given a buffer-like object
                html_bytes = download_bytes_from_html(buf.getvalue())
                st.download_button("⬇️ Download EDA Report (HTML)",
                                   data=html_bytes,
                                   file_name="eda_report.html",
                                   mime="text/html")
                st.success("Report ready!")
    else:
        st.warning("`ydata-profiling` not installed. Add it to requirements to enable HTML report.")
        # Lightweight fallback mini-report
        st.markdown("##### Quick Summary (fallback)")
        quick = {
            "shape": st.session_state.df.shape,
            "columns": list(st.session_state.df.columns),
            "missing_per_column": st.session_state.df.isna().sum().to_dict()
        }
        st.json(quick)
