import pandas as pd
import streamlit as st

st.set_page_config(page_title="🧹 Data Cleaning Tool", layout="wide")

st.title("🧹 AI-Powered Data Cleaning Tool")

uploaded_file = st.file_uploader("Upload your CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file:
    # Load file
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head(10))

    # Data Quality Report
    st.subheader("🔍 Data Quality Check")

    missing = df.isnull().sum()
    duplicates = df.duplicated().sum()
    inconsistent = {col: df[col].unique()[:5] for col in df.select_dtypes(include=['object']).columns}

    st.write(f"✅ Shape: {df.shape}")
    st.write("⚠️ Missing Values:", missing[missing > 0])
    st.write(f"⚠️ Duplicate Rows: {duplicates}")
    st.write("⚠️ Sample Categories in Text Columns:", inconsistent)

    # Cleaning Options
    st.subheader("🛠️ Cleaning Options")

    option_missing = st.selectbox("Handle Missing Values", ["Do Nothing", "Drop Rows", "Fill Mean", "Fill Median", "Fill Mode"])
    option_dup = st.selectbox("Handle Duplicates", ["Do Nothing", "Drop Duplicates"])

    if st.button("Apply Cleaning"):
        df_clean = df.copy()

        # Handle missing values
        if option_missing == "Drop Rows":
            df_clean = df_clean.dropna()
        elif option_missing == "Fill Mean":
            df_clean = df_clean.fillna(df_clean.mean(numeric_only=True))
        elif option_missing == "Fill Median":
            df_clean = df_clean.fillna(df_clean.median(numeric_only=True))
        elif option_missing == "Fill Mode":
            for col in df_clean.columns:
                df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])

        # Handle duplicates
        if option_dup == "Drop Duplicates":
            df_clean = df_clean.drop_duplicates()

        st.subheader("📊 Cleaned Dataset Preview")
        st.dataframe(df_clean.head(10))

        st.download_button(
            label="⬇️ Download Cleaned Dataset",
            data=df_clean.to_csv(index=False).encode('utf-8'),
            file_name="cleaned_dataset.csv",
            mime="text/csv"
        )
