# app.py
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from pathlib import Path

st.set_page_config(page_title="Lebanon Energy — Vis 2 & 3", page_icon="⚡", layout="wide")
st.title("⚡ Lebanon Energy — Interactive Visuals (Visual 2 & Visual 3)")

# ---------- Helpers ----------
@st.cache_data
def load_csv_auto(uploaded_file):
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    # Fallback: try local file named exactly like your upload
    p = Path("325 data.csv")
    if p.exists():
        return pd.read_csv(p)
    return None

def coerce_bool(s: pd.Series) -> pd.Series:
    if s is None:
        return pd.Series([False]*0, dtype=bool)
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False)
    s_num = pd.to_numeric(s, errors="coerce")
    if s_num.notna().any():
        return (s_num.fillna(0) != 0)
    s_str = s.astype(str).str.strip().str.lower()
    return s_str.isin({"yes","y","true","t","1"})

def derive_onehot_label(row: pd.Series, mapping: dict, default="Unknown"):
    for label, col in mapping.items():
        if col and col in row.index:
            v = row[col]
            if pd.notna(v) and coerce_bool(pd.Series([v])).iloc[0]:
                return label
    return default

# ---------- Load data ----------
with st.sidebar:
    st.header("📁 Data")
    uploaded = st.file_uploader("Upload CSV", type=["csv"], help="If omitted, the app will try to read '325 data.csv' in the same folder.")

df = load_csv_auto(uploaded)
if df is None:
    st.info("⬆️ Upload your CSV (or place a file named **325 data.csv** next to app.py).")
    st.stop()

# ---------- Column names tailored to your file ----------
# These are the exact columns found in your sample CSV
COL_GOV   = "refArea_clean" if "refArea_clean" in df.columns else "refArea"
COL_TOWN  = "Town" if "Town" in df.columns else (df.columns[0])

COL_EXISTS_YES = "Existence of alternative energy - exists"
COL_EXISTS_NO  = "Existence of alternative energy - does not exist"

COL_GRID_GOOD = "State of the power grid - good"
COL_GRID_OK   = "State of the power grid - acceptable"
COL_GRID_BAD  = "State of the power grid - bad"

ENERGY_COLS = [c for c in [
    "Type of alternative energy used - solar energy",
    "Type of alternative energy used - wind energy",
    "Type of alternative energy used - hydropower (water use)",
    "Type of alternative energy used - other",
] if c in df.columns]

# Guardrails if any column missing
for required in [COL_GOV, COL_TOWN, COL_EXISTS_YES, COL_EXISTS_NO, COL_GRID_GOOD, COL_GRID_OK, COL_GRID_BAD]:
    if required not in df.columns:
        st.error(f"Missing required column: **{required}**")
        st.stop()
if not ENERGY_COLS:
    st.error("No energy-type columns found. Expected any of: Solar, Wind, Hydropower, Other (as in your CSV).")
    st.stop()

# ---------- Derivations ----------
df = df.copy()

# Adoption %: 100 if exists, 0 if explicitly does-not-exist, else NaN
exists_yes = coerce_bool(df[COL_EXISTS_YES])
exists_no  = coerce_bool(df[COL_EXISTS_NO])
df["_adoption_pct"] = np.where(exists_yes, 100.0, np.where(exists_no, 0.0, np.nan))

# Grid label from one-hot flags
grid_map = {"Good": COL_GRID_GOOD, "Acceptable": COL_GRID_OK, "Bad": COL_GRID_BAD}
df["_grid_label"] = df.apply(lambda r: derive_onehot_label(r, grid_map, default="Unknown"), axis=1)

# ---------- Filters ----------
govs = sorted(df[COL_GOV].dropna().astype(str).unique().tolist())
with st.sidebar:
    st.header("🔎 Filters")
    sel_govs = st.multiselect("Filter by governorate", options=govs, default=govs)
    min_adopt = st.slider("Minimum adoption rate (%)", 0, 100, 0, 1)
    st.markdown("---")
    st.subheader("📊 Visual 2 Options")
    stack_mode = st.radio("Stack mode", ["Counts", "Percent"], horizontal=True)

mask = df[COL_GOV].astype(str).isin(sel_govs) & (df["_adoption_pct"].fillna(-1) >= min_adopt)
df_filt = df.loc[mask].copy()

tab2, tab3 = st.tabs(["📊 Visual 2 — Stacked Bar by Governorate", "📈 Visual 3 — Adoption vs Grid"])

# ---------- Visual 2 ----------
with tab2:
    st.subheader("Visual 2 — Alternative Energy by Governorate (Stacked)")
    # Build long-form counts per governorate × energy type
    long_frames = []
    for c in ENERGY_COLS:
        counts = (
            df_filt.assign(_flag=coerce_bool(df_filt[c]))
                   .groupby(COL_GOV, dropna=False)["_flag"]
                   .sum()
                   .rename("Count")
                   .reset_index()
        )
        counts["Energy Type"] = c.replace("Type of alternative energy used - ", "").title()
        long_frames.append(counts)
    long_df = pd.concat(long_frames, ignore_index=True)

    # Sort governorates by total count desc
    totals = long_df.groupby(COL_GOV)["Count"].sum().sort_values(ascending=False).index.tolist()

    y_field = alt.Y(
        "Count:Q",
        stack=("normalize" if stack_mode == "Percent" else None),
        title=("Share of towns (%)" if stack_mode == "Percent" else "Number of towns")
    )

    chart_v2 = (
        alt.Chart(long_df)
        .mark_bar()
        .encode(
            x=alt.X(f"{COL_GOV}:N", sort=totals, title="Governorate"),
            y=y_field,
            color=alt.Color("Energy Type:N", legend=alt.Legend(title="Energy Type")),
            order=alt.Order("Energy Type:N"),
            tooltip=[
                alt.Tooltip(f"{COL_GOV}:N", title="Governorate"),
                alt.Tooltip("Energy Type:N"),
                alt.Tooltip("Count:Q"),
            ],
        )
        .properties(height=420)
        .interactive()
    )

    # Click legend to isolate
    selection = alt.selection_point(fields=["Energy Type"], bind="legend")
    chart_v2 = chart_v2.add_params(selection).transform_filter(selection)

    st.altair_chart(chart_v2, use_container_width=True)

# ---------- Visual 3 ----------
with tab3:
    st.subheader("Visual 3 — Connectivity vs Alternative Energy Adoption")
    st.caption("Adoption derived from existence flags: 100% if exists, 0% if does not exist.")
    if df_filt.empty:
        st.warning("No data after filters. Try including more governorates or lowering the adoption threshold.")
    else:
        color_by = st.selectbox("Color by", options=["None", "Governorate"], index=1)
        enc = {
            "x": alt.X("_grid_label:N", title="Grid state"),
            "y": alt.Y("_adoption_pct:Q", title="Adoption rate (%)"),
            "tooltip": [
                alt.Tooltip(COL_TOWN + ":N", title="Town") if COL_TOWN in df_filt.columns else alt.Tooltip(COL_GOV + ":N", title="Governorate"),
                alt.Tooltip(COL_GOV + ":N", title="Governorate"),
                alt.Tooltip("_grid_label:N", title="Grid state"),
                alt.Tooltip("_adoption_pct:Q", title="Adoption (%)", format=".0f"),
            ],
        }
        if color_by == "Governorate":
            enc["color"] = alt.Color(COL_GOV + ":N", title="Governorate")

        points = alt.Chart(df_filt).mark_circle(size=80, opacity=0.7).encode(**enc)

        # Boxplot overlay to summarize distribution per grid state
        box = (
            alt.Chart(df_filt)
            .mark_boxplot(opacity=0.3)
            .encode(x=alt.X("_grid_label:N", title="Grid state"),
                    y=alt.Y("_adoption_pct:Q", title="Adoption rate (%)"))
        )

        chart_v3 = (points + box).properties(height=500).interactive()
        st.altair_chart(chart_v3, use_container_width=True)
