# app.py
# Visual 2: Overall Alternative Energy Adoption by Governorate (stacked; % math fixed)
# Visual 4: Town Alternative Energy Profile (compare towns)
# Town names cleaned from links/URIs.

import os
import re
import pandas as pd
import numpy as np
import streamlit as st
import altair as alt

st.set_page_config(
    page_title="Municipal Energy Explorer — Visuals 2 & 4",
    layout="wide",
    page_icon="🔌",
)

DATA_PATH_GUESS = "325 data.csv"

# ----------------------------
# Helpers
# ----------------------------
def extract_name_from_link(val: str) -> str:
    """Turn a URL/DBpedia URI into a readable name; otherwise return trimmed text."""
    if pd.isna(val):
        return val
    s = str(val).strip()
    if s.startswith(("http://", "https://")) or "dbpedia.org" in s:
        seg = s.rsplit("/", 1)[-1]
        seg = seg.split("?")[0].split("#")[0].replace("_", " ")
        seg = re.sub(r"\s*\((?:[^)]*)\)\s*$", "", seg).strip()
        return seg
    return s

@st.cache_data(show_spinner=False)
def load_data(uploaded_file=None):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        if not os.path.exists(DATA_PATH_GUESS):
            st.error("CSV not found. Upload from the sidebar or place '325 data.csv' next to app.py.")
            st.stop()
        df = pd.read_csv(DATA_PATH_GUESS)

    df.columns = [c.strip() for c in df.columns]

    # Clean Town column
    town_col = None
    for cand in ["Town", "town", "Municipality", "municipality"]:
        if cand in df.columns:
            town_col = cand
            break
    if town_col is None:
        st.error("Could not find a 'Town' column. Please ensure your CSV has one.")
        st.stop()
    df[town_col] = df[town_col].apply(extract_name_from_link)

    # Governorate
    if "refArea" in df.columns:
        def extract_gov(x):
            if pd.isna(x): return np.nan
            seg = str(x).rsplit("/", 1)[-1].replace("_", " ")
            seg = seg.replace("Governorate", "").strip()
            seg = re.sub(r"\s*\((?:[^)]*)\)\s*$", "", seg).strip()
            return seg
        df["Governorate"] = df["refArea"].apply(extract_gov)
    elif "Governorate" not in df.columns:
        df["Governorate"] = np.nan

    COLS = {
        "town": town_col,
        "solar": "Type of alternative energy used - solar energy",
        "wind": "Type of alternative energy used - wind energy",
        "hydro": "Type of alternative energy used - hydropower (water use)",
        "other": "Type of alternative energy used - other",
    }
    type_cols = [COLS["solar"], COLS["wind"], COLS["hydro"], COLS["other"]]
    for c in type_cols:
        if c not in df.columns:
            df[c] = 0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

    return df, COLS

# ----------------------------
# Sidebar (upload)
# ----------------------------
st.sidebar.header("Data")
uploaded = st.sidebar.file_uploader("Upload CSV (optional)", type=["csv"])
df, COLS = load_data(uploaded)

st.title("🔌 Municipal Energy Explorer")
st.caption("Only Visual 2 (overall alt-energy adoption) and Visual 4 (town profiles). Town names are cleaned from links.")

# ======================================================
# VISUAL 2 — Overall Alternative Energy Adoption by Governorate (fixed % math)
# ======================================================
st.markdown("### 2) Overall Alternative Energy Adoption — by Governorate")

energy_map_display = {
    "Solar": COLS["solar"],
    "Wind":  COLS["wind"],
    "Hydro": COLS["hydro"],
    "Other": COLS["other"],
}

left, mid = st.columns([2, 2])
with left:
    normalize = st.checkbox(
        "Show as % of towns using any alternative energy",
        value=True,
        help="Each adopting town contributes a total of 1, split across the types it uses."
    )
with mid:
    sort_by = st.selectbox(
        "Sort governorates by",
        ["Total adopting towns", "Solar", "Wind", "Hydro", "Other", "Alphabetical"],
        index=0
    )

type_cols = list(energy_map_display.values())

# Raw counts per governorate (for counts view + sorting)
grp_counts = (
    df.groupby("Governorate", dropna=False)[type_cols]
    .sum()
    .reset_index()
    .rename(columns={
        COLS["solar"]: "Solar",
        COLS["wind"]: "Wind",
        COLS["hydro"]: "Hydro",
        COLS["other"]: "Other",
    })
)

# Number of adopting towns per governorate
df_any = df.assign(types_count=df[type_cols].sum(axis=1))
any_mask = df_any["types_count"] > 0
adopting_per_gov = (
    df_any.groupby("Governorate", dropna=False)["types_count"]
    .apply(lambda s: (s > 0).sum())
    .reset_index()
    .rename(columns={"types_count": "Total adopting towns"})
)

grp_counts = grp_counts.merge(adopting_per_gov, on="Governorate", how="left")

# Normalized contributions (each adopting town contributes 1 split across its present types)
if normalize:
    df_norm = df_any.loc[any_mask, ["Governorate", *type_cols, "types_count"]].copy()
    for c in type_cols:
        df_norm[c] = df_norm[c] / df_norm["types_count"]

    grp_norm = (
        df_norm.groupby("Governorate", dropna=False)[type_cols]
        .sum()
        .reset_index()
        .rename(columns={
            COLS["solar"]: "Solar",
            COLS["wind"]: "Wind",
            COLS["hydro"]: "Hydro",
            COLS["other"]: "Other",
        })
    ).merge(adopting_per_gov, on="Governorate", how="left")

    tall = grp_norm.melt(
        id_vars=["Governorate", "Total adopting towns"],
        value_vars=["Solar", "Wind", "Hydro", "Other"],
        var_name="Energy type",
        value_name="town_equivalents"
    )
    tall["pct"] = np.where(
        tall["Total adopting towns"] > 0,
        tall["town_equivalents"] / tall["Total adopting towns"],
        0.0
    )
else:
    tall = grp_counts.melt(
        id_vars=["Governorate", "Total adopting towns"],
        value_vars=["Solar", "Wind", "Hydro", "Other"],
        var_name="Energy type",
        value_name="count"
    )

# Sorting
if sort_by == "Alphabetical":
    sort_order = sorted(grp_counts["Governorate"].fillna("Unknown").tolist())
elif sort_by == "Total adopting towns":
    sort_order = grp_counts.sort_values("Total adopting towns", ascending=False)["Governorate"].tolist()
else:
    metric_col = {"Solar": "Solar", "Wind": "Wind", "Hydro": "Hydro", "Other": "Other"}[sort_by]
    sort_order = grp_counts.sort_values(metric_col, ascending=False)["Governorate"].tolist()

# Chart
if normalize:
    y = alt.Y("pct:Q", axis=alt.Axis(format="%"), title="Share of adopting towns")
    tooltip = [
        "Governorate:N", "Energy type:N",
        alt.Tooltip("pct:Q", format=".0%"),
        "Total adopting towns:Q",
    ]
else:
    y = alt.Y("count:Q", title="Number of towns")
    tooltip = ["Governorate:N", "Energy type:N", "count:Q", "Total adopting towns:Q"]

chart2 = (
    alt.Chart(tall)
    .mark_bar()
    .encode(
        x=alt.X("Governorate:N", sort=sort_order, title="Governorate"),
        y=y,
        color=alt.Color("Energy type:N", legend=alt.Legend(title="Energy type")),
        tooltip=tooltip,
    )
    .properties(height=460)
)

st.altair_chart(chart2, use_container_width=True)

st.divider()

# ======================================================
# VISUAL 4 — Town-level Alternative Energy Profile (Interactive)
# ======================================================
st.markdown("### 4) Town Alternative Energy Profile")

energy_map_simple = {
    "Solar": COLS["solar"],
    "Wind":  COLS["wind"],
    "Hydro": COLS["hydro"],
    "Other": COLS["other"],
}
towns = sorted(df[COLS["town"]].dropna().unique().tolist())

c1, c2, c3 = st.columns([2, 2, 2])
with c1:
    selected_town = st.selectbox("Choose a town", options=towns)
with c2:
    compare_town = st.selectbox("Compare with (optional)", options=["(None)"] + towns, index=0)
with c3:
    types_chosen = st.multiselect(
        "Energy types to display",
        options=list(energy_map_simple.keys()),
        default=list(energy_map_simple.keys()),
    )

def build_town_profile(frame, town_name, label):
    row = frame[frame[COLS["town"]] == town_name].head(1)
    if row.empty:
        return pd.DataFrame(columns=["Town", "Type", "Present"])
    records = []
    for t_label, colname in energy_map_simple.items():
        present = int(row[colname].iloc[0])
        records.append({"Town": label, "Type": t_label, "Present": present})
    return pd.DataFrame(records)

profile_main = build_town_profile(df, selected_town, selected_town)
if compare_town != "(None)":
    profile_cmp = build_town_profile(df, compare_town, compare_town)
    prof_all = pd.concat([profile_main, profile_cmp], ignore_index=True)
else:
    prof_all = profile_main.copy()

if types_chosen:
    prof_all = prof_all[prof_all["Type"].isin(types_chosen)]
else:
    st.info("Select at least one energy type to display.")
    prof_all = prof_all.iloc[0:0]

chart4 = (
    alt.Chart(prof_all)
    .mark_bar()
    .encode(
        x=alt.X("Type:N", title="Energy Type"),
        y=alt.Y("Present:Q", title="Presence (0/1)", scale=alt.Scale(domain=[0, 1])),
        column=alt.Column("Town:N", title=None),
        color=alt.Color("Type:N", legend=None),
        tooltip=["Town:N", "Type:N", "Present:Q"],
    )
    .properties(height=300)
)
st.altair_chart(chart4, use_container_width=True)

with st.expander("Summary"):
    if not prof_all.empty:
        by_town = (
            prof_all.groupby(["Town"])["Present"]
            .sum()
            .reset_index()
            .rename(columns={"Present": "Number of energy types present"})
        )
        st.dataframe(by_town, use_container_width=True)
    else:
        st.write("No types selected.")

st.caption("Tip: Use the sidebar to upload a new CSV anytime. Town names are auto-cleaned from links.")
