# app.py
# Visual 2: Overall Alternative Energy Adoption by Governorate (stacked, like screenshot)
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
            st.error("CSV not found. Upload it from the sidebar or place '325 data.csv' next to app.py.")
            st.stop()
        df = pd.read_csv(DATA_PATH_GUESS)

    df.columns = [c.strip() for c in df.columns]

    # Clean Town
    town_col = None
    for cand in ["Town", "town", "Municipality", "municipality"]:
        if cand in df.columns:
            town_col = cand
            break
    if town_col is None:
        st.error("Could not find a 'Town' column. Please ensure your CSV has a Town/municipality column.")
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
        # Lighting (not used now for V2, but kept for compatibility)
        "light_good": "State of the lighting network - good",
        "light_ok": "State of the lighting network - acceptable",
        "light_bad": "State of the lighting network - bad",
        # Alternative energy
        "solar": "Type of alternative energy used - solar energy",
        "wind": "Type of alternative energy used - wind energy",
        "hydro": "Type of alternative energy used - hydropower (water use)",
        "other": "Type of alternative energy used - other",
    }
    needed = [COLS["solar"], COLS["wind"], COLS["hydro"], COLS["other"]]
    for c in needed:
        if c not in df.columns:
            df[c] = 0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

    return df, COLS

# ----------------------------
# Sidebar
# ----------------------------
st.sidebar.header("Data")
uploaded = st.sidebar.file_uploader("Upload CSV (optional)", type=["csv"])
df, COLS = load_data(uploaded)

st.title("🔌 Municipal Energy Explorer")
st.caption("Showing Visual 2 (overall alt energy adoption) and Visual 4 (town profiles). Town names are cleaned from links.")

# ======================================================
# VISUAL 2 — Overall Alternative Energy Adoption by Governorate
# (stacked bars of Solar/Wind/Hydro/Other counts; % toggle + sort control)
# ======================================================
st.markdown("### 2) Overall Alternative Energy Adoption — by Governorate")

energy_map = {
    "solar energy": COLS["solar"],
    "wind energy": COLS["wind"],
    "hydropower (water use)": COLS["hydro"],
    "other": COLS["other"],
}

left, mid, right = st.columns([2, 2, 1])
with left:
    normalize = st.checkbox("Show as % of towns using any alternative energy", value=False,
                            help="Normalize each governorate to 100% of towns that use ≥1 energy type.")
with mid:
    sort_by = st.selectbox(
        "Sort governorates by",
        ["Total adopting towns", "Solar", "Wind", "Hydro", "Other", "Alphabetical"],
        index=0
    )
with right:
    show_values = st.checkbox("Show value labels", value=False)

# Compute per governorate how many towns use each energy type (0/1 columns)
grp = (
    df.groupby("Governorate", dropna=False)[list(energy_map.values())]
    .sum()
    .reset_index()
    .rename(columns={
        COLS["solar"]: "Solar",
        COLS["wind"]: "Wind",
        COLS["hydro"]: "Hydro",
        COLS["other"]: "Other",
    })
)
# Total adopting towns (town with ≥1 type)
any_adopt = (df.assign(any_type=df[list(energy_map.values())].sum(axis=1).clip(upper=1))
             .groupby("Governorate", dropna=False)["any_type"].sum()
             .reset_index()
             .rename(columns={"any_type": "Total adopting towns"}))
grp = grp.merge(any_adopt, on="Governorate", how="left")

# Melt for stacked bars
tall = grp.melt(
    id_vars=["Governorate", "Total adopting towns"],
    value_vars=["Solar", "Wind", "Hydro", "Other"],
    var_name="Energy type",
    value_name="count"
)

# Normalize to percent of adopting towns if requested
if normalize:
    tall["pct"] = np.where(tall["Total adopting towns"] > 0,
                           tall["count"] / tall["Total adopting towns"], 0.0)

# Sorting
if sort_by == "Alphabetical":
    sort_order = sorted(grp["Governorate"].fillna("Unknown").tolist())
elif sort_by == "Total adopting towns":
    sort_order = grp.sort_values("Total adopting towns", ascending=False)["Governorate"].tolist()
else:
    col = {"Solar": "Solar", "Wind": "Wind", "Hydro": "Hydro", "Other": "Other"}[sort_by]
    sort_order = grp.sort_values(col, ascending=False)["Governorate"].tolist()

# Build chart
if normalize:
    y = alt.Y("pct:Q", axis=alt.Axis(format="%"), title="Share of adopting towns")
    tooltip = ["Governorate:N", "Energy type:N", "count:Q", "Total adopting towns:Q", alt.Tooltip("pct:Q", format=".0%")]
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

# Optional value labels
if show_values and not normalize:
    labels = (
        alt.Chart(tall)
        .mark_text(dy=-4)
        .encode(
            x=alt.X("Governorate:N", sort=sort_order),
            y=alt.Y("count:Q"),
            detail="Energy type:N",
            text=alt.Text("count:Q")
        )
    )
    st.altair_chart(chart2 + labels, use_container_width=True)

st.divider()

# ======================================================
# VISUAL 4 — Town-level Alternative Energy Profile (Interactive)
# ======================================================
st.markdown("### 4) Town Alternative Energy Profile")

energy_map_simple = {
    "Solar": COLS["solar"],
    "Wind": COLS["wind"],
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
