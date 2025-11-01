
import io
import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

st.set_page_config(page_title="Scaling Map — 3D Ridges", layout="wide")
st.title("Scaling Map — 3D Ridges (Green = good, Red = bad)")

DEFAULT_DATA_PATH = os.getenv("SCALING_DATA_PATH", "data/Scaling_Surface_Template.xlsx")
DEFAULT_DATA_URL  = os.getenv("SCALING_DATA_URL", "")  # optional raw URL

# ---------------------- Data loaders ----------------------
@st.cache_data(show_spinner=False)
def _clean(df: pd.DataFrame) -> pd.DataFrame:
    text_cols = [
        "Process_ID","Process_Name","Process_Area","Metric_Type","Observation_Date","Lot_ID",
        "Owner","Subprocess","Current_State","Trigger_Type","Actor","Data_Coupling","State_Coupling",
        "Regulatory_Constraint","Target_State","Target_Intervention"
    ]
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    for num in ["Lot_Size_Units","Metric_Value","Automation_Readiness"]:
        if num in df.columns:
            df[num] = pd.to_numeric(df[num], errors="coerce")
    return df

@st.cache_data(show_spinner=False)
def load_from_path(path_str: str) -> pd.DataFrame:
    if not os.path.exists(path_str):
        raise FileNotFoundError(path_str)
    if path_str.lower().endswith(".csv"):
        df = pd.read_csv(path_str)
    else:
        xls = pd.ExcelFile(path_str)
        sheet = "ScalingData" if "ScalingData" in xls.sheet_names else xls.sheet_names[0]
        df = pd.read_excel(xls, sheet_name=sheet)
    return _clean(df)

@st.cache_data(show_spinner=False)
def load_from_url(url: str) -> pd.DataFrame:
    import urllib.request
    with urllib.request.urlopen(url) as resp:
        data = resp.read()
    try:
        df = pd.read_excel(io.BytesIO(data), sheet_name="ScalingData")
    except Exception:
        df = pd.read_csv(io.BytesIO(data))
    return _clean(df)

def compute_scaling(df: pd.DataFrame, metric_type: str) -> pd.DataFrame:
    dfm = df[df["Metric_Type"] == metric_type].dropna(subset=["Process_ID","Lot_Size_Units","Metric_Value"]).copy()
    results = []
    for pid, g in dfm.groupby("Process_ID"):
        if g["Lot_Size_Units"].nunique() < 2:
            continue
        x = g["Lot_Size_Units"].values.astype(float)
        y = g["Metric_Value"].values.astype(float)
        A = np.vstack([x, np.ones_like(x)]).T
        slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
        yhat = intercept + slope * x
        ss_res = np.sum((y - yhat) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2) if len(y) > 1 else 0.0
        r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else np.nan

        meta = {col: g[col].iloc[0] for col in g.columns if col not in ["Lot_Size_Units","Metric_Value","Observation_Date","Lot_ID","Notes"]}
        results.append({
            "Process_ID": pid,
            "Process_Name": meta.get("Process_Name", pid),
            "Process_Area": meta.get("Process_Area", ""),
            "Current_State": meta.get("Current_State", ""),
            "Automation_Readiness": pd.to_numeric(meta.get("Automation_Readiness", np.nan), errors="coerce"),
            "Regulatory_Constraint": meta.get("Regulatory_Constraint", ""),
            "Slope_per_Unit": float(slope),
            "Intercept": float(intercept),
            "R2": float(r2),
            "N_Obs": int(len(g))
        })
    res = pd.DataFrame(results)
    if res.empty:
        return res

    # Normalize slope into 0–1 via robust percentiles
    sl = res["Slope_per_Unit"].values
    p5, p95 = np.nanpercentile(sl, 5), np.nanpercentile(sl, 95)
    rng = (p95 - p5) if (p95 - p5) > 1e-9 else 1.0
    res["Scaling_Score_0to1"] = np.clip((sl - p5) / rng, 0, 1)

    # Priority score
    reg_weight = res["Regulatory_Constraint"].map({"Low":1.0,"Medium":1.3,"High":1.7}).fillna(1.2)
    readiness = pd.to_numeric(res["Automation_Readiness"], errors="coerce").fillna(3.0)
    res["Priority_Score"] = res["Scaling_Score_0to1"] * readiness / reg_weight
    return res

def ridge_data(df_raw: pd.DataFrame, processes: list, metric_type: str):
    dfm = df_raw[df_raw["Metric_Type"] == metric_type].dropna(subset=["Process_ID","Lot_Size_Units","Metric_Value"]).copy()
    dfm = dfm[dfm["Process_ID"].isin(processes)]
    lot_sizes = np.sort(dfm["Lot_Size_Units"].unique())
    Z = np.full((len(lot_sizes), len(processes)), np.nan)
    for j, pid in enumerate(processes):
        for i, ls in enumerate(lot_sizes):
            vals = dfm[(dfm["Process_ID"] == pid) & (dfm["Lot_Size_Units"] == ls)]["Metric_Value"].values
            if len(vals) > 0:
                Z[i, j] = float(np.mean(vals))
    return lot_sizes, Z

# ---------------------- Data source controls ----------------------
st.sidebar.header("Data Source")
mode = st.sidebar.radio("Source", ["Repo default", "Upload"], index=0)
repo_path = st.sidebar.text_input("Repo path", value=DEFAULT_DATA_PATH)
repo_url = st.sidebar.text_input("Raw URL (optional)", value=DEFAULT_DATA_URL)

uploaded = st.file_uploader("Upload .xlsx or .csv", type=["xlsx","csv"]) if mode == "Upload" else None

df = None
if mode == "Repo default":
    try:
        df = load_from_path(repo_path)
        st.success(f"Loaded repo data from: {repo_path} (rows={len(df)})")
    except Exception as e:
        st.warning(f"Path load failed: {e}")
        if repo_url:
            try:
                df = load_from_url(repo_url)
                st.success(f"Loaded repo data from URL (rows={len(df)})")
            except Exception as e2:
                st.error(f"URL load failed: {e2}")
else:
    if uploaded is not None:
        try:
            if uploaded.name.lower().endswith(".csv"):
                df = pd.read_csv(uploaded)
            else:
                xls = pd.ExcelFile(uploaded)
                sheet = "ScalingData" if "ScalingData" in xls.sheet_names else xls.sheet_names[0]
                df = pd.read_excel(xls, sheet_name=sheet)
            df = _clean(df)
            st.success(f"Loaded uploaded data (rows={len(df)})")
        except Exception as e:
            st.error(f"Upload load failed: {e}")

if df is None or df.empty:
    st.info("Load a valid dataset to continue.")
    st.stop()

# ---------------------- Controls ----------------------
metric = st.selectbox("Metric_Type", sorted(df["Metric_Type"].dropna().unique().tolist()))
area_filter = st.multiselect("Filter by Process_Area (optional)", sorted(df["Process_Area"].dropna().unique().tolist()))
df_f = df[df["Process_Area"].isin(area_filter)] if area_filter else df

res = compute_scaling(df_f, metric)
if res.empty:
    st.warning("Not enough data to compute scaling. Each Process_ID needs ≥ 2 distinct Lot_Size_Units.")
    st.stop()

sort_key = st.selectbox("Sort by", ["Scaling_Score_0to1", "Priority_Score"])
top_n = st.slider("Show Top-N processes", min_value=5, max_value=min(50, len(res)), value=min(20, len(res)))

# ---------------------- Top-N bars ----------------------
st.subheader("Top-N Processes (sorted)")
top = res.sort_values(sort_key, ascending=False).head(top_n)

fig, ax = plt.subplots(figsize=(10, max(4, 0.35 * len(top))))
ax.barh(top["Process_Name"], top[sort_key].values)
ax.invert_yaxis()
ax.set_xlabel(sort_key.replace("_", " "))
ax.set_ylabel("Process")
ax.set_title("Scaling / Priority — Top-N")
st.pyplot(fig)

# ---------------------- 3D Ridges ----------------------
st.subheader("3D Ridges — per-process profiles (Green=good, Red=bad)")

default_subset = top["Process_ID"].tolist()
subset = st.multiselect("Select processes (max 20 recommended)", res["Process_ID"].tolist(), default=default_subset[:12])

if subset:
    lot_sizes, Z = ridge_data(df_f, subset, metric)
    if len(lot_sizes) > 0:
        score_map = dict(zip(res["Process_ID"], res["Scaling_Score_0to1"]))
        norm = Normalize(vmin=0.0, vmax=1.0)
        cmap = plt.get_cmap("RdYlGn_r")

        fig3d = plt.figure(figsize=(11, 7))
        ax3d = fig3d.add_subplot(111, projection="3d")

        plotted_any = False
        for j, pid in enumerate(subset):
            y_vals = Z[:, j]
            if np.all(np.isnan(y_vals)):
                continue
            plotted_any = True
            color = cmap(norm(score_map.get(pid, 0.0)))
            ax3d.plot(lot_sizes, y_vals, zs=j, zdir='x', linewidth=2.0, color=color)

        ax3d.set_xticks(np.arange(len(subset)))
        ax3d.set_xticklabels(subset, rotation=45, ha="right")
        ax3d.set_xlabel("Process_ID")
        ax3d.set_ylabel("Lot Size")
        ax3d.set_zlabel(metric)
        ax3d.set_title("3D Ridges — Metric vs Lot Size by Process")

        if plotted_any:
            sm = ScalarMappable(norm=norm, cmap=cmap)
            sm.set_array([])
            fig3d.colorbar(sm, ax=ax3d, shrink=0.6, pad=0.1).set_label("Scaling Score (0 = good, 1 = bad)")

        st.pyplot(fig3d)
    else:
        st.info("Add more observations across multiple lot sizes to plot ridges.")
else:
    st.info("Select at least one process for the ridges.")

# ---------------------- Single process profile ----------------------
st.subheader("Process Profile — Lot Size vs Metric")
pid = st.selectbox("Choose a process to inspect", res["Process_ID"])
dfp = df_f[(df_f["Metric_Type"] == metric) & (df_f["Process_ID"] == pid)].dropna(subset=["Lot_Size_Units","Metric_Value"]).sort_values("Lot_Size_Units")
if not dfp.empty:
    figp, axp = plt.subplots()
    axp.plot(dfp["Lot_Size_Units"], dfp["Metric_Value"], marker="o")
    axp.set_xlabel("Lot Size")
    axp.set_ylabel(metric)
    axp.set_title(f"{dfp['Process_Name'].iloc[0]} — Metric vs Lot Size")
    st.pyplot(figp)

# ---------------------- Tables ----------------------
st.subheader("Per-Process Summary")
st.dataframe(res.sort_values(sort_key, ascending=False))
