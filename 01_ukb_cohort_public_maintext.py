
# -*- coding: utf-8 -*-

"""UK Biobank cohort analysis pipeline for incident bladder cancer.

Public release aligned with the analyses reported in the main manuscript.
Default paths assume a simple repository layout with data/ and results/
folders at the project root.

Example
-------
python 01_ukb_cohort_public_main.py
python 01_ukb_cohort_public_main.py --raw-zip ./data/ukb_moduleA_backup_raw.zip --result-root ./results/ukb_cohort --olink ./data/ukb_olink_merged.csv
"""

import os
import argparse
import re
import ast
import json
import zipfile
import warnings
from datetime import datetime
from typing import List, Dict, Callable, Tuple, Optional

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore")
warnings.simplefilter("ignore", ConvergenceWarning)

# Optional dependency for spline modelling
try:
    from patsy import dmatrix
    PATSY_AVAILABLE = True
except Exception:
    PATSY_AVAILABLE = False

# Optional dependency for regression modules
try:
    import statsmodels.api as sm
    STATS_AVAILABLE = True
except Exception:
    STATS_AVAILABLE = False


# =========================================================
# 0. Paths and runtime configuration
# =========================================================
DEFAULT_RAW_ZIP_PATH = os.environ.get("UKB_RAW_ZIP_PATH", os.path.join("data", "ukb_moduleA_backup_raw.zip"))
DEFAULT_RESULT_ROOT = os.environ.get("UKB_RESULT_ROOT", os.path.join("results", "ukb_cohort"))

RAW_ZIP_PATH = DEFAULT_RAW_ZIP_PATH
RESULT_ROOT = DEFAULT_RESULT_ROOT

RUN_ID = datetime.now().strftime("RUN_%Y%m%d_%H%M%S")
RUN_DIR = ""
DIR_MAIN = ""
DIR_SUPPORT = ""
DIR_DOSE = ""
DIR_DIAG = ""
DIR_OLINK = ""
DIR_LOG = ""
DIR_RCS = ""

OLINK_MERGED_CANDIDATES = [
    os.path.join("data", "ukb_olink_merged.csv"),
    os.path.join("data", "ukb_olink_merged.xlsx"),
    os.path.join("data", "olink_merged.csv"),
    os.path.join("data", "olink_merged.xlsx"),
]


def initialize_run_directories(result_root: str) -> None:
    global RESULT_ROOT, RUN_ID, RUN_DIR
    global DIR_MAIN, DIR_SUPPORT, DIR_DOSE, DIR_DIAG, DIR_OLINK, DIR_LOG, DIR_RCS

    RESULT_ROOT = result_root
    RUN_ID = datetime.now().strftime("RUN_%Y%m%d_%H%M%S")
    RUN_DIR = os.path.join(RESULT_ROOT, RUN_ID)
    DIR_MAIN = os.path.join(RUN_DIR, "01_main")
    DIR_SUPPORT = os.path.join(RUN_DIR, "02_support")
    DIR_DOSE = os.path.join(RUN_DIR, "03_dose_response")
    DIR_DIAG = os.path.join(RUN_DIR, "04_diagnostics")
    DIR_OLINK = os.path.join(RUN_DIR, "05_olink")
    DIR_LOG = os.path.join(RUN_DIR, "06_logs")
    DIR_RCS = os.path.join(RUN_DIR, "07_rcs_threshold")
    for path in [RUN_DIR, DIR_MAIN, DIR_SUPPORT, DIR_DOSE, DIR_DIAG, DIR_OLINK, DIR_LOG, DIR_RCS]:
        os.makedirs(path, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="UK Biobank bladder cancer cohort pipeline")
    parser.add_argument("--raw-zip", default=DEFAULT_RAW_ZIP_PATH, help="Path to the UK Biobank source zip file containing a single CSV")
    parser.add_argument("--result-root", default=DEFAULT_RESULT_ROOT, help="Directory used to create the timestamped output run folder")
    parser.add_argument("--olink", nargs="*", default=None, help="Optional candidate paths for the merged Olink table")
    return parser.parse_args()

# =========================================================
# 1. Core settings
# =========================================================
DEFAULT_CANCER_CENSOR_DATE = pd.Timestamp("2023-05-31")
MIN_ROWS = 200
MIN_EVENTS = 20
MIN_OLINK_ROWS = 100
MIN_OLINK_EVENTS = 10
PRIMARY_EXPOSURES = {
    "pm25_2010": "participant.p24006",
    "no2_2010": "participant.p24003",
    "nox_2010": "participant.p24004",
    "pm25_absorb_2010": "participant.p24007",
}

SECONDARY_EXPOSURES = {
    "pm10_2010": "participant.p24005",
    "pmcoarse_2010": "participant.p24008",
    "traffic_near": "participant.p24009",
    "dist_road_inv": "participant.p24010",
    "traffic_major": "participant.p24011",
    "dist_major_inv": "participant.p24012",
}

ALL_EXPOSURES = {}
ALL_EXPOSURES.update(PRIMARY_EXPOSURES)
ALL_EXPOSURES.update(SECONDARY_EXPOSURES)

BASELINE_COLS = {
    "eid": "participant.eid",
    "baseline_date": "participant.p53_i0",
    "recruit_center": "participant.p54_i0",
    "address_years": "participant.p699_i0",
    "age": "participant.p21022",
    "sex": "participant.p31",
    "ethnicity": "participant.p21000_i0",
    "white_british": "participant.p22006",
    "education": "participant.p6138_i0",
    "smoking_status": "participant.p20116_i0",
    "alcohol_freq": "participant.p1558_i0",
    "bmi": "participant.p21001_i0",
    "urban_rural": "participant.p20118_i0",
}

DEATH_DATE_COLS = ["participant.p40000_i0", "participant.p40000_i1"]
CANCER_DATE_PREFIX = "participant.p40005_i"
CANCER_CODE_PREFIX = "participant.p40006_i"
CANCER_BEHAV_PREFIX = "participant.p40012_i"

SPECIAL_MISSING_VALUES = {-1, -3, -7, -10, -11, -13, -17, -21, -23, -27, -121, -313, -818}

MAIN_EXPOSURES = [
    "nox_2010",
    "pm25_2010",
    "traffic_combustion_score",
    "particle_score",
]

SUPPORT_EXPOSURES = [
    "no2_2010",
    "pm25_absorb_2010",
    "pm10_2010",
    "pmcoarse_2010",
]


SUBSET_FILTERS: Dict[str, Callable[[pd.DataFrame], pd.DataFrame]] = {
    "main": lambda d: d,
    "address5": lambda d: d.loc[d["address_years"].notna() & (d["address_years"] >= 5), :],
    "address5_lag5": lambda d: d.loc[
        (d["address_years"].notna()) &
        (d["address_years"] >= 5) &
        (~((d["status_bc"] == 1) & (d["fu_time_bc_years"] <= 5))), :
    ],
    "white_british": lambda d: d.loc[d["white_british"] == 1, :],
    "no_prev_malignancy": lambda d: d.loc[d["prevalent_any_malignancy"] == 0, :],
}

MODEL_CANDIDATES = {
    "Model1": ["age", "sex"],
    "Model2a": ["age", "sex", "smoking_status", "bmi"],
    "Model2b": ["age", "sex", "smoking_status", "bmi", "education", "alcohol_freq"],
    "Model2c": ["age", "sex", "smoking_status", "bmi", "education", "alcohol_freq", "urban_rural", "ethnicity"],
}

RCS_TARGET_EXPOSURES = [
    "nox_2010",
    "pm25_2010",
    "traffic_combustion_score",
    "particle_score",
]

RCS_TARGET_SUBSETS = [
    "address5",
    "address5_lag5",
]


PROTEIN_AXIS_RULES = {
    "endothelial_adhesion": ["SELE", "TIE1", "ICAM1", "VCAM1", "PECAM1", "ANGPT2", "FLT1", "KDR"],
    "inflammatory_innate": ["IL22RA2", "SERPINA4", "CXCL8", "CXCL9", "CXCL10", "IL6", "TNFRSF19", "TNFRSF1A", "TNFRSF1B"],
    "epithelial_remodeling": ["REG4", "CDHR5", "TMEM9", "MMP7", "EPCAM", "KRT19", "TFF3"],
    "metabolism_stress": ["HK2", "TFRC", "LEAP2", "SMOC1", "CA9", "GDF15", "HMOX1"],
    "coagulation_kallikrein": ["KLKB1", "F2", "F3", "SERPINF2", "PLAT", "PLAU"],
}

OLINK_TECHNICAL_KEYWORDS = [
    "plate", "batch", "panel", "qc", "sample", "visit", "aliquot", "site",
    "assay_warning", "uniprot", "olinkid", "lod", "limit_of_detection"
]

# =========================================================
# 2. Utilities
# =========================================================
def log(msg: str) -> None:
    print(msg, flush=True)
    with open(os.path.join(DIR_LOG, "analysis_log.txt"), "a", encoding="utf-8") as f:
        f.write(msg + "\n")

def unzip_read_csv(zip_path: str) -> pd.DataFrame:
    """Read the single CSV stored in a zip archive."""
    with zipfile.ZipFile(zip_path, "r") as zf:
        csv_names = [x for x in zf.namelist() if x.lower().endswith(".csv")]
        if len(csv_names) != 1:
            raise ValueError(f"Expected exactly one CSV file inside the zip archive, found {len(csv_names)}")
        csv_name = csv_names[0]
        log(f"[INFO] Reading CSV from zip archive: {csv_name}")
        with zf.open(csv_name) as f:
            df = pd.read_csv(f, low_memory=False)
    return df

def find_cols_by_prefix(cols: List[str], prefix: str) -> List[str]:
    """Find columns by prefix and sort by instance index."""
    out = [c for c in cols if c.startswith(prefix)]

    def idx(x):
        m = re.search(r"_i(\d+)", x)
        return int(m.group(1)) if m else 9999

    return sorted(out, key=idx)

def map_sex(x):
    """Map UK Biobank sex codes."""
    if pd.isna(x):
        return np.nan
    try:
        x = int(float(x))
    except Exception:
        return np.nan
    return {0: "Female", 1: "Male"}.get(x, np.nan)

def map_smoking(x):
    """Map UK Biobank smoking codes."""
    if pd.isna(x):
        return np.nan
    try:
        x = int(float(x))
    except Exception:
        return np.nan
    return {0: "Never", 1: "Former", 2: "Current"}.get(x, np.nan)

def map_white_british(x):
    """Convert White British indicator to 0/1."""
    if pd.isna(x):
        return np.nan
    try:
        x = int(float(x))
    except Exception:
        return np.nan
    return 1 if x == 1 else 0

def ukb_to_numeric_clean(series: pd.Series) -> pd.Series:
    """Clean standard numeric UKB fields."""
    s = pd.to_numeric(series, errors="coerce")
    s = s.mask(s.isin(SPECIAL_MISSING_VALUES), np.nan)
    return s

def parse_education_high(x):
    """
    Parse participant.p6138_i0.

    Example raw values include strings such as ``[1,6]``, ``[1,2,3,6]``,
    ``[-7]`` or ``NaN``.

    Simplified coding:
    1 = contains code 1 (college or university)
    0 = valid response without code 1
    NaN = missing or only negative special codes
    """
    if pd.isna(x):
        return np.nan

    s = str(x).strip()
    if s == "":
        return np.nan

    try:
        vals = ast.literal_eval(s)
    except Exception:
        return np.nan

    if not isinstance(vals, (list, tuple)):
        return np.nan

    vals = [v for v in vals if isinstance(v, (int, float))]
    if len(vals) == 0:
        return np.nan

    if all(v < 0 for v in vals):
        return np.nan

    return 1.0 if 1 in vals else 0.0

def zscore(series: pd.Series) -> pd.Series:
    """Standardize a numeric series."""
    s = pd.to_numeric(series, errors="coerce")
    mu = s.mean(skipna=True)
    sd = s.std(skipna=True)
    if pd.isna(sd) or sd == 0:
        return pd.Series(np.nan, index=s.index)
    return (s - mu) / sd

def add_scaled_exposure(df: pd.DataFrame, var: str) -> Tuple[pd.DataFrame, float, float]:
    """Add per-SD and per-IQR scaled exposure variables."""
    x = pd.to_numeric(df[var], errors="coerce")
    sd = float(x.std(skipna=True))
    iqr = float(x.quantile(0.75) - x.quantile(0.25))
    df[f"{var}_per_sd"] = x / sd if sd > 0 else np.nan
    df[f"{var}_per_iqr"] = x / iqr if iqr > 0 else np.nan
    return df, sd, iqr

def add_quartile_and_trend(df: pd.DataFrame, var: str) -> pd.DataFrame:
    """Add quartile groups and a quartile trend variable."""
    qvar = f"{var}_q4"
    tvar = f"{var}_q4_trend"

    valid = df[var].notna()
    if valid.sum() == 0:
        df[qvar] = np.nan
        df[tvar] = np.nan
        return df

    df.loc[valid, qvar] = pd.qcut(
        df.loc[valid, var],
        4,
        labels=["Q1", "Q2", "Q3", "Q4"],
        duplicates="drop"
    )
    df[qvar] = df[qvar].astype("category")

    trend_map = {"Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4}
    df[tvar] = df[qvar].astype(str).map(trend_map)
    df.loc[df[qvar].isna(), tvar] = np.nan

    return df

def get_cat_cols(covs: List[str]) -> List[str]:
    """Return covariates that should be treated as categorical."""
    return [c for c in ["sex", "smoking_status", "alcohol_freq", "urban_rural", "ethnicity"] if c in covs]

def build_model_df(df: pd.DataFrame, exposure_var: str, covs: List[str], cat_cols: List[str]) -> pd.DataFrame:
    """
    Build the Cox model matrix.

    Steps: keep required columns, remove incomplete rows, one-hot encode
    categorical covariates, and drop constant or near-zero variance columns.
    """
    cols = ["fu_time_bc_years", "status_bc", exposure_var] + covs
    cols = [c for c in cols if c in df.columns]

    tmp = df.loc[:, cols].copy()
    tmp = tmp.replace([np.inf, -np.inf], np.nan).dropna()
    if tmp.empty:
        return pd.DataFrame()

    for c in tmp.columns:
        if c not in cat_cols:
            tmp[c] = pd.to_numeric(tmp[c], errors="coerce")

    tmp = tmp.replace([np.inf, -np.inf], np.nan).dropna()
    if tmp.empty:
        return pd.DataFrame()

    use_cat = [c for c in cat_cols if c in tmp.columns]
    if use_cat:
        tmp = pd.get_dummies(tmp, columns=use_cat, drop_first=True)

    for c in tmp.columns:
        tmp[c] = pd.to_numeric(tmp[c], errors="coerce")

    tmp = tmp.replace([np.inf, -np.inf], np.nan).dropna()
    if tmp.empty:
        return pd.DataFrame()

    protected = {"fu_time_bc_years", "status_bc"}
    drop_cols = []
    for c in tmp.columns:
        if c in protected:
            continue
        if tmp[c].nunique(dropna=True) <= 1:
            drop_cols.append(c)
            continue
        v = tmp[c].var()
        if pd.isna(v) or v < 1e-12:
            drop_cols.append(c)

    if drop_cols:
        tmp = tmp.drop(columns=drop_cols, errors="ignore")

    if exposure_var not in tmp.columns:
        return pd.DataFrame()
    if tmp.shape[1] < 3:
        return pd.DataFrame()

    return tmp

def build_quartile_model_df(df: pd.DataFrame, qvar: str, covs: List[str], cat_cols: List[str]) -> pd.DataFrame:
    """
    Build the Cox model matrix for quartile-based analyses.

    The quartile exposure variable is dummy-encoded together with the other
    categorical covariates.
    """
    cols = ["fu_time_bc_years", "status_bc", qvar] + covs
    cols = [c for c in cols if c in df.columns]

    tmp = df.loc[:, cols].copy()
    tmp = tmp.replace([np.inf, -np.inf], np.nan).dropna()
    if tmp.empty:
        return pd.DataFrame()

    dummy_cols = [qvar] + [c for c in cat_cols if c in tmp.columns]
    tmp = pd.get_dummies(tmp, columns=dummy_cols, drop_first=True)

    for c in tmp.columns:
        tmp[c] = pd.to_numeric(tmp[c], errors="coerce")

    tmp = tmp.replace([np.inf, -np.inf], np.nan).dropna()
    if tmp.empty:
        return pd.DataFrame()

    protected = {"fu_time_bc_years", "status_bc"}
    drop_cols = []
    for c in tmp.columns:
        if c in protected:
            continue
        if tmp[c].nunique(dropna=True) <= 1:
            drop_cols.append(c)
            continue
        v = tmp[c].var()
        if pd.isna(v) or v < 1e-12:
            drop_cols.append(c)

    if drop_cols:
        tmp = tmp.drop(columns=drop_cols, errors="ignore")

    if tmp.shape[1] < 3:
        return pd.DataFrame()

    return tmp

def fit_cox(df_model: pd.DataFrame) -> CoxPHFitter:
    """
    Fit a Cox model and add a minimal penalizer only if needed for convergence.
    """
    if df_model is None or df_model.empty:
        raise ValueError("fit_cox received empty dataframe")

    try:
        cph = CoxPHFitter()
        cph.fit(df_model, duration_col="fu_time_bc_years", event_col="status_bc")
        return cph
    except Exception:
        pass

    try:
        cph = CoxPHFitter(penalizer=1e-6)
        cph.fit(df_model, duration_col="fu_time_bc_years", event_col="status_bc")
        return cph
    except Exception:
        pass

    cph = CoxPHFitter(penalizer=1e-4)
    cph.fit(df_model, duration_col="fu_time_bc_years", event_col="status_bc")
    return cph

def extract_term(cph: CoxPHFitter, term: str, extra: dict) -> dict:
    """Extract a single term from a fitted Cox model."""
    s = cph.summary.loc[term]
    out = dict(extra)
    out.update({
        "term": term,
        "coef": s["coef"],
        "HR": np.exp(s["coef"]),
        "LCL": np.exp(s["coef lower 95%"]),
        "UCL": np.exp(s["coef upper 95%"]),
        "p": s["p"]
    })
    return out

def safe_read_table(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        return pd.read_csv(path, low_memory=False)
    if ext in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    if ext == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported file type: {path}")

def first_existing_path(candidates: List[str]) -> Optional[str]:
    for p in candidates:
        if os.path.exists(p):
            return p
    return None

def normalize_token(x: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "", str(x).upper())

def ensure_datetime(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df

def build_complete_case_n_events(df: pd.DataFrame) -> Tuple[int, int]:
    if df.empty:
        return 0, 0
    n = df.shape[0]
    e = int(pd.to_numeric(df["status_bc"], errors="coerce").fillna(0).sum()) if "status_bc" in df.columns else 0
    return n, e


def choose_best_technical_covariates(df: pd.DataFrame, max_n: int = 6) -> List[str]:
    out = []
    for c in df.columns:
        lc = c.lower()
        if any(k in lc for k in OLINK_TECHNICAL_KEYWORDS):
            out.append(c)
    return out[:max_n]

def normalize_gene_name(x: str) -> str:
    x = str(x).upper().strip()
    x = re.sub(r"\s+", "", x)
    x = x.replace("-", "")
    return x

def match_protein_column(col: str, gene: str) -> bool:
    col_norm = normalize_gene_name(col)
    gene_norm = normalize_gene_name(gene)
    if col_norm == gene_norm:
        return True
    if col_norm.startswith(gene_norm + "NPX") or col_norm.endswith(gene_norm):
        return True
    if re.search(rf"(^|[^A-Z0-9]){re.escape(gene_norm)}([^A-Z0-9]|$)", re.sub(r"[^A-Z0-9]+", " ", str(col).upper())):
        return True
    return False

def find_protein_columns(df: pd.DataFrame) -> Dict[str, str]:
    """
    Identify candidate Olink protein columns by starting from numeric fields
    and excluding known UKB exposures, covariates, and outcomes.
    """
    exclude = set(
        ["eid", "status_bc", "fu_time_bc_years", "baseline_date", "bc_date", "death_date",
         "age", "sex", "smoking_status", "bmi", "education", "alcohol_freq", "urban_rural", "ethnicity",
         "white_british", "address_years", "recruit_center", "prevalent_bca", "prevalent_any_malignancy",
         "traffic_combustion_score", "particle_score"] + list(ALL_EXPOSURES.keys())
    )
    out = {}
    for c in df.columns:
        if c in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            if c.lower().startswith("participant."):
                continue
            out[c] = c
    return out

def build_linear_df(df: pd.DataFrame, y: str, xvars: List[str], cat_cols: List[str]) -> pd.DataFrame:
    cols = [y] + xvars
    cols = [c for c in cols if c in df.columns]
    tmp = df.loc[:, cols].copy().replace([np.inf, -np.inf], np.nan).dropna()
    if tmp.empty:
        return pd.DataFrame()

    use_cat = [c for c in cat_cols if c in tmp.columns]
    if use_cat:
        tmp = pd.get_dummies(tmp, columns=use_cat, drop_first=True)

    for c in tmp.columns:
        tmp[c] = pd.to_numeric(tmp[c], errors="coerce")
    tmp = tmp.replace([np.inf, -np.inf], np.nan).dropna()

    drop_cols = []
    protected = {y}
    for c in tmp.columns:
        if c in protected:
            continue
        if tmp[c].nunique(dropna=True) <= 1:
            drop_cols.append(c)
            continue
        v = tmp[c].var()
        if pd.isna(v) or v < 1e-12:
            drop_cols.append(c)
    if drop_cols:
        tmp = tmp.drop(columns=drop_cols, errors="ignore")
    return tmp

def fit_ols_hc3(df: pd.DataFrame, y: str):
    if (not STATS_AVAILABLE) or df.empty or y not in df.columns:
        return None
    x = df.drop(columns=[y])
    x = sm.add_constant(x, has_constant="add")
    model = sm.OLS(df[y], x).fit(cov_type="HC3")
    return model


# =========================================================
# 3. Build master analysis table
# =========================================================
def build_master_analysis(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Build the incident bladder cancer analysis dataset."""
    log("[INFO] Selecting analysis columns")

    raw_cols = list(df_raw.columns)
    cancer_date_cols = find_cols_by_prefix(raw_cols, CANCER_DATE_PREFIX)
    cancer_code_cols = find_cols_by_prefix(raw_cols, CANCER_CODE_PREFIX)
    cancer_beh_cols = find_cols_by_prefix(raw_cols, CANCER_BEHAV_PREFIX)

    needed_cols = []
    needed_cols += [v for v in BASELINE_COLS.values() if v in raw_cols]
    needed_cols += [v for v in ALL_EXPOSURES.values() if v in raw_cols]
    needed_cols += [c for c in DEATH_DATE_COLS if c in raw_cols]
    needed_cols += cancer_date_cols + cancer_code_cols + cancer_beh_cols
    needed_cols = list(dict.fromkeys(needed_cols))

    df_raw = df_raw.loc[:, needed_cols].copy()
    log(f"[INFO] Reduced raw dataset shape: {df_raw.shape}")

    rename_map = {v: k for k, v in BASELINE_COLS.items() if v in df_raw.columns}
    for new_name, old_name in ALL_EXPOSURES.items():
        if old_name in df_raw.columns:
            rename_map[old_name] = new_name

    df = df_raw.rename(columns=rename_map).copy()

    numeric_cols = [
        "eid", "recruit_center", "address_years", "age", "sex",
        "ethnicity", "white_british", "smoking_status",
        "alcohol_freq", "bmi", "urban_rural"
    ] + list(ALL_EXPOSURES.keys())

    for c in numeric_cols:
        if c in df.columns:
            df[c] = ukb_to_numeric_clean(df[c])

    if "education" in df.columns:
        df["education"] = df["education"].apply(parse_education_high)

    if "baseline_date" in df.columns:
        df["baseline_date"] = pd.to_datetime(df["baseline_date"], errors="coerce")
    for c in DEATH_DATE_COLS:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    for c in cancer_date_cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")

    log("[INFO] Constructing cancer outcomes")

    bc_dates = []
    any_malig_dates = []
    death_dates = []

    for idx in range(len(df)):
        row = df.iloc[idx]

        bc_vals = []
        for dcol, ccol in zip(cancer_date_cols, cancer_code_cols):
            code = row.get(ccol, np.nan)
            if pd.isna(code):
                continue
            if str(code).strip().upper().startswith("C67"):
                dt = row.get(dcol, pd.NaT)
                if pd.notna(dt):
                    bc_vals.append(dt)
        bc_dates.append(min(bc_vals) if bc_vals else pd.NaT)

        mal_vals = []
        for dcol, ccol, bcol in zip(cancer_date_cols, cancer_code_cols, cancer_beh_cols):
            beh = row.get(bcol, np.nan)
            code = row.get(ccol, np.nan)
            if pd.isna(beh) or pd.isna(code):
                continue
            if str(beh).strip() == "3":
                dt = row.get(dcol, pd.NaT)
                if pd.notna(dt):
                    mal_vals.append(dt)
        any_malig_dates.append(min(mal_vals) if mal_vals else pd.NaT)

        dvals = []
        for c in DEATH_DATE_COLS:
            if c in df.columns:
                dt = row.get(c, pd.NaT)
                if pd.notna(dt):
                    dvals.append(dt)
        death_dates.append(min(dvals) if dvals else pd.NaT)

    df["bc_date"] = bc_dates
    df["any_malignant_cancer_date"] = any_malig_dates
    df["death_date"] = death_dates

    df["prevalent_bca"] = np.where(
        pd.notna(df["bc_date"]) &
        pd.notna(df["baseline_date"]) &
        (df["bc_date"] < df["baseline_date"]),
        1, 0
    )

    df["prevalent_any_malignancy"] = np.where(
        pd.notna(df["any_malignant_cancer_date"]) &
        pd.notna(df["baseline_date"]) &
        (df["any_malignant_cancer_date"] < df["baseline_date"]),
        1, 0
    )

    if "sex" in df.columns:
        df["sex"] = df["sex"].map(map_sex).astype("category")
    if "smoking_status" in df.columns:
        df["smoking_status"] = df["smoking_status"].map(map_smoking).astype("category")
    if "white_british" in df.columns:
        df["white_british"] = df["white_british"].map(map_white_british)

    for c in ["ethnicity", "alcohol_freq", "urban_rural"]:
        if c in df.columns:
            df[c] = df[c].astype("category")

    if "education" in df.columns:
        df["education"] = pd.to_numeric(df["education"], errors="coerce")

    # Composite exposure scores based on the original mean z-score definition.
    traffic_components = []
    if "no2_2010" in df.columns:
        traffic_components.append(zscore(df["no2_2010"]))
    if "nox_2010" in df.columns:
        traffic_components.append(zscore(df["nox_2010"]))
    if "pm25_absorb_2010" in df.columns:
        traffic_components.append(zscore(df["pm25_absorb_2010"]))
    if traffic_components:
        df["traffic_combustion_score"] = pd.concat(traffic_components, axis=1).mean(axis=1, skipna=True)

    particle_components = []
    if "pm25_2010" in df.columns:
        particle_components.append(zscore(df["pm25_2010"]))
    if "pm10_2010" in df.columns:
        particle_components.append(zscore(df["pm10_2010"]))
    if "pmcoarse_2010" in df.columns:
        particle_components.append(zscore(df["pmcoarse_2010"]))
    if particle_components:
        df["particle_score"] = pd.concat(particle_components, axis=1).mean(axis=1, skipna=True)


    # Follow-up definition.
    df["cancer_censor_date"] = DEFAULT_CANCER_CENSOR_DATE
    end_dates = []
    status_bc = []

    for _, row in df[["baseline_date", "bc_date", "death_date", "cancer_censor_date"]].iterrows():
        baseline_date = row["baseline_date"]
        if pd.isna(baseline_date):
            end_dates.append(pd.NaT)
            status_bc.append(np.nan)
            continue

        candidates = [x for x in [row["bc_date"], row["death_date"], row["cancer_censor_date"]] if pd.notna(x)]
        if not candidates:
            end_dates.append(pd.NaT)
            status_bc.append(np.nan)
            continue

        end_date = min(candidates)
        end_dates.append(end_date)

        bc_event = int(
            pd.notna(row["bc_date"]) and
            end_date == row["bc_date"] and
            row["bc_date"] >= baseline_date
        )
        status_bc.append(bc_event)

    df["end_of_followup_bc"] = end_dates
    df["status_bc"] = status_bc
    df["fu_time_bc_years"] = (df["end_of_followup_bc"] - df["baseline_date"]).dt.days / 365.25

    keep = (
        df["baseline_date"].notna() &
        (df["prevalent_bca"] == 0) &
        df["fu_time_bc_years"].notna() &
        (df["fu_time_bc_years"] > 0)
    )

    df_main = df.loc[keep, :].copy()
    df_main.to_csv(os.path.join(DIR_LOG, "master_analysis.csv"), index=False)
    return df_main

# =========================================================
# 4. Main association models
# =========================================================
def apply_subset_filter(df: pd.DataFrame, subset_name: str) -> pd.DataFrame:
    return SUBSET_FILTERS[subset_name](df)

def diagnose_missingness(df: pd.DataFrame, exposure: str, covs: List[str], subset: str, model_name: str) -> pd.DataFrame:
    """Summarize missingness and complete-case counts."""
    rows = []
    needed = [c for c in [exposure] + covs if c in df.columns]
    n0 = df.shape[0]
    e0 = int(df["status_bc"].sum())

    rows.append({
        "subset": subset,
        "model": model_name,
        "stage": "initial",
        "variable": "__ALL__",
        "n": n0,
        "events": e0,
        "missing_n": np.nan,
        "missing_pct": np.nan
    })

    for c in needed:
        miss = int(df[c].isna().sum())
        rows.append({
            "subset": subset,
            "model": model_name,
            "stage": "variable_missing",
            "variable": c,
            "n": n0,
            "events": e0,
            "missing_n": miss,
            "missing_pct": miss / n0 if n0 > 0 else np.nan
        })

    cc_cols = ["fu_time_bc_years", "status_bc"] + needed
    cc = df.loc[:, [c for c in cc_cols if c in df.columns]].replace([np.inf, -np.inf], np.nan).dropna()

    rows.append({
        "subset": subset,
        "model": model_name,
        "stage": "complete_case",
        "variable": "__ALL__",
        "n": cc.shape[0],
        "events": int(cc["status_bc"].sum()) if "status_bc" in cc.columns else np.nan,
        "missing_n": np.nan,
        "missing_pct": np.nan
    })

    return pd.DataFrame(rows)

def run_main_result(df: pd.DataFrame, exposure: str, subset_name: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run the main per-SD Cox models."""
    out_rows = []
    diag_rows = []

    dset = apply_subset_filter(df, subset_name)
    if dset.empty or exposure not in dset.columns:
        return pd.DataFrame(), pd.DataFrame()

    dset = dset.copy()
    dset, sd_val, iqr_val = add_scaled_exposure(dset, exposure)
    exposure_var = f"{exposure}_per_sd"

    for model_name, cov_template in MODEL_CANDIDATES.items():
        covs = [c for c in cov_template if c in dset.columns]
        diag_rows.append(diagnose_missingness(dset, exposure_var, covs, subset_name, model_name))

        model_df = build_model_df(dset, exposure_var, covs, get_cat_cols(covs))
        if model_df.empty:
            continue
        if model_df.shape[0] < MIN_ROWS or model_df["status_bc"].sum() < MIN_EVENTS:
            continue

        try:
            cph = fit_cox(model_df)
        except Exception as e:
            log(f"[WARN] Cox model failed | subset={subset_name} | exposure={exposure} | model={model_name} | {e}")
            continue

        row = extract_term(
            cph, exposure_var,
            {
                "subset": subset_name,
                "model": model_name,
                "exposure": exposure,
                "scale": "per_sd",
                "n": model_df.shape[0],
                "events": int(model_df["status_bc"].sum()),
                "raw_sd": sd_val,
                "raw_iqr": iqr_val,
                "covs_used": ";".join(covs)
            }
        )
        out_rows.append(row)

    return (
        pd.DataFrame(out_rows),
        pd.concat(diag_rows, ignore_index=True) if diag_rows else pd.DataFrame()
    )

# =========================================================
# 5. Dose-response analyses
# =========================================================
def run_dose_response(df: pd.DataFrame, exposure: str, subset_name: str) -> pd.DataFrame:
    """
    Dose-response analyses based on Model2c:
    1. per SD
    2. per IQR
    3. quartiles
    4. quartile trend
    """
    if exposure not in df.columns:
        return pd.DataFrame()

    dset = apply_subset_filter(df, subset_name)
    if dset.empty:
        return pd.DataFrame()

    dset = dset.copy()
    dset, sd_val, iqr_val = add_scaled_exposure(dset, exposure)
    dset = add_quartile_and_trend(dset, exposure)

    covs = [c for c in MODEL_CANDIDATES["Model2c"] if c in dset.columns]
    cat_cols = get_cat_cols(covs)

    rows = []

    # 1. per SD
    model_df_sd = build_model_df(dset, f"{exposure}_per_sd", covs, cat_cols)
    if not model_df_sd.empty and model_df_sd.shape[0] >= MIN_ROWS and model_df_sd["status_bc"].sum() >= MIN_EVENTS:
        try:
            cph = fit_cox(model_df_sd)
            rows.append(extract_term(
                cph, f"{exposure}_per_sd",
                {
                    "subset": subset_name,
                    "exposure": exposure,
                    "analysis": "per_sd",
                    "model": "Model2c",
                    "n": model_df_sd.shape[0],
                    "events": int(model_df_sd["status_bc"].sum()),
                    "raw_sd": sd_val,
                    "raw_iqr": iqr_val
                }
            ))
        except Exception as e:
            log(f"[WARN] Dose-response per-SD model failed | subset={subset_name} | exposure={exposure} | {e}")

    # 2. per IQR
    model_df_iqr = build_model_df(dset, f"{exposure}_per_iqr", covs, cat_cols)
    if not model_df_iqr.empty and model_df_iqr.shape[0] >= MIN_ROWS and model_df_iqr["status_bc"].sum() >= MIN_EVENTS:
        try:
            cph = fit_cox(model_df_iqr)
            rows.append(extract_term(
                cph, f"{exposure}_per_iqr",
                {
                    "subset": subset_name,
                    "exposure": exposure,
                    "analysis": "per_iqr",
                    "model": "Model2c",
                    "n": model_df_iqr.shape[0],
                    "events": int(model_df_iqr["status_bc"].sum()),
                    "raw_sd": sd_val,
                    "raw_iqr": iqr_val
                }
            ))
        except Exception as e:
            log(f"[WARN] Dose-response per-IQR model failed | subset={subset_name} | exposure={exposure} | {e}")

    # 3. quartile
    qvar = f"{exposure}_q4"
    if qvar in dset.columns:
        model_df_q = build_quartile_model_df(dset, qvar, covs, cat_cols)
        if not model_df_q.empty and model_df_q.shape[0] >= MIN_ROWS and model_df_q["status_bc"].sum() >= MIN_EVENTS:
            try:
                cph = fit_cox(model_df_q)
                for term in cph.summary.index:
                    if term.startswith(f"{qvar}_"):
                        rows.append(extract_term(
                            cph, term,
                            {
                                "subset": subset_name,
                                "exposure": exposure,
                                "analysis": "quartile",
                                "model": "Model2c",
                                "n": model_df_q.shape[0],
                                "events": int(model_df_q["status_bc"].sum()),
                                "raw_sd": sd_val,
                                "raw_iqr": iqr_val
                            }
                        ))
            except Exception as e:
                log(f"[WARN] Dose-response quartile model failed | subset={subset_name} | exposure={exposure} | {e}")

    # 4. trend
    tvar = f"{exposure}_q4_trend"
    if tvar in dset.columns:
        model_df_trend = build_model_df(dset, tvar, covs, cat_cols)
        if not model_df_trend.empty and model_df_trend.shape[0] >= MIN_ROWS and model_df_trend["status_bc"].sum() >= MIN_EVENTS:
            try:
                cph = fit_cox(model_df_trend)
                rows.append(extract_term(
                    cph, tvar,
                    {
                        "subset": subset_name,
                        "exposure": exposure,
                        "analysis": "quartile_trend",
                        "model": "Model2c",
                        "n": model_df_trend.shape[0],
                        "events": int(model_df_trend["status_bc"].sum()),
                        "raw_sd": sd_val,
                        "raw_iqr": iqr_val
                    }
                ))
            except Exception as e:
                log(f"[WARN] Dose-response quartile-trend model failed | subset={subset_name} | exposure={exposure} | {e}")

    return pd.DataFrame(rows)

# =========================================================
# 6. Restricted cubic spline analysis
# =========================================================

# =========================================================
# 6. Restricted cubic spline analysis
# =========================================================
def get_rcs_spec(x: pd.Series) -> dict:
    """
    Derive a fixed knot specification from the observed exposure distribution.
    """
    if not PATSY_AVAILABLE:
        return {}

    s = pd.to_numeric(x, errors="coerce").replace([np.inf, -np.inf], np.nan)
    valid = s.dropna()

    if valid.shape[0] < 100:
        return {}
    if valid.nunique() < 6:
        return {}

    q = valid.quantile([0.05, 0.35, 0.65, 0.95])
    if q.isna().any():
        return {}

    lower_bound = float(valid.min())
    upper_bound = float(valid.max())
    inner_knots = [float(q.loc[0.35]), float(q.loc[0.65])]

    all_check = [lower_bound] + inner_knots + [upper_bound]
    if len(np.unique(all_check)) < 4:
        return {}

    return {
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "inner_knots": inner_knots
    }

def make_rcs_basis_from_spec(x: pd.Series, var_name: str, spec: dict) -> pd.DataFrame:
    """
    Generate spline basis terms for new data using a fixed specification.
    """
    if not PATSY_AVAILABLE or not spec:
        return pd.DataFrame(index=x.index)

    s = pd.to_numeric(x, errors="coerce").replace([np.inf, -np.inf], np.nan)
    valid = s.dropna()

    if valid.empty:
        return pd.DataFrame(index=x.index)

    basis_valid = dmatrix(
        "cr(x, knots=inner_knots, lower_bound=lower_bound, upper_bound=upper_bound) - 1",
        {
            "x": valid,
            "inner_knots": spec["inner_knots"],
            "lower_bound": spec["lower_bound"],
            "upper_bound": spec["upper_bound"]
        },
        return_type="dataframe"
    )

    basis_valid.columns = [f"{var_name}_rcs_{i+1}" for i in range(basis_valid.shape[1])]

    basis = pd.DataFrame(index=x.index, columns=basis_valid.columns, dtype=float)
    basis.loc[valid.index, :] = basis_valid.values
    return basis

def run_rcs_analysis(df: pd.DataFrame, exposure: str, subset_name: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Return two tables: spline-term coefficients and a prediction grid with
    relative hazard ratios referenced to the median exposure level.
    """
    if not PATSY_AVAILABLE:
        return pd.DataFrame(), pd.DataFrame()

    if exposure not in df.columns:
        return pd.DataFrame(), pd.DataFrame()

    dset = apply_subset_filter(df, subset_name)
    if dset.empty:
        return pd.DataFrame(), pd.DataFrame()

    covs = [c for c in MODEL_CANDIDATES["Model2c"] if c in dset.columns]
    cat_cols = get_cat_cols(covs)

    # Fix knot locations from the observed exposure distribution
    spec = get_rcs_spec(dset[exposure])
    if not spec:
        return pd.DataFrame(), pd.DataFrame()

    basis = make_rcs_basis_from_spec(dset[exposure], exposure, spec)
    if basis.empty or basis.shape[1] == 0:
        return pd.DataFrame(), pd.DataFrame()

    dtmp = pd.concat([dset.copy(), basis], axis=1)
    spline_terms = list(basis.columns)

    cols = ["fu_time_bc_years", "status_bc"] + spline_terms + covs
    cols = [c for c in cols if c in dtmp.columns]
    model_df = dtmp.loc[:, cols].copy()
    model_df = model_df.replace([np.inf, -np.inf], np.nan).dropna()
    if model_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    use_cat = [c for c in cat_cols if c in model_df.columns]
    if use_cat:
        model_df = pd.get_dummies(model_df, columns=use_cat, drop_first=True)

    for c in model_df.columns:
        model_df[c] = pd.to_numeric(model_df[c], errors="coerce")
    model_df = model_df.replace([np.inf, -np.inf], np.nan).dropna()

    protected = {"fu_time_bc_years", "status_bc"}
    drop_cols = []
    for c in model_df.columns:
        if c in protected:
            continue
        if model_df[c].nunique(dropna=True) <= 1:
            drop_cols.append(c)
            continue
        v = model_df[c].var()
        if pd.isna(v) or v < 1e-12:
            drop_cols.append(c)
    if drop_cols:
        model_df = model_df.drop(columns=drop_cols, errors="ignore")

    kept_terms = [x for x in spline_terms if x in model_df.columns]
    if len(kept_terms) == 0:
        return pd.DataFrame(), pd.DataFrame()

    if model_df.shape[0] < MIN_ROWS or model_df["status_bc"].sum() < MIN_EVENTS:
        return pd.DataFrame(), pd.DataFrame()

    try:
        cph = fit_cox(model_df)
    except Exception as e:
        log(f"[WARN] RCS model failed | subset={subset_name} | exposure={exposure} | {e}")
        return pd.DataFrame(), pd.DataFrame()

    # 1) coefficient table
    coef_rows = []
    for term in cph.summary.index:
        if term in kept_terms:
            coef_rows.append(extract_term(
                cph, term,
                {
                    "subset": subset_name,
                    "exposure": exposure,
                    "analysis": "rcs_basis",
                    "model": "Model2c",
                    "n": model_df.shape[0],
                    "events": int(model_df["status_bc"].sum())
                }
            ))
    coef_df = pd.DataFrame(coef_rows)

    # 2) prediction grid
    x = pd.to_numeric(dset[exposure], errors="coerce").dropna()
    if x.empty:
        return coef_df, pd.DataFrame()

    grid = np.linspace(float(x.quantile(0.05)), float(x.quantile(0.95)), 100)
    ref_x = float(x.median())

    grid_basis = make_rcs_basis_from_spec(pd.Series(grid), exposure, spec)
    ref_basis = make_rcs_basis_from_spec(pd.Series([ref_x]), exposure, spec)

    if grid_basis.empty or ref_basis.empty:
        return coef_df, pd.DataFrame()

    beta_terms = [t for t in kept_terms if t in cph.params_.index]
    beta = np.array([float(cph.params_.loc[t]) for t in beta_terms])

    grid_basis = grid_basis[beta_terms].copy()
    ref_basis = ref_basis[beta_terms].copy()

    grid_lp = np.dot(grid_basis.values, beta)
    ref_lp = np.dot(ref_basis.iloc[0].values, beta)

    pred_df = pd.DataFrame({
        "subset": subset_name,
        "exposure": exposure,
        "x": grid,
        "relative_HR_vs_median": np.exp(grid_lp - ref_lp)
    })

    return coef_df, pred_df

# =========================================================
# 7. Threshold scan
# =========================================================

# =========================================================
# 8. WQS preparation tables
# =========================================================

# =========================================================
# 11. Smoking interaction and stratified analyses
# =========================================================

# =========================================================
# 12. Olink merge and protein domains
# =========================================================
def load_olink_merged() -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    path = first_existing_path(OLINK_MERGED_CANDIDATES)
    if path is None:
        log("[INFO] No merged Olink table found; skipping Olink analyses")
        return None, None
    try:
        df = safe_read_table(path)
    except Exception as e:
        log(f"[WARN] Failed to read the merged Olink table: {e}")
        return None, None

    # Identify the participant identifier.
    eid_candidates = [c for c in df.columns if normalize_token(c) in ["EID", "PARTICIPANTEID", "PARTICIPANTEID0", "ID"]]
    if "eid" in df.columns:
        pass
    elif eid_candidates:
        df = df.rename(columns={eid_candidates[0]: "eid"})
    else:
        log("[WARN] No participant identifier was detected in the Olink table; skipping Olink analyses")
        return None, path

    df["eid"] = pd.to_numeric(df["eid"], errors="coerce")
    df = df.loc[df["eid"].notna(), :].copy()
    log(f"[INFO] Loaded merged Olink table: {path} | shape={df.shape}")
    return df, path

def prepare_olink_merged(df_olink: pd.DataFrame, df_main: pd.DataFrame) -> pd.DataFrame:
    base_cols = [
        "eid", "status_bc", "fu_time_bc_years",
        "age", "sex", "smoking_status", "bmi", "education", "alcohol_freq", "urban_rural", "ethnicity",
        "traffic_combustion_score", "particle_score",
        "nox_2010", "pm25_2010", "no2_2010", "pm25_absorb_2010", "pm10_2010", "pmcoarse_2010"
    ]
    bridge = df_main[[c for c in base_cols if c in df_main.columns]].drop_duplicates("eid")
    out = df_olink.merge(bridge, on="eid", how="left", suffixes=("", "_bridge"))
    return out

def build_protein_axis_definitions(df_olink: pd.DataFrame) -> pd.DataFrame:
    protein_cols = find_protein_columns(df_olink)
    all_cols = list(protein_cols.keys())

    rows = []
    for axis_name, genes in PROTEIN_AXIS_RULES.items():
        matched_cols = []
        matched_genes = []
        for g in genes:
            hits = [c for c in all_cols if match_protein_column(c, g)]
            if len(hits) > 0:
                matched_cols.extend(hits)
                matched_genes.extend([g] * len(hits))
        matched_cols = list(dict.fromkeys(matched_cols))
        for c in matched_cols:
            rows.append({
                "axis_name": axis_name,
                "protein_col": c,
                "mapped_gene": next((g for g in genes if match_protein_column(c, g)), np.nan)
            })
    return pd.DataFrame(rows)

def construct_protein_axes(df_olink: pd.DataFrame, axis_def: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    out = df_olink.copy()
    summary_rows = []

    if axis_def.empty:
        return out, pd.DataFrame()

    for axis_name in sorted(axis_def["axis_name"].dropna().unique()):
        cols = axis_def.loc[axis_def["axis_name"] == axis_name, "protein_col"].dropna().tolist()
        cols = [c for c in cols if c in out.columns]
        if len(cols) == 0:
            continue

        tmp = out[cols].apply(pd.to_numeric, errors="coerce")
        valid_cols = [c for c in cols if tmp[c].notna().sum() >= max(50, int(0.2 * len(tmp)))]
        if len(valid_cols) == 0:
            continue

        zmat = tmp[valid_cols].apply(zscore, axis=0)
        out[f"{axis_name}_mean_z"] = zmat.mean(axis=1, skipna=True)

        axis_score_col = f"{axis_name}_mean_z"
        method = "mean_z"

        out[f"{axis_name}_score"] = out[axis_score_col]
        summary_rows.append({
            "axis_name": axis_name,
            "n_proteins_total": len(cols),
            "n_proteins_used": len(valid_cols),
            "proteins_used": ";".join(valid_cols),
            "score_source": axis_score_col,
            "construction_method": method,
            "n_nonmissing_score": int(out[f"{axis_name}_score"].notna().sum())
        })

    return out, pd.DataFrame(summary_rows)

def run_olink_axis_association(df_olink: pd.DataFrame, exposure_vars: List[str]) -> pd.DataFrame:
    if not STATS_AVAILABLE:
        log("[INFO] statsmodels is not available; skipping Olink regression modules")
        return pd.DataFrame()

    axis_vars = [c for c in df_olink.columns if c.endswith("_score")]
    axis_vars = [c for c in axis_vars if c not in ["traffic_combustion_score", "particle_score", "traffic_combustion_score_pca", "particle_score_pca"]]
    if len(axis_vars) == 0:
        return pd.DataFrame()

    tech_covs = choose_best_technical_covariates(df_olink)
    base_covs = [c for c in ["age", "sex", "smoking_status", "bmi", "education", "alcohol_freq", "urban_rural", "ethnicity"] if c in df_olink.columns]
    covs = base_covs + tech_covs
    rows = []

    for axis_var in axis_vars:
        for exp in exposure_vars:
            if exp not in df_olink.columns:
                continue
            work = df_olink.copy()
            work[exp] = zscore(work[exp])
            model_df = build_linear_df(work, axis_var, [exp] + covs, get_cat_cols(covs))
            if model_df.empty or model_df.shape[0] < MIN_OLINK_ROWS:
                continue
            fit = fit_ols_hc3(model_df, axis_var)
            if fit is None or exp not in fit.params.index:
                continue
            rows.append({
                "protein_axis": axis_var,
                "exposure": exp,
                "beta": fit.params[exp],
                "se": fit.bse[exp],
                "t": fit.tvalues[exp],
                "p": fit.pvalues[exp],
                "n": model_df.shape[0],
                "covs_used": ";".join(covs),
            })
    out = pd.DataFrame(rows)
    if not out.empty:
        out["fdr_p"] = out.groupby("exposure")["p"].transform(lambda s: pd.Series(np.minimum(1, np.array(_bh_fdr(s.tolist()))), index=s.index))
    return out

def _bh_fdr(pvals: List[float]) -> List[float]:
    pvals = np.array(pvals, dtype=float)
    n = len(pvals)
    if n == 0:
        return []
    order = np.argsort(pvals)
    ranked = pvals[order]
    adj = ranked * n / (np.arange(n) + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0, 1)
    out = np.empty(n, dtype=float)
    out[order] = adj
    return out.tolist()

def run_olink_single_protein_scan(df_olink: pd.DataFrame, exposure_vars: List[str], axis_def: pd.DataFrame) -> pd.DataFrame:
    if not STATS_AVAILABLE:
        return pd.DataFrame()

    target_proteins = sorted(axis_def["protein_col"].dropna().unique().tolist()) if not axis_def.empty else []
    target_proteins = [c for c in target_proteins if c in df_olink.columns]
    if len(target_proteins) == 0:
        return pd.DataFrame()

    axis_map = {}
    for _, r in axis_def.iterrows():
        axis_map.setdefault(r["protein_col"], set()).add(r["axis_name"])

    tech_covs = choose_best_technical_covariates(df_olink)
    base_covs = [c for c in ["age", "sex", "smoking_status", "bmi", "education", "alcohol_freq", "urban_rural", "ethnicity"] if c in df_olink.columns]
    covs = base_covs + tech_covs

    rows = []
    for protein in target_proteins:
        for exp in exposure_vars:
            if exp not in df_olink.columns:
                continue
            work = df_olink.copy()
            work[exp] = zscore(work[exp])
            work[protein] = zscore(work[protein])
            model_df = build_linear_df(work, protein, [exp] + covs, get_cat_cols(covs))
            if model_df.empty or model_df.shape[0] < MIN_OLINK_ROWS:
                continue
            fit = fit_ols_hc3(model_df, protein)
            if fit is None or exp not in fit.params.index:
                continue
            rows.append({
                "protein": protein,
                "protein_axes": ";".join(sorted(axis_map.get(protein, []))),
                "exposure": exp,
                "beta": fit.params[exp],
                "se": fit.bse[exp],
                "t": fit.tvalues[exp],
                "p": fit.pvalues[exp],
                "n": model_df.shape[0],
                "covs_used": ";".join(covs),
            })
    out = pd.DataFrame(rows)
    if not out.empty:
        out["fdr_p"] = out.groupby("exposure")["p"].transform(lambda s: pd.Series(np.minimum(1, np.array(_bh_fdr(s.tolist()))), index=s.index))
    return out

def run_olink_axis_to_bca(df_olink: pd.DataFrame) -> pd.DataFrame:
    axis_vars = [c for c in df_olink.columns if c.endswith("_score")]
    axis_vars = [c for c in axis_vars if c not in ["traffic_combustion_score", "particle_score", "traffic_combustion_score_pca", "particle_score_pca"]]

    rows = []
    for axis_var in axis_vars:
        covs = [c for c in MODEL_CANDIDATES["Model2c"] if c in df_olink.columns]
        model_df = build_model_df(df_olink.copy(), axis_var, covs, get_cat_cols(covs))
        if model_df.empty or model_df.shape[0] < MIN_OLINK_ROWS or model_df["status_bc"].sum() < MIN_OLINK_EVENTS:
            continue
        try:
            cph = fit_cox(model_df)
            rows.append(extract_term(
                cph, axis_var,
                {
                    "protein_axis": axis_var,
                    "analysis": "axis_to_bca",
                    "model": "Model2c",
                    "n": model_df.shape[0],
                    "events": int(model_df["status_bc"].sum()),
                    "covs_used": ";".join(covs)
                }
            ))
        except Exception as e:
            log(f"[WARN] Protein-axis-to-BCa model failed | axis={axis_var} | {e}")
            continue
    return pd.DataFrame(rows)

def run_olink_single_protein_to_bca(df_olink: pd.DataFrame, axis_def: pd.DataFrame) -> pd.DataFrame:
    target_proteins = sorted(axis_def["protein_col"].dropna().unique().tolist()) if not axis_def.empty else []
    target_proteins = [c for c in target_proteins if c in df_olink.columns]

    rows = []
    for protein in target_proteins:
        tmp = df_olink.copy()
        tmp[protein] = zscore(tmp[protein])
        covs = [c for c in MODEL_CANDIDATES["Model2c"] if c in tmp.columns]
        model_df = build_model_df(tmp, protein, covs, get_cat_cols(covs))
        if model_df.empty or model_df.shape[0] < MIN_OLINK_ROWS or model_df["status_bc"].sum() < MIN_OLINK_EVENTS:
            continue
        try:
            cph = fit_cox(model_df)
            row = extract_term(
                cph, protein,
                {
                    "protein": protein,
                    "analysis": "protein_to_bca",
                    "model": "Model2c",
                    "n": model_df.shape[0],
                    "events": int(model_df["status_bc"].sum()),
                    "covs_used": ";".join(covs)
                }
            )
            rows.append(row)
        except Exception as e:
            log(f"[WARN] Single-protein-to-BCa model failed | protein={protein} | {e}")
            continue

    out = pd.DataFrame(rows)
    if not out.empty:
        out["fdr_p"] = _bh_fdr(out["p"].tolist())
    return out

# =========================================================
# 13. SMR and triangulation integration
# =========================================================

# =========================================================
# 14. Entry point
# =========================================================
def main(args: argparse.Namespace):
    log("=" * 80)
    log("UKB cohort analysis pipeline")
    log(f"[INFO] Run directory: {RUN_DIR}")
    log("=" * 80)

    config = {
        "run_id": RUN_ID,
        "raw_zip_path": RAW_ZIP_PATH,
        "result_root": RESULT_ROOT,
        "main_exposures": MAIN_EXPOSURES,
        "support_exposures": SUPPORT_EXPOSURES,
        "subsets": list(SUBSET_FILTERS.keys()),
        "models": MODEL_CANDIDATES,
        "default_cancer_censor_date": str(DEFAULT_CANCER_CENSOR_DATE.date()),
        "patsy_available": PATSY_AVAILABLE,
        "statsmodels_available": STATS_AVAILABLE,
        "olink_candidates": OLINK_MERGED_CANDIDATES,
        "protein_axis_rules": PROTEIN_AXIS_RULES,
    }
    with open(os.path.join(DIR_LOG, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    # -----------------------------------------------------
    # A. Read inputs and build the analysis dataset
    # -----------------------------------------------------
    df_raw = unzip_read_csv(RAW_ZIP_PATH)
    log(f"[INFO] Raw dataset shape: {df_raw.shape}")

    df_main = build_master_analysis(df_raw)
    del df_raw

    # -----------------------------------------------------
    # B. subset summary
    # -----------------------------------------------------
    subset_rows = []
    for name, fn in SUBSET_FILTERS.items():
        dset = fn(df_main)
        subset_rows.append({
            "subset": name,
            "n": dset.shape[0],
            "events": int(dset["status_bc"].sum()) if not dset.empty else 0,
            "mean_fu_years": dset["fu_time_bc_years"].mean() if not dset.empty else np.nan,
            "mean_address_years": dset["address_years"].mean() if ("address_years" in dset.columns and not dset.empty) else np.nan
        })
    pd.DataFrame(subset_rows).to_excel(os.path.join(DIR_DIAG, "D1_subset_summary.xlsx"), index=False)

    # -----------------------------------------------------
    # C. Main results
    # -----------------------------------------------------
    all_results = []
    all_diag = []

    for exposure in MAIN_EXPOSURES + SUPPORT_EXPOSURES:
        if exposure not in df_main.columns:
            continue
        for subset_name in ["main", "address5", "address5_lag5"]:
            log(f"[INFO] Main Cox analysis | exposure={exposure} | subset={subset_name}")
            res, diag = run_main_result(df_main, exposure, subset_name)
            if not res.empty:
                all_results.append(res)
            if not diag.empty:
                all_diag.append(diag)

    df_results = pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()
    df_diag = pd.concat(all_diag, ignore_index=True) if all_diag else pd.DataFrame()

    if not df_results.empty:
        df_main_results = df_results.loc[df_results["exposure"].isin(MAIN_EXPOSURES)].copy()
        df_main_results.to_excel(os.path.join(DIR_MAIN, "T1_main_results.xlsx"), index=False)

        if not df_main_results.empty:
            pivot_main = df_main_results.pivot_table(
                index=["exposure", "subset"],
                columns="model",
                values=["HR", "LCL", "UCL", "p", "n", "events"],
                aggfunc="first"
            )
            pivot_main.to_excel(os.path.join(DIR_MAIN, "T1_main_results_pivot.xlsx"))

        df_support_results = df_results.loc[df_results["exposure"].isin(SUPPORT_EXPOSURES)].copy()
        df_support_results.to_excel(os.path.join(DIR_SUPPORT, "S1_support_results.xlsx"), index=False)

        if not df_support_results.empty:
            pivot_support = df_support_results.pivot_table(
                index=["exposure", "subset"],
                columns="model",
                values=["HR", "LCL", "UCL", "p", "n", "events"],
                aggfunc="first"
            )
            pivot_support.to_excel(os.path.join(DIR_SUPPORT, "S1_support_results_pivot.xlsx"))

        df_combined = df_results.loc[df_results["exposure"].isin(["traffic_combustion_score", "particle_score"])].copy()
        if not df_combined.empty:
            df_combined.to_excel(os.path.join(DIR_MAIN, "T2_combined_exposure_results.xlsx"), index=False)

    if not df_diag.empty:
        df_diag.to_excel(os.path.join(DIR_DIAG, "D2_missingness_and_complete_case.xlsx"), index=False)

    # -----------------------------------------------------
    # D. Dose-response analyses
    # -----------------------------------------------------
    dose_rows = []
    for exposure in ["nox_2010", "pm25_2010", "traffic_combustion_score", "particle_score"]:
        if exposure not in df_main.columns:
            continue
        for subset_name in ["main", "address5", "address5_lag5"]:
            log(f"[INFO] Dose-response analysis | exposure={exposure} | subset={subset_name}")
            res = run_dose_response(df_main, exposure, subset_name)
            if not res.empty:
                dose_rows.append(res)

    df_dose = pd.concat(dose_rows, ignore_index=True) if dose_rows else pd.DataFrame()
    if not df_dose.empty:
        df_dose.to_excel(os.path.join(DIR_DOSE, "T3_dose_response.xlsx"), index=False)

    # -----------------------------------------------------
    # E. Exposure distributions and correlations
    # -----------------------------------------------------
    exp_cols = [c for c in MAIN_EXPOSURES + SUPPORT_EXPOSURES if c in df_main.columns]
    if exp_cols:
        dist_rows = []
        for c in exp_cols:
            s = pd.to_numeric(df_main[c], errors="coerce")
            dist_rows.append({
                "exposure": c,
                "n_nonmissing": int(s.notna().sum()),
                "missing_n": int(s.isna().sum()),
                "mean": s.mean(),
                "sd": s.std(),
                "p25": s.quantile(0.25),
                "median": s.quantile(0.50),
                "p75": s.quantile(0.75),
                "min": s.min(),
                "max": s.max()
            })
        pd.DataFrame(dist_rows).to_excel(os.path.join(DIR_DIAG, "D3_exposure_distribution.xlsx"), index=False)

        corr = df_main[exp_cols].corr(numeric_only=True)
        corr.to_excel(os.path.join(DIR_DIAG, "D4_exposure_correlation.xlsx"))


    # -----------------------------------------------------
    # F. Restricted cubic spline analyses
    # -----------------------------------------------------
    rcs_coef_rows = []
    rcs_pred_rows = []

    for exposure in RCS_TARGET_EXPOSURES:
        if exposure not in df_main.columns:
            continue
        for subset_name in RCS_TARGET_SUBSETS:
            log(f"[INFO] RCS analysis | exposure={exposure} | subset={subset_name}")
            coef_df, pred_df = run_rcs_analysis(df_main, exposure, subset_name)
            if not coef_df.empty:
                rcs_coef_rows.append(coef_df)
            if not pred_df.empty:
                rcs_pred_rows.append(pred_df)

    if rcs_coef_rows:
        pd.concat(rcs_coef_rows, ignore_index=True).to_excel(
            os.path.join(DIR_RCS, "R1_rcs_coefficients.xlsx"),
            index=False
        )

    if rcs_pred_rows:
        pd.concat(rcs_pred_rows, ignore_index=True).to_excel(
            os.path.join(DIR_RCS, "R2_rcs_prediction_grid.xlsx"),
            index=False
        )


    # -----------------------------------------------------
    # G. Olink domain and single-protein analyses
    # -----------------------------------------------------
    olink_meta_rows = []
    df_olink_raw, olink_path = load_olink_merged()
    if df_olink_raw is not None:
        olink_meta_rows.append({"module": "olink", "status": "loaded", "path": olink_path, "n": df_olink_raw.shape[0], "p": df_olink_raw.shape[1]})

        df_olink = prepare_olink_merged(df_olink_raw, df_main)
        axis_def = build_protein_axis_definitions(df_olink)
        if not axis_def.empty:
            axis_def.to_excel(os.path.join(DIR_OLINK, "O5_protein_axis_definitions.xlsx"), index=False)

        df_olink2, axis_summary = construct_protein_axes(df_olink, axis_def)
        if not axis_summary.empty:
            axis_summary.to_excel(os.path.join(DIR_OLINK, "O6_protein_axis_summary.xlsx"), index=False)

        exposure_vars = [c for c in ["traffic_combustion_score", "particle_score", "nox_2010", "pm25_2010"] if c in df_olink2.columns]

        axis_assoc = run_olink_axis_association(df_olink2, exposure_vars)
        if not axis_assoc.empty:
            axis_assoc.to_excel(os.path.join(DIR_OLINK, "O7_pollution_to_protein_axis.xlsx"), index=False)

        single_assoc = run_olink_single_protein_scan(df_olink2, exposure_vars, axis_def)
        if not single_assoc.empty:
            single_assoc.to_excel(os.path.join(DIR_OLINK, "O8_pollution_to_single_protein.xlsx"), index=False)

        axis_to_bca = run_olink_axis_to_bca(df_olink2)
        if not axis_to_bca.empty:
            axis_to_bca.to_excel(os.path.join(DIR_OLINK, "O9_protein_axis_to_BCa.xlsx"), index=False)

        protein_to_bca = run_olink_single_protein_to_bca(df_olink2, axis_def)
        if not protein_to_bca.empty:
            protein_to_bca.to_excel(os.path.join(DIR_OLINK, "O10_single_protein_to_BCa.xlsx"), index=False)


        # Export the merged Olink table with derived axis scores for reuse.
        keep_axis_cols = [c for c in df_olink2.columns if c.endswith("_score") or c.endswith("_mean_z")]
        bridge_cols = [c for c in ["eid", "status_bc", "fu_time_bc_years", "traffic_combustion_score", "particle_score", "age", "sex", "smoking_status"] if c in df_olink2.columns]
        out_cols = list(dict.fromkeys(bridge_cols + keep_axis_cols))
        if out_cols:
            df_olink2.loc[:, out_cols].to_excel(os.path.join(DIR_OLINK, "O11_olink_with_axis_scores.xlsx"), index=False)

    if olink_meta_rows:
        pd.DataFrame(olink_meta_rows).to_excel(os.path.join(DIR_OLINK, "O12_external_module_status.xlsx"), index=False)

    log("[DONE] Analysis finished")
    log(RUN_DIR)

if __name__ == "__main__":
    main(parse_args())