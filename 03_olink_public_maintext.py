# -*- coding: utf-8 -*-

"""UK Biobank Olink analysis pipeline for the bladder cancer manuscript.

Public release aligned with the proteomic analyses reported in the manuscript.
This script starts from the UK Biobank source table and the Olink proteomics
file, performs sample harmonisation and technical adjustment, constructs the
predefined proteomic response domains, and runs the main association models.

Example
-------
python 03_olink_public_maintext.py
python 03_olink_public_maintext.py --air-input ./data/ukb_moduleA_backup_raw.zip --proteomics ./data/proteomics.csv --field-names ./data/field_names.txt --result-root ./results/olink
"""

import os
import argparse
import re
import ast
import json
import zipfile
import warnings
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from lifelines import CoxPHFitter
from lifelines.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore")
warnings.simplefilter("ignore", ConvergenceWarning)

try:
    import statsmodels.api as sm
    STATS_AVAILABLE = True
except Exception:
    STATS_AVAILABLE = False




# =========================================================
# 0. Paths and runtime configuration
# =========================================================
DEFAULT_AIR_INPUT = os.environ.get("OLINK_AIR_INPUT", os.path.join("data", "ukb_moduleA_backup_raw.zip"))
DEFAULT_PROTEOMICS_INPUT = os.environ.get("OLINK_PROTEOMICS_INPUT", os.path.join("data", "proteomics.csv"))
DEFAULT_FIELD_NAMES = os.environ.get("OLINK_FIELD_NAMES", os.path.join("data", "field_names.txt"))
DEFAULT_RESULT_ROOT = os.environ.get("OLINK_RESULT_ROOT", os.path.join("results", "olink"))

AIR_RAW_CANDIDATES = [DEFAULT_AIR_INPUT]
PROTEOMICS_RAW_CANDIDATES = [DEFAULT_PROTEOMICS_INPUT]
FIELD_NAME_CANDIDATES = [DEFAULT_FIELD_NAMES]
RESULT_ROOT = DEFAULT_RESULT_ROOT

RUN_ID = datetime.now().strftime("RUN_%Y%m%d_%H%M%S")
RUN_DIR = ""
DIR_MAIN = ""
DIR_QC = ""
DIR_AXIS = ""
DIR_ASSOC = ""
DIR_DATA = ""
DIR_LOG = ""


def initialize_run_directories(result_root: str) -> None:
    global RESULT_ROOT, RUN_ID, RUN_DIR
    global DIR_MAIN, DIR_QC, DIR_AXIS, DIR_ASSOC, DIR_DATA, DIR_LOG

    RESULT_ROOT = result_root
    RUN_ID = datetime.now().strftime("RUN_%Y%m%d_%H%M%S")
    RUN_DIR = os.path.join(RESULT_ROOT, RUN_ID)
    DIR_MAIN = os.path.join(RUN_DIR, "01_main")
    DIR_QC = os.path.join(RUN_DIR, "02_qc")
    DIR_AXIS = os.path.join(RUN_DIR, "03_axis")
    DIR_ASSOC = os.path.join(RUN_DIR, "04_association")
    DIR_DATA = os.path.join(RUN_DIR, "05_data")
    DIR_LOG = os.path.join(RUN_DIR, "06_logs")

    for path in [RUN_DIR, DIR_MAIN, DIR_QC, DIR_AXIS, DIR_ASSOC, DIR_DATA, DIR_LOG]:
        os.makedirs(path, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="UK Biobank Olink analysis pipeline")
    parser.add_argument("--air-input", default=DEFAULT_AIR_INPUT, help="Path to the UK Biobank source table or an exposure-only table")
    parser.add_argument("--proteomics", default=DEFAULT_PROTEOMICS_INPUT, help="Path to the Olink proteomics table")
    parser.add_argument("--field-names", default=DEFAULT_FIELD_NAMES, help="Optional text file containing allowed protein field names")
    parser.add_argument("--result-root", default=DEFAULT_RESULT_ROOT, help="Directory used to create the timestamped run folder")
    return parser.parse_args()

# =========================================================
# 1. Core settings
# =========================================================
DEFAULT_CANCER_CENSOR_DATE = pd.Timestamp("2023-05-31")
SPECIAL_MISSING_VALUES = {-1, -3, -7, -10, -11, -13, -17, -21, -23, -27, -121, -313, -818}
MIN_ROWS_LINEAR = 100
MIN_ROWS_COX = 100
MIN_EVENTS_COX = 10
# Optional manual overrides used only when automatic detection fails
MANUAL_PROTEOMICS_EID_COL = None
MANUAL_TECH_COLS: List[str] = []
MANUAL_PROTEIN_COLS: List[str] = []

MAIN_EXPOSURES = [
    "traffic_combustion_score",
    "particle_score",
]

SUPPORT_EXPOSURES = [
    "no2_2010",
    "nox_2010",
    "pm25_2010",
    "pm25_absorb_2010",
    "pm10_2010",
    "pmcoarse_2010",
]

ALL_ANALYSIS_EXPOSURES = MAIN_EXPOSURES + SUPPORT_EXPOSURES

MODEL_COVS = [
    "age",
    "sex",
    "smoking_status",
    "bmi",
    "education",
    "alcohol_freq",
    "urban_rural",
    "ethnicity",
]

TECH_KEYWORDS = [
    "plate", "batch", "panel", "site", "qc", "assay", "warning", "sample",
    "aliquot", "olink", "lod", "limit_of_detection", "run", "shipment",
]

PROTEIN_AXIS_RULES = {
    "endothelial_adhesion": ["SELE", "TIE1", "ICAM1", "VCAM1", "PECAM1", "ANGPT2", "FLT1", "KDR"],
    "inflammatory_innate": ["SERPINA4", "CXCL8", "CXCL9", "CXCL10", "IL6", "TNFRSF19", "TNFRSF1A", "TNFRSF1B"],
    "epithelial_remodeling": ["REG4", "CDHR5", "MMP7", "EPCAM", "KRT19", "TFF3"],
    "metabolism_stress": ["HK2", "TFRC", "SMOC1", "CA9", "GDF15", "HMOX1"],
    "coagulation_kallikrein": ["KLKB1", "F2", "F3", "SERPINF2", "PLAT", "PLAU"],
}

PRIMARY_EXPOSURES = {
    "pm25_2010": "participant.p24006",
    "no2_2010": "participant.p24003",
    "nox_2010": "participant.p24004",
    "pm25_absorb_2010": "participant.p24007",
}

SECONDARY_EXPOSURES = {
    "pm10_2010": "participant.p24005",
    "pmcoarse_2010": "participant.p24008",
}

BASELINE_COLS = {
    "eid": "participant.eid",
    "baseline_date": "participant.p53_i0",
    "address_years": "participant.p699_i0",
    "age": "participant.p21022",
    "sex": "participant.p31",
    "ethnicity": "participant.p21000_i0",
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


# =========================================================
# 2. Utility helpers
# =========================================================
def log(msg: str) -> None:
    print(msg, flush=True)
    with open(os.path.join(DIR_LOG, "analysis_log.txt"), "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def first_existing_path(candidates: List[str]) -> Optional[str]:
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def safe_read_table(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        try:
            return pd.read_csv(path, low_memory=False)
        except UnicodeDecodeError:
            return pd.read_csv(path, low_memory=False, encoding="gb18030")
    if ext in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    if ext == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported file type: {path}")


def unzip_read_csv(zip_path: str) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path, "r") as zf:
        csv_names = [x for x in zf.namelist() if x.lower().endswith(".csv")]
        if len(csv_names) != 1:
            raise ValueError(f"The zip archive must contain exactly one CSV file; found {len(csv_names)}.")
        csv_name = csv_names[0]
        log(f"[INFO] Reading CSV from zip archive: {csv_name}")
        with zf.open(csv_name) as f:
            return pd.read_csv(f, low_memory=False)


def normalize_token(x: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "", str(x).upper())


def normalize_gene_name(x: str) -> str:
    x = str(x).upper().strip()
    x = re.sub(r"NPX$", "", x)
    x = re.sub(r"[^A-Z0-9]+", "", x)
    return x


def zscore(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    mu = s.mean(skipna=True)
    sd = s.std(skipna=True)
    if pd.isna(sd) or sd == 0:
        return pd.Series(np.nan, index=s.index)
    return (s - mu) / sd


def ukb_to_numeric_clean(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    return s.mask(s.isin(SPECIAL_MISSING_VALUES), np.nan)


def map_sex(x):
    if pd.isna(x):
        return np.nan
    try:
        x = int(float(x))
    except Exception:
        return np.nan
    return {0: "Female", 1: "Male"}.get(x, np.nan)


def map_smoking(x):
    if pd.isna(x):
        return np.nan
    try:
        x = int(float(x))
    except Exception:
        return np.nan
    return {0: "Never", 1: "Former", 2: "Current"}.get(x, np.nan)


def parse_education_high(x):
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


def find_cols_by_prefix(cols: List[str], prefix: str) -> List[str]:
    out = [c for c in cols if c.startswith(prefix)]

    def _idx(x):
        m = re.search(r"_i(\d+)", x)
        return int(m.group(1)) if m else 9999

    return sorted(out, key=_idx)


def bh_fdr(pvals: List[float]) -> List[float]:
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


def get_cat_cols(cols: List[str]) -> List[str]:
    return [c for c in cols if c in ["sex", "smoking_status", "alcohol_freq", "urban_rural", "ethnicity"]]


def fit_cox(df_model: pd.DataFrame) -> CoxPHFitter:
    if df_model.empty:
        raise ValueError("empty df for Cox")
    try:
        cph = CoxPHFitter()
        cph.fit(df_model, duration_col="fu_time_bc_years", event_col="status_bc")
        return cph
    except Exception:
        cph = CoxPHFitter(penalizer=1e-6)
        cph.fit(df_model, duration_col="fu_time_bc_years", event_col="status_bc")
        return cph


def build_linear_df(df: pd.DataFrame, y: str, xvars: List[str], cat_cols: List[str]) -> pd.DataFrame:
    cols = [y] + [x for x in xvars if x in df.columns]
    tmp = df.loc[:, cols].copy().replace([np.inf, -np.inf], np.nan)
    tmp = tmp.dropna()
    if tmp.empty:
        return pd.DataFrame()

    use_cat = [c for c in cat_cols if c in tmp.columns]
    if use_cat:
        tmp = pd.get_dummies(tmp, columns=use_cat, drop_first=True, dtype=float)

    for c in tmp.columns:
        tmp[c] = pd.to_numeric(tmp[c], errors="coerce")
    tmp = tmp.replace([np.inf, -np.inf], np.nan).dropna()
    if tmp.empty:
        return pd.DataFrame()

    drop_cols = []
    for c in tmp.columns:
        if c == y:
            continue
        if tmp[c].nunique(dropna=True) <= 1:
            drop_cols.append(c)
            continue
        v = tmp[c].var()
        if pd.isna(v) or v < 1e-12:
            drop_cols.append(c)
    if drop_cols:
        tmp = tmp.drop(columns=drop_cols, errors="ignore")
    if y not in tmp.columns or tmp.shape[1] < 2:
        return pd.DataFrame()
    return tmp


def fit_ols_hc3(df: pd.DataFrame, y: str):
    if (not STATS_AVAILABLE) or df.empty or y not in df.columns:
        return None

    x = df.drop(columns=[y]).copy()
    yy = pd.to_numeric(df[y], errors="coerce")

    # Force all predictors to numeric to avoid pandas nullable or object dtypes in statsmodels
    for c in x.columns:
        x[c] = pd.to_numeric(x[c], errors="coerce")

    x = sm.add_constant(x, has_constant="add")
    x = x.astype(float)
    yy = yy.astype(float)

    valid = yy.notna()
    if hasattr(x, "notna"):
        valid &= x.notna().all(axis=1)
    x = x.loc[valid, :]
    yy = yy.loc[valid]

    if x.empty or yy.empty or x.shape[0] < 20:
        return None

    return sm.OLS(yy, x).fit(cov_type="HC3")


def build_cox_df(df: pd.DataFrame, exposure: str, covs: List[str], cat_cols: List[str]) -> pd.DataFrame:
    cols = ["fu_time_bc_years", "status_bc", exposure] + [c for c in covs if c in df.columns]
    tmp = df.loc[:, cols].copy().replace([np.inf, -np.inf], np.nan).dropna()
    if tmp.empty:
        return pd.DataFrame()

    use_cat = [c for c in cat_cols if c in tmp.columns]
    if use_cat:
        tmp = pd.get_dummies(tmp, columns=use_cat, drop_first=True, dtype=float)

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

    if exposure not in tmp.columns or tmp.shape[1] < 3:
        return pd.DataFrame()
    return tmp


def extract_cox_term(cph: CoxPHFitter, term: str, extra: dict) -> dict:
    s = cph.summary.loc[term]
    out = dict(extra)
    out.update({
        "term": term,
        "coef": s["coef"],
        "HR": np.exp(s["coef"]),
        "LCL": np.exp(s["coef lower 95%"]),
        "UCL": np.exp(s["coef upper 95%"]),
        "p": s["p"],
    })
    return out


def detect_eid_col(df: pd.DataFrame) -> Optional[str]:
    if MANUAL_PROTEOMICS_EID_COL and MANUAL_PROTEOMICS_EID_COL in df.columns:
        return MANUAL_PROTEOMICS_EID_COL
    candidates = [
        "eid", "participant.eid", "n_eid", "id", "ID", "IID", "sample_id", "SampleID"
    ]
    for c in candidates:
        if c in df.columns:
            return c
    for c in df.columns:
        if normalize_token(c) == "EID":
            return c
    for c in df.columns:
        if "EID" in normalize_token(c):
            return c
    return None


def choose_best_duplicate_rows(df: pd.DataFrame, value_cols: List[str]) -> pd.DataFrame:
    if df.empty:
        return df
    tmp = df.copy()
    value_cols = [c for c in value_cols if c in tmp.columns]
    if value_cols:
        tmp["_nonmissing_value_count"] = tmp[value_cols].notna().sum(axis=1)
    else:
        tmp["_nonmissing_value_count"] = tmp.notna().sum(axis=1)
    tmp = tmp.sort_values(["eid", "_nonmissing_value_count"], ascending=[True, False])
    tmp = tmp.drop_duplicates(subset=["eid"], keep="first")
    return tmp.drop(columns=["_nonmissing_value_count"], errors="ignore")



def normalize_protein_symbol(x: str) -> str:
    return str(x).strip().lower()


def load_field_name_whitelist() -> Tuple[set, Optional[str]]:
    path = first_existing_path(FIELD_NAME_CANDIDATES)
    if path is None:
        return set(), None
    fields = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            val = line.strip()
            if val:
                fields.append(normalize_protein_symbol(val))
    return set(fields), path


def exact_protein_lookup_map(raw_protein_cols: List[str]) -> Dict[str, str]:
    """
    Build an exact lookup from the raw proteomics column name to itself.
    No fuzzy, token-based, or substring matching is allowed here.
    """
    lookup = {}
    for c in raw_protein_cols:
        key = normalize_protein_symbol(c)
        if key not in lookup:
            lookup[key] = c
    return lookup


def validate_axis_definition_strict(axis_def: pd.DataFrame) -> pd.DataFrame:
    """Validate that the axis definition uses only exact, one-to-one mappings."""
    if axis_def is None or axis_def.empty:
        return pd.DataFrame(columns=list(axis_def.columns) + ['strict_validation_pass']) if isinstance(axis_def, pd.DataFrame) else pd.DataFrame()

    out = axis_def.copy()
    out['mapped_gene_norm'] = out['mapped_gene'].map(normalize_protein_symbol)
    out['protein_col_raw_norm'] = out['protein_col_raw'].map(normalize_protein_symbol)
    out['strict_validation_pass'] = (out['mapped_gene_norm'] == out['protein_col_raw_norm']).astype(int)

    bad = out.loc[out['strict_validation_pass'] != 1, ['axis_name', 'mapped_gene', 'protein_col_raw']].copy()
    if not bad.empty:
        raise ValueError(
             'Detected non-exact protein-domain mappings. Please review the entries below:\n' +
            bad.to_string(index=False)
        )

    return out.drop(columns=['mapped_gene_norm', 'protein_col_raw_norm'])



def build_validated_protein_reference(axis_def: pd.DataFrame) -> pd.DataFrame:
    """Extract the validated one-to-one protein reference from the axis definition."""
    if axis_def is None or axis_def.empty:
        return pd.DataFrame(columns=['protein_col_adjusted', 'mapped_gene', 'protein_col_raw', 'axis_name', 'is_representative_protein', 'strict_validation_pass'])

    cols = [c for c in ['protein_col_adjusted', 'mapped_gene', 'protein_col_raw', 'axis_name', 'is_representative_protein', 'strict_validation_pass'] if c in axis_def.columns]
    ref = axis_def[cols].drop_duplicates().copy()
    if 'strict_validation_pass' in ref.columns:
        ref = ref.loc[ref['strict_validation_pass'] == 1, :].copy()

    dup = ref.groupby('protein_col_adjusted')[['mapped_gene', 'protein_col_raw']].nunique(dropna=False)
    bad = dup.loc[(dup['mapped_gene'] > 1) | (dup['protein_col_raw'] > 1), :]
    if not bad.empty:
        raise ValueError('A validated protein maps to multiple genes or raw columns. The pipeline stopped to avoid ambiguous output.')

    return ref.reset_index(drop=True)


def enforce_validated_mapping_on_table(df: pd.DataFrame, axis_def: pd.DataFrame, table_name: str) -> pd.DataFrame:
    """Validate that downstream single-protein outputs only use the exact proteins defined in the validated axis table."""
    if df is None or df.empty:
        return pd.DataFrame() if df is None else df

    if 'protein_col_adjusted' not in df.columns:
        return df.copy()

    ref = build_validated_protein_reference(axis_def)
    if ref.empty:
        raise ValueError(f'{table_name}: the validated protein reference is empty, so downstream validation cannot proceed.')

    ref_one = ref[['protein_col_adjusted', 'mapped_gene', 'protein_col_raw']].drop_duplicates().copy()
    out = df.copy()
    out = out.merge(ref_one, on='protein_col_adjusted', how='inner', suffixes=('', '__ref'))
    if out.shape[0] != df.shape[0]:
        missing = sorted(set(df['protein_col_adjusted'].astype(str)) - set(out['protein_col_adjusted'].astype(str)))
        raise ValueError(
            f'{table_name}: records were found outside the validated protein list. Unexpected protein_col_adjusted values: ' + ';'.join(missing[:20])
        )

    for col in ['mapped_gene', 'protein_col_raw']:
        ref_col = f'{col}__ref'
        if ref_col not in out.columns:
            continue
        if col in out.columns:
            lhs = out[col].astype(str).map(normalize_protein_symbol)
            rhs = out[ref_col].astype(str).map(normalize_protein_symbol)
            bad_mask = out[col].notna() & out[ref_col].notna() & (lhs != rhs)
            if bad_mask.any():
                bad = out.loc[bad_mask, ['protein_col_adjusted', col, ref_col]].copy()
                raise ValueError(
                    f'{table_name}: {col} does not match the validated reference.\n' + bad.head(20).to_string(index=False)
                )
        out[col] = out[ref_col]
        out = out.drop(columns=[ref_col])

    return out


# =========================================================
# 3. Read and standardise the air pollution / cohort table
# =========================================================
def load_air_raw() -> Tuple[pd.DataFrame, str]:
    path = first_existing_path(AIR_RAW_CANDIDATES)
    if path is None:
        raise FileNotFoundError("Air pollution / UK Biobank cohort input not found.")
    if path.lower().endswith(".zip"):
        df = unzip_read_csv(path)
    else:
        df = safe_read_table(path)
    log(f"[INFO] Air input: {path} | shape={df.shape}")
    return df, path


def is_ukb_raw_table(df: pd.DataFrame) -> bool:
    return all(c in df.columns for c in ["participant.eid", "participant.p53_i0", "participant.p24003", "participant.p40005_i0"])


def is_air_exposure_only_table(df: pd.DataFrame) -> bool:
    cols = set(df.columns)
    has_id = ("Participant ID" in cols) or ("participant.eid" in cols) or ("eid" in cols)
    exposure_markers = [
        "Nitrogen dioxide air pollution; 2010",
        "Nitrogen oxides air pollution; 2010",
        "Particulate matter air pollution (pm2.5); 2010",
        "Particulate matter air pollution (pm2.5) absorbance; 2010",
        "Particulate matter air pollution (pm10); 2010",
        "Particulate matter air pollution 2.5-10um; 2010",
    ]
    has_exposure = any(c in cols for c in exposure_markers)
    lacks_outcome = not any(c in cols for c in ["status_bc", "fu_time_bc_years", "participant.p53_i0"])
    return has_id and has_exposure and lacks_outcome


def standardize_exposure_only_air_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    rename_map = {
        "Participant ID": "eid",
        "participant.eid": "eid",
        "Nitrogen dioxide air pollution; 2010": "no2_2010",
        "Nitrogen oxides air pollution; 2010": "nox_2010",
        "Particulate matter air pollution (pm2.5); 2010": "pm25_2010",
        "Particulate matter air pollution (pm2.5) absorbance; 2010": "pm25_absorb_2010",
        "Particulate matter air pollution (pm10); 2010": "pm10_2010",
        "Particulate matter air pollution 2.5-10um; 2010": "pmcoarse_2010",
    }
    out = out.rename(columns={k: v for k, v in rename_map.items() if k in out.columns})

    # Additional token-based column matching
    auto_map = {}
    for c in out.columns:
        nc = normalize_token(c)
        if nc in ["EID", "PARTICIPANTID", "PARTICIPANTEID"]:
            auto_map[c] = "eid"
        elif nc in ["NITROGENDIOXIDEAIRPOLLUTION2010", "NO22010"]:
            auto_map[c] = "no2_2010"
        elif nc in ["NITROGENOXIDESAIRPOLLUTION2010", "NOX2010"]:
            auto_map[c] = "nox_2010"
        elif nc in ["PARTICULATEMATTERAIRPOLLUTIONPM252010", "PM252010"]:
            auto_map[c] = "pm25_2010"
        elif nc in ["PARTICULATEMATTERAIRPOLLUTIONPM25ABSORBANCE2010", "PM25ABSORB2010", "PM25ABSORBANCE2010"]:
            auto_map[c] = "pm25_absorb_2010"
        elif nc in ["PARTICULATEMATTERAIRPOLLUTIONPM102010", "PM102010"]:
            auto_map[c] = "pm10_2010"
        elif nc in ["PARTICULATEMATTERAIRPOLLUTION2510UM2010", "PMCOARSE2010", "PM2510UM2010"]:
            auto_map[c] = "pmcoarse_2010"
    if auto_map:
        out = out.rename(columns=auto_map)

    keep = [c for c in ["eid", "no2_2010", "nox_2010", "pm25_2010", "pm25_absorb_2010", "pm10_2010", "pmcoarse_2010"] if c in out.columns]
    out = out.loc[:, keep].copy()
    if "eid" not in out.columns:
        raise ValueError("No eid column was detected in the exposure-only air pollution table. Check whether a participant ID column is present or edit the script configuration.")
    out["eid"] = pd.to_numeric(out["eid"], errors="coerce")
    out = out.loc[out["eid"].notna(), :].copy()
    out = out.drop_duplicates(subset=["eid"], keep="first")
    return out


def load_ukb_zip_raw() -> pd.DataFrame:
    zip_candidates = [p for p in AIR_RAW_CANDIDATES if str(p).lower().endswith('.zip') and os.path.exists(p)]
    if not zip_candidates:
        raise FileNotFoundError("The provided air pollution file contains exposures only, but no UK Biobank source zip was available to recover outcomes and covariates.")
    zip_path = zip_candidates[0]
    log(f"[INFO] Reading UK Biobank source zip to recover outcomes and covariates: {zip_path}")
    return unzip_read_csv(zip_path)


def standardize_clean_air_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    rename_map = {}
    explicit_map = {
        "participant.eid": "eid",
        "Participant ID": "eid",
        "eid": "eid",
        "status_bc": "status_bc",
        "fu_time_bc_years": "fu_time_bc_years",
        "no2_2010": "no2_2010",
        "nox_2010": "nox_2010",
        "pm25_2010": "pm25_2010",
        "pm25_absorb_2010": "pm25_absorb_2010",
        "pm10_2010": "pm10_2010",
        "pmcoarse_2010": "pmcoarse_2010",
    }
    for k, v in explicit_map.items():
        if k in out.columns:
            rename_map[k] = v
    for c in out.columns:
        nc = normalize_token(c)
        if nc in ["EID", "PARTICIPANTID", "PARTICIPANTEID"]:
            rename_map[c] = "eid"
        elif nc in ["FUTIMEBCYEARS", "FUYEARS", "FOLLOWUPYEARS"]:
            rename_map[c] = "fu_time_bc_years"
        elif nc in ["STATUSBC", "BCSTATUS", "EVENTBC"]:
            rename_map[c] = "status_bc"
        elif nc in ["NO22010", "NITROGENDIOXIDEAIRPOLLUTION2010"]:
            rename_map[c] = "no2_2010"
        elif nc in ["NOX2010", "NITROGENOXIDESAIRPOLLUTION2010"]:
            rename_map[c] = "nox_2010"
        elif nc in ["PM252010", "PARTICULATEMATTERAIRPOLLUTIONPM252010"]:
            rename_map[c] = "pm25_2010"
        elif nc in ["PM25ABSORB2010", "PM25ABSORBANCE2010", "PARTICULATEMATTERAIRPOLLUTIONPM25ABSORBANCE2010"]:
            rename_map[c] = "pm25_absorb_2010"
        elif nc in ["PM102010", "PARTICULATEMATTERAIRPOLLUTIONPM102010"]:
            rename_map[c] = "pm10_2010"
        elif nc in ["PMCOARSE2010", "PM2510UM2010", "PARTICULATEMATTERAIRPOLLUTION2510UM2010"]:
            rename_map[c] = "pmcoarse_2010"
    if rename_map:
        out = out.rename(columns=rename_map)

    need_numeric = [
        "eid", "fu_time_bc_years", "status_bc", "age", "bmi", "education",
        "alcohol_freq", "urban_rural",
        "no2_2010", "nox_2010", "pm25_2010", "pm25_absorb_2010", "pm10_2010", "pmcoarse_2010",
    ]
    for c in need_numeric:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    if "sex" in out.columns:
        out["sex"] = out["sex"].astype(str).replace({"0": "Female", "1": "Male"})
        out.loc[out["sex"].isin(["nan", "None"]), "sex"] = np.nan
        out["sex"] = out["sex"].astype("category")

    if "smoking_status" in out.columns:
        out["smoking_status"] = out["smoking_status"].astype(str)
        out["smoking_status"] = out["smoking_status"].replace({
            "0": "Never", "1": "Former", "2": "Current"
        })
        out.loc[out["smoking_status"].isin(["nan", "None"]), "smoking_status"] = np.nan
        out["smoking_status"] = out["smoking_status"].astype("category")

    for c in ["ethnicity", "alcohol_freq", "urban_rural"]:
        if c in out.columns:
            out[c] = out[c].astype("category")

    # Construct the two pollution axes when they are not already present
    if "traffic_combustion_score" not in out.columns:
        components = []
        for c in ["no2_2010", "nox_2010", "pm25_absorb_2010"]:
            if c in out.columns:
                components.append(zscore(out[c]))
        if components:
            out["traffic_combustion_score"] = pd.concat(components, axis=1).mean(axis=1, skipna=True)

    if "particle_score" not in out.columns:
        components = []
        for c in ["pm25_2010", "pm10_2010", "pmcoarse_2010"]:
            if c in out.columns:
                components.append(zscore(out[c]))
        if components:
            out["particle_score"] = pd.concat(components, axis=1).mean(axis=1, skipna=True)

    keep_required = [
        "eid", "status_bc", "fu_time_bc_years", "age", "sex", "smoking_status", "bmi",
        "education", "alcohol_freq", "urban_rural", "ethnicity", "traffic_combustion_score", "particle_score",
        "no2_2010", "nox_2010", "pm25_2010", "pm25_absorb_2010", "pm10_2010", "pmcoarse_2010",
    ]
    keep_cols = [c for c in keep_required if c in out.columns]
    out = out.loc[:, keep_cols].copy()
    if "eid" not in out.columns:
        raise ValueError(
            "No eid column was detected in the current air pollution table. If this is an exposure-only export, ensure that the UK Biobank source zip is also available; otherwise review the ID column name."
        )
    out = out.loc[out["eid"].notna(), :].copy()
    out["eid"] = pd.to_numeric(out["eid"], errors="coerce")
    out = out.loc[out["eid"].notna(), :].copy()
    out = out.loc[out["fu_time_bc_years"].notna() & (out["fu_time_bc_years"] > 0), :].copy()
    return out


def build_air_from_ukb_raw(df_raw: pd.DataFrame) -> pd.DataFrame:
    raw_cols = list(df_raw.columns)
    cancer_date_cols = find_cols_by_prefix(raw_cols, CANCER_DATE_PREFIX)
    cancer_code_cols = find_cols_by_prefix(raw_cols, CANCER_CODE_PREFIX)
    cancer_beh_cols = find_cols_by_prefix(raw_cols, CANCER_BEHAV_PREFIX)

    all_exposure_map = {}
    all_exposure_map.update(PRIMARY_EXPOSURES)
    all_exposure_map.update(SECONDARY_EXPOSURES)

    needed_cols = []
    needed_cols += [v for v in BASELINE_COLS.values() if v in raw_cols]
    needed_cols += [v for v in all_exposure_map.values() if v in raw_cols]
    needed_cols += [c for c in DEATH_DATE_COLS if c in raw_cols]
    needed_cols += cancer_date_cols + cancer_code_cols + cancer_beh_cols
    needed_cols = list(dict.fromkeys(needed_cols))

    df_raw = df_raw.loc[:, needed_cols].copy()
    rename_map = {v: k for k, v in BASELINE_COLS.items() if v in df_raw.columns}
    for new_name, old_name in all_exposure_map.items():
        if old_name in df_raw.columns:
            rename_map[old_name] = new_name
    df = df_raw.rename(columns=rename_map).copy()

    numeric_cols = [
        "eid", "address_years", "age", "sex", "ethnicity", "smoking_status",
        "alcohol_freq", "urban_rural", "bmi",
    ] + list(all_exposure_map.keys())
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

    bc_dates = []
    death_dates = []
    prevalent_bca = []
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
        bc_dt = min(bc_vals) if bc_vals else pd.NaT
        bc_dates.append(bc_dt)

        dvals = []
        for c in DEATH_DATE_COLS:
            if c in df.columns:
                dt = row.get(c, pd.NaT)
                if pd.notna(dt):
                    dvals.append(dt)
        death_dt = min(dvals) if dvals else pd.NaT
        death_dates.append(death_dt)

        if pd.notna(bc_dt) and pd.notna(row.get("baseline_date", pd.NaT)) and bc_dt < row.get("baseline_date", pd.NaT):
            prevalent_bca.append(1)
        else:
            prevalent_bca.append(0)

    df["bc_date"] = bc_dates
    df["death_date"] = death_dates
    df["prevalent_bca"] = prevalent_bca

    if "sex" in df.columns:
        df["sex"] = df["sex"].map(map_sex).astype("category")
    if "smoking_status" in df.columns:
        df["smoking_status"] = df["smoking_status"].map(map_smoking).astype("category")
    for c in ["ethnicity", "alcohol_freq", "urban_rural"]:
        if c in df.columns:
            df[c] = df[c].astype("category")

    traffic_components = []
    for c in ["no2_2010", "nox_2010", "pm25_absorb_2010"]:
        if c in df.columns:
            traffic_components.append(zscore(df[c]))
    if traffic_components:
        df["traffic_combustion_score"] = pd.concat(traffic_components, axis=1).mean(axis=1, skipna=True)

    particle_components = []
    for c in ["pm25_2010", "pm10_2010", "pmcoarse_2010"]:
        if c in df.columns:
            particle_components.append(zscore(df[c]))
    if particle_components:
        df["particle_score"] = pd.concat(particle_components, axis=1).mean(axis=1, skipna=True)

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
        bc_event = int(pd.notna(row["bc_date"]) and end_date == row["bc_date"] and row["bc_date"] >= baseline_date)
        status_bc.append(bc_event)

    df["end_of_followup_bc"] = end_dates
    df["status_bc"] = status_bc
    df["fu_time_bc_years"] = (df["end_of_followup_bc"] - df["baseline_date"]).dt.days / 365.25

    keep = (
        df["eid"].notna() &
        df["baseline_date"].notna() &
        (df["prevalent_bca"] == 0) &
        df["fu_time_bc_years"].notna() &
        (df["fu_time_bc_years"] > 0)
    )
    out = df.loc[keep, [
        c for c in [
            "eid", "status_bc", "fu_time_bc_years", "age", "sex", "smoking_status", "bmi", "education",
            "alcohol_freq", "urban_rural", "ethnicity", "traffic_combustion_score", "particle_score",
            "no2_2010", "nox_2010", "pm25_2010", "pm25_absorb_2010", "pm10_2010", "pmcoarse_2010",
            "address_years"
        ] if c in df.columns
    ]].copy()
    return out


def build_air_analytic_table(df_air_raw: pd.DataFrame, air_path: Optional[str] = None) -> pd.DataFrame:
    if is_ukb_raw_table(df_air_raw):
        log("[INFO] Detected a UK Biobank source table; building the analytic cohort table directly.")
        out = build_air_from_ukb_raw(df_air_raw)
    elif is_air_exposure_only_table(df_air_raw):
        log("[INFO] Detected an exposure-only air pollution table; merging it with the cohort table built from the UK Biobank source zip.")
        exp_only = standardize_exposure_only_air_table(df_air_raw)
        df_ukb_raw = load_ukb_zip_raw()
        cohort_base = build_air_from_ukb_raw(df_ukb_raw)
        drop_exp = [c for c in ["no2_2010", "nox_2010", "pm25_2010", "pm25_absorb_2010", "pm10_2010", "pmcoarse_2010", "traffic_combustion_score", "particle_score"] if c in cohort_base.columns]
        cohort_base = cohort_base.drop(columns=drop_exp, errors="ignore")
        out = cohort_base.merge(exp_only, on="eid", how="left")
        if all(c in out.columns for c in ["no2_2010", "nox_2010", "pm25_absorb_2010"]):
            out["traffic_combustion_score"] = pd.concat([zscore(out["no2_2010"]), zscore(out["nox_2010"]), zscore(out["pm25_absorb_2010"])], axis=1).mean(axis=1, skipna=True)
        if all(c in out.columns for c in ["pm25_2010", "pm10_2010", "pmcoarse_2010"]):
            out["particle_score"] = pd.concat([zscore(out["pm25_2010"]), zscore(out["pm10_2010"]), zscore(out["pmcoarse_2010"])], axis=1).mean(axis=1, skipna=True)
    else:
        log("[INFO] Detected a pre-assembled air pollution / cohort table; applying standard column cleaning.")
        out = standardize_clean_air_table(df_air_raw)

    if "eid" not in out.columns:
        raise ValueError("The analytic air table was built without an eid column. Review the input table structure.")
    out["eid"] = pd.to_numeric(out["eid"], errors="coerce")
    out = out.loc[out["eid"].notna(), :].copy()
    out = out.drop_duplicates(subset=["eid"], keep="first")
    return out


# =========================================================
# 4. Read and standardise the proteomics table
# =========================================================
def load_proteomics_raw() -> Tuple[pd.DataFrame, str]:
    path = first_existing_path(PROTEOMICS_RAW_CANDIDATES)
    if path is None:
        raise FileNotFoundError("Proteomics / Olink input not found.")
    df = safe_read_table(path)
    log(f"[INFO] Proteomics input: {path} | shape={df.shape}")
    return df, path


def detect_technical_and_protein_cols(df: pd.DataFrame, eid_col: str, field_whitelist: Optional[set] = None) -> Tuple[List[str], List[str], pd.DataFrame]:
    tech_cols = []
    protein_cols = []
    candidate_info = []
    whitelist = set(field_whitelist or set())

    for c in df.columns:
        if c == eid_col:
            candidate_info.append({"column": c, "role": "id", "reason": "eid"})
            continue

        s = df[c]
        lc = str(c).lower()
        is_numeric = pd.api.types.is_numeric_dtype(s)
        n_nonmissing = int(s.notna().sum())
        nunique = int(s.nunique(dropna=True))
        norm_c = normalize_protein_symbol(c)

        if c in MANUAL_TECH_COLS:
            tech_cols.append(c)
            candidate_info.append({"column": c, "role": "technical", "reason": "manual_technical"})
            continue

        if c in MANUAL_PROTEIN_COLS:
            protein_cols.append(c)
            candidate_info.append({"column": c, "role": "protein", "reason": "manual_protein"})
            continue

        if whitelist:
            if norm_c in whitelist and norm_c != "eid" and is_numeric and n_nonmissing >= 30 and nunique >= 20:
                protein_cols.append(c)
                candidate_info.append({"column": c, "role": "protein", "reason": "field_name_whitelist_exact"})
                continue

        if any(k in lc for k in TECH_KEYWORDS):
            tech_cols.append(c)
            candidate_info.append({"column": c, "role": "technical", "reason": "tech_keyword"})
            continue

        if (not is_numeric) and nunique <= 100:
            if nunique > 1:
                tech_cols.append(c)
                candidate_info.append({"column": c, "role": "technical", "reason": "low_cardinality_text"})
            else:
                candidate_info.append({"column": c, "role": "drop", "reason": "constant_text"})
            continue

        if is_numeric and n_nonmissing >= 30 and nunique >= 20:
            if whitelist:
                candidate_info.append({"column": c, "role": "drop", "reason": "numeric_not_in_whitelist"})
            else:
                protein_cols.append(c)
                candidate_info.append({"column": c, "role": "protein", "reason": "numeric_candidate_no_whitelist"})
            continue

        candidate_info.append({"column": c, "role": "drop", "reason": "other"})

    protein_cols = list(dict.fromkeys(protein_cols))
    tech_cols = [c for c in dict.fromkeys(tech_cols) if c != eid_col and c not in protein_cols]
    info_df = pd.DataFrame(candidate_info)
    return tech_cols, protein_cols, info_df


def prepare_proteomics_table(
    df_raw: pd.DataFrame,
    field_whitelist: Optional[set] = None,
) -> Tuple[pd.DataFrame, List[str], List[str], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    eid_col = detect_eid_col(df_raw)
    if eid_col is None:
        raise ValueError("No eid column was detected in the proteomics table. Set MANUAL_PROTEOMICS_EID_COL if automatic detection fails.")

    df = df_raw.copy().rename(columns={eid_col: "eid"})
    df["eid"] = pd.to_numeric(df["eid"], errors="coerce")
    df = df.loc[df["eid"].notna(), :].copy()

    tech_cols, protein_cols, info_df = detect_technical_and_protein_cols(df, "eid", field_whitelist=field_whitelist)

    duplicate_before = int(df["eid"].duplicated(keep=False).sum())
    df_dedup = choose_best_duplicate_rows(df, protein_cols)
    duplicate_after = int(df_dedup["eid"].duplicated(keep=False).sum())

    for c in protein_cols:
        if c in df_dedup.columns:
            df_dedup[c] = pd.to_numeric(df_dedup[c], errors="coerce")

    protein_missing = pd.DataFrame({
        "protein_col": protein_cols,
        "n_nonmissing": [int(df_dedup[c].notna().sum()) for c in protein_cols],
        "missing_rate": [1 - float(df_dedup[c].notna().mean()) for c in protein_cols],
        "n_unique": [int(df_dedup[c].nunique(dropna=True)) for c in protein_cols],
    }).sort_values(["missing_rate", "protein_col"], ascending=[True, True]) if protein_cols else pd.DataFrame()

    qc_summary = pd.DataFrame([
        {"item": "raw_rows", "value": df_raw.shape[0]},
        {"item": "rows_with_eid", "value": df.shape[0]},
        {"item": "duplicate_rows_before_dedup", "value": duplicate_before},
        {"item": "rows_after_dedup", "value": df_dedup.shape[0]},
        {"item": "duplicate_rows_after_dedup", "value": duplicate_after},
        {"item": "n_technical_cols", "value": len(tech_cols)},
        {"item": "n_protein_cols", "value": len(protein_cols)},
    ])

    return df_dedup, tech_cols, protein_cols, info_df, protein_missing, qc_summary


# =========================================================
# 5. Merge, sample flow, and the analytic dataset
# =========================================================
def build_id_overlap_tables(df_air: pd.DataFrame, df_prot: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    air_ids = pd.Index(sorted(df_air["eid"].dropna().astype(int).unique().tolist()))
    prot_ids = pd.Index(sorted(df_prot["eid"].dropna().astype(int).unique().tolist()))
    overlap = air_ids.intersection(prot_ids)
    air_only = air_ids.difference(prot_ids)
    prot_only = prot_ids.difference(air_ids)

    summary = pd.DataFrame([
        {"dataset": "air_analytic", "n_unique_eid": len(air_ids)},
        {"dataset": "proteomics_clean", "n_unique_eid": len(prot_ids)},
        {"dataset": "overlap", "n_unique_eid": len(overlap)},
        {"dataset": "air_only", "n_unique_eid": len(air_only)},
        {"dataset": "proteomics_only", "n_unique_eid": len(prot_only)},
    ])

    overlap_df = pd.DataFrame({"eid": overlap})
    air_only_df = pd.DataFrame({"eid": air_only})
    prot_only_df = pd.DataFrame({"eid": prot_only})
    return summary, overlap_df, air_only_df, prot_only_df


def build_analytic_dataset(df_air: pd.DataFrame, df_prot: pd.DataFrame, protein_cols: List[str], tech_cols: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    merged = df_air.merge(df_prot, on="eid", how="inner", suffixes=("", "_prot"))
    merged["status_bc"] = pd.to_numeric(merged["status_bc"], errors="coerce")
    merged["fu_time_bc_years"] = pd.to_numeric(merged["fu_time_bc_years"], errors="coerce")

    complete_base_cols = [
        "eid", "status_bc", "fu_time_bc_years", "age", "sex", "smoking_status", "bmi",
        "education", "alcohol_freq", "urban_rural", "ethnicity", "traffic_combustion_score", "particle_score"
    ]
    complete_base_cols = [c for c in complete_base_cols if c in merged.columns]

    analytic = merged.copy()
    analytic = analytic.replace([np.inf, -np.inf], np.nan)
    analytic = analytic.loc[analytic["fu_time_bc_years"].notna() & (analytic["fu_time_bc_years"] > 0), :].copy()
    analytic = analytic.dropna(subset=complete_base_cols)

    # Exposure-specific complete-case handling is performed later. Here we only require the main exposures and model covariates.
    analytic = analytic.reset_index(drop=True)

    flow = pd.DataFrame([
        {"step": "air_analytic_rows", "n": df_air.shape[0], "events": int(pd.to_numeric(df_air["status_bc"], errors="coerce").fillna(0).sum())},
        {"step": "proteomics_clean_rows", "n": df_prot.shape[0], "events": np.nan},
        {"step": "inner_merge_rows", "n": merged.shape[0], "events": int(pd.to_numeric(merged["status_bc"], errors="coerce").fillna(0).sum())},
        {"step": "olink_analytic_set", "n": analytic.shape[0], "events": int(pd.to_numeric(analytic["status_bc"], errors="coerce").fillna(0).sum())},
        {"step": "n_protein_cols", "n": len(protein_cols), "events": np.nan},
        {"step": "n_technical_cols", "n": len(tech_cols), "events": np.nan},
    ])
    return analytic, flow


# =========================================================
# 6. Technical adjustment of protein measurements
# =========================================================
def split_tech_covariates(df: pd.DataFrame, tech_cols: List[str]) -> Tuple[List[str], List[str]]:
    cat_cols = []
    num_cols = []
    for c in tech_cols:
        if c not in df.columns:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            nunique = int(df[c].nunique(dropna=True))
            if nunique <= 12:
                cat_cols.append(c)
            else:
                num_cols.append(c)
        else:
            cat_cols.append(c)
    return cat_cols, num_cols


def build_tech_design_df(df: pd.DataFrame, y: str, tech_cat_cols: List[str], tech_num_cols: List[str]) -> pd.DataFrame:
    cols = [y] + [c for c in tech_cat_cols + tech_num_cols if c in df.columns]
    tmp = df.loc[:, cols].copy().replace([np.inf, -np.inf], np.nan)
    tmp = tmp.dropna(subset=[y])
    if tmp.empty:
        return pd.DataFrame()

    use_cat = [c for c in tech_cat_cols if c in tmp.columns]
    if use_cat:
        tmp = pd.get_dummies(tmp, columns=use_cat, drop_first=True, dtype=float)

    for c in tmp.columns:
        tmp[c] = pd.to_numeric(tmp[c], errors="coerce")

    # Rows with missing technical covariates are removed for the adjustment model
    tmp = tmp.replace([np.inf, -np.inf], np.nan).dropna()
    if tmp.empty:
        return pd.DataFrame()

    drop_cols = []
    for c in tmp.columns:
        if c == y:
            continue
        if tmp[c].nunique(dropna=True) <= 1:
            drop_cols.append(c)
    if drop_cols:
        tmp = tmp.drop(columns=drop_cols, errors="ignore")
    return tmp


def residualize_proteins(df: pd.DataFrame, protein_cols: List[str], tech_cols: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    out = pd.DataFrame(index=df.index)
    out["eid"] = df["eid"].values

    tech_cat_cols, tech_num_cols = split_tech_covariates(df, tech_cols)
    summary_rows = []

    for protein in protein_cols:
        raw = pd.to_numeric(df[protein], errors="coerce")
        adj_col = f"adj__{protein}"

        if raw.notna().sum() < 30:
            out[adj_col] = np.nan
            summary_rows.append({
                "protein_col": protein,
                "method": "skip_low_nonmissing",
                "n_nonmissing_raw": int(raw.notna().sum()),
                "n_nonmissing_adj": 0,
                "n_tech_terms": 0,
            })
            continue

        if (not STATS_AVAILABLE) or len(tech_cols) == 0:
            out[adj_col] = zscore(raw)
            summary_rows.append({
                "protein_col": protein,
                "method": "raw_zscore_no_tech_adjustment",
                "n_nonmissing_raw": int(raw.notna().sum()),
                "n_nonmissing_adj": int(out[adj_col].notna().sum()),
                "n_tech_terms": 0,
            })
            continue

        work = df[[protein] + [c for c in tech_cols if c in df.columns]].copy()
        work = work.rename(columns={protein: "_y"})
        model_df = build_tech_design_df(work, "_y", tech_cat_cols, tech_num_cols)

        if model_df.empty or model_df.shape[0] < 30 or model_df.shape[1] <= 1:
            out[adj_col] = zscore(raw)
            summary_rows.append({
                "protein_col": protein,
                "method": "fallback_raw_zscore",
                "n_nonmissing_raw": int(raw.notna().sum()),
                "n_nonmissing_adj": int(out[adj_col].notna().sum()),
                "n_tech_terms": max(model_df.shape[1] - 1, 0) if not model_df.empty else 0,
            })
            continue

        fit = fit_ols_hc3(model_df, "_y")
        if fit is None:
            out[adj_col] = zscore(raw)
            summary_rows.append({
                "protein_col": protein,
                "method": "fallback_raw_zscore_fit_failed",
                "n_nonmissing_raw": int(raw.notna().sum()),
                "n_nonmissing_adj": int(out[adj_col].notna().sum()),
                "n_tech_terms": max(model_df.shape[1] - 1, 0),
            })
            continue

        x = sm.add_constant(model_df.drop(columns=["_y"]), has_constant="add")
        pred = pd.Series(fit.predict(x), index=model_df.index)
        resid = model_df["_y"] - pred
        resid_z = zscore(resid)

        full = pd.Series(np.nan, index=df.index, dtype=float)
        full.loc[model_df.index] = resid_z.values
        out[adj_col] = full.values

        summary_rows.append({
            "protein_col": protein,
            "method": "tech_residual_zscore",
            "n_nonmissing_raw": int(raw.notna().sum()),
            "n_nonmissing_adj": int(full.notna().sum()),
            "n_tech_terms": max(model_df.shape[1] - 1, 0),
        })

    return out, pd.DataFrame(summary_rows)


# =========================================================
# 7. Protein-domain definition and construction
# =========================================================
def build_axis_definition(adjusted_protein_cols: List[str], field_whitelist: Optional[set] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build the predefined protein-domain definitions using exact matches only.
    A protein is included only when the requested gene symbol matches the raw
    proteomics column name exactly after case normalisation. If a field-name
    whitelist is provided, the same exact match must also appear in that list.
    """
    raw_cols = [c.replace("adj__", "", 1) for c in adjusted_protein_cols]
    lookup = exact_protein_lookup_map(raw_cols)
    whitelist = set(field_whitelist or set())

    rows = []
    summary_rows = []

    for axis_name, genes in PROTEIN_AXIS_RULES.items():
        matched = []
        missing = []
        for gene in genes:
            gene_key = normalize_protein_symbol(gene)

            if whitelist and gene_key not in whitelist:
                missing.append(gene)
                continue

            raw_hit = lookup.get(gene_key)
            if raw_hit is None:
                missing.append(gene)
                continue

            # Keep only exact raw-column matches for the requested gene symbol
            if normalize_protein_symbol(raw_hit) != gene_key:
                missing.append(gene)
                continue

            adj_hit = f"adj__{raw_hit}"
            if adj_hit not in adjusted_protein_cols:
                missing.append(gene)
                continue

            matched.append(gene)
            rows.append({
                "axis_name": axis_name,
                "mapped_gene": gene,
                "protein_col_adjusted": adj_hit,
                "protein_col_raw": raw_hit,
                "match_rule": "strict_exact_column_name",
            })

        summary_rows.append({
            "axis_name": axis_name,
            "n_requested": len(genes),
            "n_matched": len(matched),
            "n_missing": len(missing),
            "matched_genes": ";".join(matched),
            "missing_genes": ";".join(missing),
        })

    axis_def = pd.DataFrame(rows)
    if not axis_def.empty:
        axis_def = axis_def.drop_duplicates(subset=["axis_name", "protein_col_adjusted"]).reset_index(drop=True)
        axis_def = validate_axis_definition_strict(axis_def)
    axis_summary = pd.DataFrame(summary_rows)
    return axis_def, axis_summary

def construct_protein_axes(df: pd.DataFrame, axis_def: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    out = df.copy()
    summary_rows = []

    if axis_def.empty:
        return out, pd.DataFrame()

    for axis_name in sorted(axis_def["axis_name"].unique().tolist()):
        cols = axis_def.loc[axis_def["axis_name"] == axis_name, "protein_col_adjusted"].tolist()
        cols = [c for c in cols if c in out.columns]
        if len(cols) == 0:
            continue

        tmp = out[cols].apply(pd.to_numeric, errors="coerce")
        valid_cols = [c for c in cols if tmp[c].notna().sum() >= 30]
        if len(valid_cols) == 0:
            continue

        out[f"{axis_name}_mean_z"] = tmp[valid_cols].mean(axis=1, skipna=True)
        out[f"{axis_name}_score"] = out[f"{axis_name}_mean_z"]

        summary_rows.append({
            "axis_name": axis_name,
            "n_axis_proteins_total": len(cols),
            "n_axis_proteins_used": len(valid_cols),
            "proteins_used_raw": ";".join([c.replace("adj__", "", 1) for c in valid_cols]),
            "proteins_used_adjusted": ";".join(valid_cols),
            "main_score_col": f"{axis_name}_mean_z",
            "main_method": "mean_z",
            "n_nonmissing_score": int(out[f"{axis_name}_score"].notna().sum()),
        })

    return out, pd.DataFrame(summary_rows)


# =========================================================
# 8. Association analyses
# =========================================================
def run_pollution_to_axis(df: pd.DataFrame, exposures: List[str], axis_cols: List[str]) -> pd.DataFrame:
    rows = []
    covs = [c for c in MODEL_COVS if c in df.columns]
    cat_cols = get_cat_cols(covs)

    for axis_col in axis_cols:
        for exposure in exposures:
            if exposure not in df.columns:
                continue
            work = df.copy()
            work[exposure] = zscore(work[exposure])
            model_df = build_linear_df(work, axis_col, [exposure] + covs, cat_cols)
            if model_df.empty or model_df.shape[0] < MIN_ROWS_LINEAR:
                continue
            fit = fit_ols_hc3(model_df, axis_col)
            if fit is None or exposure not in fit.params.index:
                continue
            rows.append({
                "protein_axis": axis_col,
                "exposure": exposure,
                "beta": fit.params[exposure],
                "se": fit.bse[exposure],
                "t": fit.tvalues[exposure],
                "p": fit.pvalues[exposure],
                "n": model_df.shape[0],
                "covs_used": ";".join(covs),
                "analysis_group": "main_axis" if exposure in MAIN_EXPOSURES else "support_pollutant",
            })
    out = pd.DataFrame(rows)
    if not out.empty:
        out["fdr_p"] = out.groupby("exposure")["p"].transform(lambda s: pd.Series(bh_fdr(s.tolist()), index=s.index))
    return out


def run_pollution_to_single_protein(df: pd.DataFrame, exposures: List[str], axis_def: pd.DataFrame) -> pd.DataFrame:
    rows = []
    covs = [c for c in MODEL_COVS if c in df.columns]
    cat_cols = get_cat_cols(covs)

    if axis_def.empty:
        return pd.DataFrame()

    protein_axis_map = (
        axis_def.groupby("protein_col_adjusted")["axis_name"]
        .apply(lambda s: ";".join(sorted(set(s.tolist()))))
        .to_dict()
    )
    gene_map = (
        axis_def.groupby("protein_col_adjusted")["mapped_gene"]
        .apply(lambda s: ";".join(sorted(set(s.tolist()))))
        .to_dict()
    )

    protein_cols = sorted(axis_def["protein_col_adjusted"].unique().tolist())
    for protein in protein_cols:
        if protein not in df.columns:
            continue
        for exposure in exposures:
            if exposure not in df.columns:
                continue
            work = df.copy()
            work[exposure] = zscore(work[exposure])
            model_df = build_linear_df(work, protein, [exposure] + covs, cat_cols)
            if model_df.empty or model_df.shape[0] < MIN_ROWS_LINEAR:
                continue
            fit = fit_ols_hc3(model_df, protein)
            if fit is None or exposure not in fit.params.index:
                continue
            rows.append({
                "protein_col_adjusted": protein,
                "protein_col_raw": protein.replace("adj__", "", 1),
                "protein_axis": protein_axis_map.get(protein, ""),
                "mapped_gene": gene_map.get(protein, protein.replace("adj__", "", 1)),
                "exposure": exposure,
                "beta": fit.params[exposure],
                "se": fit.bse[exposure],
                "t": fit.tvalues[exposure],
                "p": fit.pvalues[exposure],
                "n": model_df.shape[0],
                "covs_used": ";".join(covs),
                "analysis_group": "main_axis" if exposure in MAIN_EXPOSURES else "support_pollutant",
            })
    out = pd.DataFrame(rows)
    if not out.empty:
        out["fdr_p"] = out.groupby("exposure")["p"].transform(lambda s: pd.Series(bh_fdr(s.tolist()), index=s.index))
        out = enforce_validated_mapping_on_table(out, axis_def, "T7_pollution_to_single_protein")
    return out


def run_axis_to_bca(df: pd.DataFrame, axis_cols: List[str]) -> pd.DataFrame:
    rows = []
    covs = [c for c in MODEL_COVS if c in df.columns]
    cat_cols = get_cat_cols(covs)

    for axis_col in axis_cols:
        work = df.copy()
        work[axis_col] = zscore(work[axis_col])
        model_df = build_cox_df(work, axis_col, covs, cat_cols)
        if model_df.empty or model_df.shape[0] < MIN_ROWS_COX or model_df["status_bc"].sum() < MIN_EVENTS_COX:
            continue
        try:
            cph = fit_cox(model_df)
        except Exception as e:
            log(f"[WARN] axis->BCa model failed | axis={axis_col} | {e}")
            continue
        rows.append(extract_cox_term(cph, axis_col, {
            "protein_axis": axis_col,
            "analysis": "protein_axis_to_bca",
            "n": model_df.shape[0],
            "events": int(model_df["status_bc"].sum()),
            "covs_used": ";".join(covs),
        }))
    out = pd.DataFrame(rows)
    if not out.empty:
        out["fdr_p"] = bh_fdr(out["p"].tolist())
    return out


def main(args: argparse.Namespace):
    log("=" * 80)
    log("UK Biobank Olink analysis pipeline")
    log(f"[INFO] Run directory: {RUN_DIR}")
    log("=" * 80)

    if not STATS_AVAILABLE:
        raise ImportError("statsmodels is required for this pipeline.")

    global AIR_RAW_CANDIDATES, PROTEOMICS_RAW_CANDIDATES, FIELD_NAME_CANDIDATES
    AIR_RAW_CANDIDATES = [args.air_input]
    PROTEOMICS_RAW_CANDIDATES = [args.proteomics]
    FIELD_NAME_CANDIDATES = [args.field_names] if args.field_names else []

    config = {
        "run_id": RUN_ID,
        "script_version": "public_maintext_v1",
        "air_raw_candidates": AIR_RAW_CANDIDATES,
        "proteomics_raw_candidates": PROTEOMICS_RAW_CANDIDATES,
        "field_name_candidates": FIELD_NAME_CANDIDATES,
        "result_root": RESULT_ROOT,
        "main_exposures": MAIN_EXPOSURES,
        "support_exposures": SUPPORT_EXPOSURES,
        "model_covs": MODEL_COVS,
        "protein_axis_rules": PROTEIN_AXIS_RULES,
        "default_cancer_censor_date": str(DEFAULT_CANCER_CENSOR_DATE.date()),
        "statsmodels_available": STATS_AVAILABLE,
        "manual_proteomics_eid_col": MANUAL_PROTEOMICS_EID_COL,
        "manual_tech_cols": MANUAL_TECH_COLS,
        "manual_protein_cols": MANUAL_PROTEIN_COLS,
    }
    with open(os.path.join(DIR_LOG, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    # A. Read inputs
    df_air_raw, air_path = load_air_raw()
    df_prot_raw, prot_path = load_proteomics_raw()
    field_whitelist, field_path = load_field_name_whitelist()
    log(f"[INFO] Field-name whitelist: {field_path} | n_fields={len(field_whitelist)}")

    # B. Build the air pollution / cohort analytic table
    df_air = build_air_analytic_table(df_air_raw, air_path)
    df_air.to_csv(os.path.join(DIR_DATA, "air_analytic_table.csv"), index=False)
    log(f"[INFO] Air analytic table: shape={df_air.shape}")

    # C. Standardise the proteomics table
    df_prot, tech_cols, protein_cols, prot_col_info, protein_missing, prot_qc = prepare_proteomics_table(df_prot_raw, field_whitelist=field_whitelist)
    df_prot.to_csv(os.path.join(DIR_DATA, "proteomics_deduplicated.csv"), index=False)
    prot_col_info.to_excel(os.path.join(DIR_QC, "Q1_proteomics_column_role_map.xlsx"), index=False)
    protein_missing.to_excel(os.path.join(DIR_QC, "Q2_protein_missingness.xlsx"), index=False)
    prot_qc.to_excel(os.path.join(DIR_QC, "T3_olink_qc_summary.xlsx"), index=False)
    log(f"[INFO] Proteomics table after deduplication: shape={df_prot.shape} | n_tech={len(tech_cols)} | n_protein={len(protein_cols)}")

    # C1. ID overlap
    id_summary, overlap_ids, air_only_ids, prot_only_ids = build_id_overlap_tables(df_air, df_prot)
    id_summary.to_excel(os.path.join(DIR_QC, "T2_id_overlap_qc.xlsx"), index=False)
    with pd.ExcelWriter(os.path.join(DIR_QC, "Q3_id_lists.xlsx"), engine="openpyxl") as writer:
        overlap_ids.to_excel(writer, sheet_name="overlap_ids", index=False)
        air_only_ids.to_excel(writer, sheet_name="air_only_ids", index=False)
        prot_only_ids.to_excel(writer, sheet_name="proteomics_only_ids", index=False)

    # D. Build the analytic dataset
    analytic, sample_flow = build_analytic_dataset(df_air, df_prot, protein_cols, tech_cols)
    sample_flow.to_excel(os.path.join(DIR_MAIN, "T1_sample_flow.xlsx"), index=False)
    analytic.to_csv(os.path.join(DIR_DATA, "olink_analytic_dataset_before_adjustment.csv"), index=False)
    log(f"[INFO] Olink analytic set: shape={analytic.shape} | events={int(pd.to_numeric(analytic['status_bc'], errors='coerce').fillna(0).sum())}")

    # E. Technical adjustment
    adjusted_proteins, adjustment_summary = residualize_proteins(analytic, protein_cols, tech_cols)
    adjustment_summary.to_excel(os.path.join(DIR_QC, "Q4_protein_adjustment_summary.xlsx"), index=False)

    analytic2 = analytic.merge(adjusted_proteins, on="eid", how="left")
    analytic2.to_csv(os.path.join(DIR_DATA, "olink_analytic_dataset_after_adjustment.csv"), index=False)

    # F. Build the predefined proteomic response domains
    adjusted_cols = [c for c in analytic2.columns if c.startswith("adj__")]
    axis_def, axis_definition_summary = build_axis_definition(adjusted_cols, field_whitelist)
    axis_def.to_excel(os.path.join(DIR_AXIS, "T4_protein_axis_definition.xlsx"), index=False)
    axis_def[["axis_name","mapped_gene","protein_col_raw","protein_col_adjusted","match_rule","strict_validation_pass"]].to_excel(os.path.join(DIR_AXIS, "Q4_axis_definition_validation.xlsx"), index=False)

    analytic3, axis_summary = construct_protein_axes(analytic2, axis_def)
    if not axis_summary.empty:
        axis_summary = axis_summary.merge(axis_definition_summary, on="axis_name", how="left")
    else:
        axis_summary = axis_definition_summary.copy()
    axis_summary.to_excel(os.path.join(DIR_AXIS, "T5_protein_axis_summary.xlsx"), index=False)
    analytic3.to_csv(os.path.join(DIR_DATA, "olink_analytic_dataset_with_axes.csv"), index=False)

    axis_cols = [c for c in analytic3.columns if c.endswith("_score") and c not in MAIN_EXPOSURES]

    # G. Association analyses
    t6 = run_pollution_to_axis(analytic3, [c for c in MAIN_EXPOSURES if c in analytic3.columns], axis_cols)
    t6.to_excel(os.path.join(DIR_ASSOC, "T6_pollution_to_protein_axis.xlsx"), index=False)

    t7 = run_pollution_to_single_protein(analytic3, [c for c in MAIN_EXPOSURES if c in analytic3.columns], axis_def)
    t7.to_excel(os.path.join(DIR_ASSOC, "T7_pollution_to_single_protein.xlsx"), index=False)

    t8 = run_axis_to_bca(analytic3, axis_cols)
    t8.to_excel(os.path.join(DIR_ASSOC, "T8_protein_axis_to_BCa.xlsx"), index=False)

    log("[DONE] Pipeline finished")
    log(RUN_DIR)


if __name__ == "__main__":
    cli_args = parse_args()
    initialize_run_directories(cli_args.result_root)
    main(cli_args)
