#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Public PERM analysis script for the UK Biobank air pollution–bladder cancer project.

This script is intentionally limited to the main-text attenuation analysis.
It starts from an Olink analytic dataset that already contains:
    - follow-up time and bladder cancer event indicator
    - air-pollution axes
    - proteomic domain scores
    - model covariates

Included analyses:
    - 2 exposure axes: traffic_combustion_score, particle_score
    - 3 analytic subsets: main, address5, address5_lag5
    - 6 protein-domain blocks: 5 single domains + joint 5-domain block

Not included:
    - mediation analyses
    - inflammation / clinical / smoking add-on blocks
    - automatic file discovery under local project folders
    - figure generation
"""

from __future__ import annotations

import argparse
import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter


DEFAULT_EXPOSURES = ["traffic_combustion_score", "particle_score"]
PROTEIN_BLOCKS: Dict[str, List[str]] = {
    "protein_epithelial": ["epithelial_remodeling_score"],
    "protein_metabolism": ["metabolism_stress_score"],
    "protein_inflammatory": ["inflammatory_innate_score"],
    "protein_endothelial": ["endothelial_adhesion_score"],
    "protein_coagulation": ["coagulation_kallikrein_score"],
    "protein_all_5_axes": [
        "epithelial_remodeling_score",
        "metabolism_stress_score",
        "inflammatory_innate_score",
        "endothelial_adhesion_score",
        "coagulation_kallikrein_score",
    ],
}

SPECIAL_MISSING_VALUES = {
    -1, -3, -7, -10, -11, -13, -17, -21, -23, -27, -121, -313, -818
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run public PERM analyses from an Olink analytic dataset.")
    parser.add_argument(
        "--input",
        required=True,
        help="Path to olink_analytic_dataset_with_axes (.csv, .xlsx, .xls, or .parquet).",
    )
    parser.add_argument(
        "--result-root",
        default="results_perm_public",
        help="Directory used to store outputs.",
    )
    parser.add_argument(
        "--min-n",
        type=int,
        default=500,
        help="Minimum complete-case sample size required for a model pair.",
    )
    parser.add_argument(
        "--min-events",
        type=int,
        default=20,
        help="Minimum number of events required for a model pair.",
    )
    return parser.parse_args()


def safe_read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        try:
            return pd.read_csv(path, low_memory=False)
        except UnicodeDecodeError:
            return pd.read_csv(path, low_memory=False, encoding="gb18030")
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported input format: {path}")


def sanitize(name: str) -> str:
    name = str(name).strip().lower()
    name = re.sub(r"[^0-9a-zA-Z_]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    if re.match(r"^[0-9]", name):
        name = f"v_{name}"
    return name


def safe_numeric(series: pd.Series) -> pd.Series:
    out = pd.to_numeric(series, errors="coerce")
    return out.mask(out.isin(SPECIAL_MISSING_VALUES))


def zscore(series: pd.Series) -> pd.Series:
    s = safe_numeric(series)
    sd = s.std(skipna=True)
    if pd.isna(sd) or sd == 0:
        return pd.Series(np.nan, index=s.index)
    return (s - s.mean(skipna=True)) / sd


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


class Logger:
    def __init__(self, log_file: Path):
        self.log_file = log_file

    def write(self, message: str) -> None:
        print(message, flush=True)
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(message + "\n")


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [sanitize(c) for c in out.columns]

    alias_map = {
        "participant_eid": "eid",
        "event_bc": "status_bc",
        "bc_status": "status_bc",
        "followup_years": "fu_time_bc_years",
        "fu_years": "fu_time_bc_years",
        "years_at_current_address": "address_years",
    }
    rename_map = {c: alias_map[c] for c in out.columns if c in alias_map}
    if rename_map:
        out = out.rename(columns=rename_map)

    required = ["eid", "status_bc", "fu_time_bc_years"]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    numeric_candidates = {
        "eid", "status_bc", "fu_time_bc_years", "address_years", "age", "bmi", "education",
        "traffic_combustion_score", "particle_score", "epithelial_remodeling_score",
        "metabolism_stress_score", "inflammatory_innate_score", "endothelial_adhesion_score",
        "coagulation_kallikrein_score", "white_british",
    }
    for col in numeric_candidates:
        if col in out.columns:
            out[col] = safe_numeric(out[col])

    categorical_candidates = ["sex", "smoking_status", "alcohol_freq", "urban_rural", "ethnicity"]
    for col in categorical_candidates:
        if col in out.columns:
            out[col] = out[col].astype("string")
            out.loc[out[col].isin(["", "nan", "NA", "None"]), col] = pd.NA

    out = out.loc[out["eid"].notna()].copy()
    out["eid"] = safe_numeric(out["eid"]).astype("Int64")
    out = out.loc[out["fu_time_bc_years"].notna() & (out["fu_time_bc_years"] > 0)].copy()
    return out


def choose_base_covariates(df: pd.DataFrame) -> List[str]:
    preferred = [
        "age", "sex", "smoking_status", "bmi", "education", "alcohol_freq", "urban_rural", "ethnicity"
    ]
    if all(col in df.columns for col in preferred):
        return preferred

    fallback = [
        "age", "sex", "smoking_status", "bmi", "education", "alcohol_freq", "urban_rural", "white_british"
    ]
    if all(col in df.columns for col in fallback):
        return fallback

    available = [c for c in preferred if c in df.columns]
    missing = [c for c in preferred if c not in df.columns]
    if missing:
        # Fallback to whatever is present, but keep the choice explicit in outputs.
        return available
    return preferred


def add_standardized_variables(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for exposure in DEFAULT_EXPOSURES:
        if exposure in out.columns:
            out[f"{exposure}_z"] = zscore(out[exposure])
    for block_vars in PROTEIN_BLOCKS.values():
        for var in block_vars:
            if var in out.columns:
                out[f"{var}_z"] = zscore(out[var])
    return out


def build_subsets(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    subsets = {"main": df.copy()}

    if "address_years" in df.columns:
        address5 = df.loc[df["address_years"].fillna(-999) >= 5].copy()
    else:
        address5 = df.copy()
    subsets["address5"] = address5

    address5_lag5 = address5.copy()
    keep = (address5_lag5["status_bc"] != 1) | (address5_lag5["fu_time_bc_years"] > 5)
    subsets["address5_lag5"] = address5_lag5.loc[keep].copy()
    return subsets


def categorical_covariates(covariates: List[str]) -> List[str]:
    return [c for c in covariates if c in {"sex", "smoking_status", "alcohol_freq", "urban_rural", "ethnicity"}]


def prepare_design_matrix(
    df: pd.DataFrame,
    exposure_z: str,
    base_covariates: List[str],
    block_vars: List[str],
) -> Tuple[pd.DataFrame, List[str], List[str], List[str]]:
    used_base = [c for c in base_covariates if c in df.columns]

    used_block = []
    for var in block_vars:
        if f"{var}_z" in df.columns:
            used_block.append(f"{var}_z")
        elif var in df.columns:
            used_block.append(var)

    needed = ["fu_time_bc_years", "status_bc", exposure_z] + used_base + used_block
    needed = [c for c in needed if c in df.columns]
    work = df[needed].copy()

    cat_cols = categorical_covariates(used_base)
    for col in cat_cols:
        if col in work.columns:
            work[col] = work[col].astype("string")

    non_cat = [c for c in work.columns if c not in cat_cols]
    for col in non_cat:
        work[col] = safe_numeric(work[col])

    work = work.dropna(axis=0, how="any").copy()
    if work.empty:
        return pd.DataFrame(), used_base, used_block, []

    ycols = ["fu_time_bc_years", "status_bc"]
    xcols = [c for c in work.columns if c not in ycols]
    X = pd.get_dummies(work[xcols], columns=[c for c in cat_cols if c in xcols], drop_first=True, dtype=float)
    X.columns = [sanitize(c) for c in X.columns]

    design = pd.concat([work[ycols].reset_index(drop=True), X.reset_index(drop=True)], axis=1)
    exposure_clean = sanitize(exposure_z)
    block_clean = [sanitize(c) for c in used_block]
    return design, used_base, used_block, [exposure_clean] + block_clean


class CoxResult:
    def __init__(self, hr: float, lci: float, uci: float, p: float, coef: float, n: int, events: int):
        self.hr = hr
        self.lci = lci
        self.uci = uci
        self.p = p
        self.coef = coef
        self.n = n
        self.events = events


def fit_cox_extract(df_design: pd.DataFrame, exposure_z: str) -> CoxResult:
    exposure_clean = sanitize(exposure_z)

    try:
        cph = CoxPHFitter()
        cph.fit(df_design, duration_col="fu_time_bc_years", event_col="status_bc", show_progress=False)
    except Exception:
        cph = CoxPHFitter(penalizer=1e-6)
        cph.fit(df_design, duration_col="fu_time_bc_years", event_col="status_bc", show_progress=False)

    row = cph.summary.loc[exposure_clean]
    return CoxResult(
        hr=float(math.exp(row["coef"])),
        lci=float(math.exp(row["coef lower 95%"])),
        uci=float(math.exp(row["coef upper 95%"])),
        p=float(row["p"]),
        coef=float(row["coef"]),
        n=int(df_design.shape[0]),
        events=int(df_design["status_bc"].sum()),
    )


def calc_perm(hr_base: float, hr_adj: float) -> float:
    denom = hr_base - 1.0
    if abs(denom) < 1e-12:
        return np.nan
    return (hr_base - hr_adj) / denom * 100.0


def calc_perm_log(hr_base: float, hr_adj: float) -> float:
    if hr_base <= 0 or hr_adj <= 0:
        return np.nan
    denom = math.log(hr_base)
    if abs(denom) < 1e-12:
        return np.nan
    return (math.log(hr_base) - math.log(hr_adj)) / denom * 100.0


def run_perm(
    df: pd.DataFrame,
    subsets: Dict[str, pd.DataFrame],
    base_covariates: List[str],
    min_n: int,
    min_events: int,
    logger: Logger,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    results = []
    qc_rows = []
    failures = []

    for subset_name, subset_df in subsets.items():
        for exposure in DEFAULT_EXPOSURES:
            exposure_z = f"{exposure}_z"
            if exposure_z not in subset_df.columns:
                failures.append({
                    "subset": subset_name,
                    "exposure": exposure,
                    "block_name": "__all__",
                    "reason": f"missing {exposure_z}",
                })
                continue

            for block_name, block_vars in PROTEIN_BLOCKS.items():
                try:
                    design_adj, used_base, used_block, protected = prepare_design_matrix(
                        subset_df, exposure_z, base_covariates, block_vars
                    )
                    if design_adj.empty:
                        qc_rows.append({
                            "subset": subset_name,
                            "exposure": exposure,
                            "block_name": block_name,
                            "status": "empty_after_complete_case",
                        })
                        continue

                    protected_set = set(protected[1:])  # exclude exposure itself
                    ycols = ["fu_time_bc_years", "status_bc"]
                    xcols = [c for c in design_adj.columns if c not in ycols]
                    base_xcols = [c for c in xcols if c not in protected_set]
                    design_base = design_adj[ycols + base_xcols].copy()

                    n = int(design_adj.shape[0])
                    events = int(design_adj["status_bc"].sum())
                    if n < min_n or events < min_events:
                        qc_rows.append({
                            "subset": subset_name,
                            "exposure": exposure,
                            "block_name": block_name,
                            "status": f"insufficient_n_or_events_n{n}_e{events}",
                            "n": n,
                            "events": events,
                        })
                        continue

                    base_fit = fit_cox_extract(design_base, exposure_z)
                    adj_fit = fit_cox_extract(design_adj, exposure_z)

                    results.append({
                        "subset": subset_name,
                        "exposure": exposure,
                        "exposure_z": exposure_z,
                        "block_name": block_name,
                        "block_vars": ";".join(used_block),
                        "base_covariates_used": ";".join(used_base),
                        "n": base_fit.n,
                        "events": base_fit.events,
                        "hr_base": base_fit.hr,
                        "lci_base": base_fit.lci,
                        "uci_base": base_fit.uci,
                        "p_base": base_fit.p,
                        "coef_base": base_fit.coef,
                        "hr_adj": adj_fit.hr,
                        "lci_adj": adj_fit.lci,
                        "uci_adj": adj_fit.uci,
                        "p_adj": adj_fit.p,
                        "coef_adj": adj_fit.coef,
                        "perm": calc_perm(base_fit.hr, adj_fit.hr),
                        "perm_log": calc_perm_log(base_fit.hr, adj_fit.hr),
                        "delta_hr": base_fit.hr - adj_fit.hr,
                        "delta_coef": base_fit.coef - adj_fit.coef,
                    })

                    qc_rows.append({
                        "subset": subset_name,
                        "exposure": exposure,
                        "block_name": block_name,
                        "status": "ok",
                        "n": n,
                        "events": events,
                        "base_covariates_used": ";".join(used_base),
                        "block_vars": ";".join(used_block),
                    })
                except Exception as exc:
                    logger.write(
                        f"[WARN] subset={subset_name} | exposure={exposure} | block={block_name} | {exc}"
                    )
                    failures.append({
                        "subset": subset_name,
                        "exposure": exposure,
                        "block_name": block_name,
                        "reason": str(exc),
                    })

    result_df = pd.DataFrame(results)
    qc_df = pd.DataFrame(qc_rows)
    failure_df = pd.DataFrame(failures)
    return result_df, qc_df, failure_df


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    result_root = Path(args.result_root)
    run_id = datetime.now().strftime("RUN_%Y%m%d_%H%M%S")
    run_dir = result_root / run_id
    log_dir = run_dir / "01_logs"
    data_dir = run_dir / "02_data"
    qc_dir = run_dir / "03_qc"
    res_dir = run_dir / "04_results"

    for directory in [log_dir, data_dir, qc_dir, res_dir]:
        ensure_dir(directory)

    logger = Logger(log_dir / "run_log.txt")
    logger.write("=" * 88)
    logger.write("PERM public analysis")
    logger.write(f"[INFO] input={input_path}")
    logger.write(f"[INFO] run_dir={run_dir}")
    logger.write("=" * 88)

    df = safe_read_table(input_path)
    df = standardize_columns(df)
    df = add_standardized_variables(df)

    base_covariates = choose_base_covariates(df)
    logger.write(f"[INFO] base covariates: {', '.join(base_covariates)}")

    preview_cols = [
        c for c in [
            "eid", "status_bc", "fu_time_bc_years", "address_years", *base_covariates,
            *DEFAULT_EXPOSURES, *list({v for block in PROTEIN_BLOCKS.values() for v in block})
        ] if c in df.columns
    ]
    df[preview_cols].head(100).to_csv(data_dir / "dataset_preview_head100.csv", index=False, encoding="utf-8-sig")

    availability_rows = []
    for col in preview_cols:
        availability_rows.append({
            "variable": col,
            "n_nonmissing": int(df[col].notna().sum()),
            "missing_rate": float(1 - df[col].notna().mean()),
        })
    pd.DataFrame(availability_rows).to_csv(qc_dir / "variable_availability.csv", index=False, encoding="utf-8-sig")

    subsets = build_subsets(df)
    subset_counts = []
    for name, dsub in subsets.items():
        subset_counts.append({
            "subset": name,
            "n": int(dsub.shape[0]),
            "events": int(pd.to_numeric(dsub["status_bc"], errors="coerce").fillna(0).sum()),
        })
    pd.DataFrame(subset_counts).to_csv(qc_dir / "subset_counts.csv", index=False, encoding="utf-8-sig")

    result_df, qc_df, failure_df = run_perm(
        df=df,
        subsets=subsets,
        base_covariates=base_covariates,
        min_n=args.min_n,
        min_events=args.min_events,
        logger=logger,
    )

    if not result_df.empty:
        result_df = result_df.sort_values(["exposure", "subset", "block_name"]).reset_index(drop=True)
        result_df.to_csv(res_dir / "perm_results_all.csv", index=False, encoding="utf-8-sig")

        ranked = result_df.copy()
        ranked["perm_rank_within_exposure_subset"] = ranked.groupby(["exposure", "subset"])["perm"].rank(
            ascending=False,
            method="dense",
        )
        ranked.to_csv(res_dir / "perm_ranked.csv", index=False, encoding="utf-8-sig")

        plot_df = ranked[["subset", "exposure", "block_name", "perm", "perm_log", "n", "events"]].copy()
        plot_df.to_csv(res_dir / "perm_for_plot.csv", index=False, encoding="utf-8-sig")
    else:
        logger.write("[WARN] No valid PERM results were generated.")

    qc_df.to_csv(qc_dir / "perm_model_qc.csv", index=False, encoding="utf-8-sig")
    failure_df.to_csv(qc_dir / "perm_failures.csv", index=False, encoding="utf-8-sig")

    meta = {
        "script": "06_PERM_public.py",
        "run_id": run_id,
        "input": str(input_path),
        "run_dir": str(run_dir),
        "exposures": DEFAULT_EXPOSURES,
        "protein_blocks": PROTEIN_BLOCKS,
        "base_covariates": base_covariates,
        "min_n": args.min_n,
        "min_events": args.min_events,
    }
    with open(log_dir / "run_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    logger.write(f"[INFO] Finished. Results written to: {run_dir}")


if __name__ == "__main__":
    main()
