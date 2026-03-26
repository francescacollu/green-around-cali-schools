"""
Merge filtered public schools CSV with Santa Clara FRPM on normalized school_code.

Reads:  data/raw/public_schools_filtered.csv, data/raw/frpm2425_santaclara.csv
Writes: data/cleaned/public_schools_frpm_santaclara_merged.csv
         outputs/public_schools_frpm_santaclara_qa.txt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "cleaning"))
from frpm_io import FRPM_FULL_RENAME, normalize_frpm_columns, school_code_as_str

DEFAULT_PUBLIC = ROOT / "data" / "raw" / "public_schools_filtered.csv"
DEFAULT_FRPM = ROOT / "data" / "raw" / "frpm2425_santaclara.csv"
DEFAULT_MERGED = ROOT / "data" / "cleaned" / "public_schools_frpm_santaclara_merged.csv"
DEFAULT_QA = ROOT / "outputs" / "public_schools_frpm_santaclara_qa.txt"

NO_DATA = "No Data"

MERGED_LEADING_COLUMNS = [
    "school_code",
    "school_name",
    "school_name_frpm",
    "city",
    "latitude",
    "longitude",
    "is_charter",
    "charter_school_yn",
    "charter_raw",
]


def _reorder_merged_columns(df: pd.DataFrame) -> pd.DataFrame:
    first = [c for c in MERGED_LEADING_COLUMNS if c in df.columns]
    rest = [c for c in df.columns if c not in first]
    return df[first + rest]


PUBLIC_RENAME = {
    "CDSCode": "cds_code",
    "SchoolID": "school_id",
    "NCESDist": "nces_dist",
    "NCESSchool": "nces_school",
    "StatusType": "status_type",
    "County": "county",
    "District": "district",
    "School": "school_name",
    "Street": "street",
    "StreetAbr": "street_abr",
    "City": "city",
    "Zip": "zip",
    "State": "state",
    "MailStreet": "mail_street",
    "MailStrAbr": "mail_str_abr",
    "MailCity": "mail_city",
    "MailZip": "mail_zip",
    "MailState": "mail_state",
    "Phone": "phone",
    "Ext": "ext",
    "FaxNumber": "fax_number",
    "WebSite": "website",
    "OpenDate": "open_date",
    "ClosedDate": "closed_date",
    "Charter": "charter_raw",
    "CharterNum": "charter_num",
    "FundingType": "funding_type",
    "DOC": "doc",
    "DOCType": "doc_type",
    "SOC": "soc",
    "SOCType": "soc_type",
    "EdOpsCode": "ed_ops_code",
    "EdOpsName": "ed_ops_name",
    "EILCode": "eil_code",
    "EILName": "eil_name",
    "GSoffered": "gs_offered",
    "GSserved": "gs_served",
    "Virtual": "virtual",
    "Magnet": "magnet",
    "YearRoundYN": "year_round_yn",
    "FederalDFCDistrictID": "federal_dfc_district_id",
    "Latitude": "latitude",
    "Longitude": "longitude",
    "AdmFName": "adm_f_name",
    "AdmLName": "adm_l_name",
    "LastUpDate": "last_up_date",
    "Multilingual": "multilingual",
}


def _parse_charter_public(val: str) -> bool | None:
    s = str(val).strip()
    if not s or s == NO_DATA:
        return None
    if s.upper() == "Y":
        return True
    if s.upper() == "N":
        return False
    return None


def _parse_charter_frpm(val: str) -> bool | None:
    s = str(val).strip()
    if not s or s.upper() == "N/A":
        return None
    low = s.lower()
    if low.startswith("y"):
        return True
    if low.startswith("n"):
        return False
    return None


def _strip_pct(s: str) -> str:
    return str(s).strip().replace("%", "").strip()


def _to_float_maybe(x: Any) -> float | None:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    s = str(x).strip()
    if not s or s == NO_DATA:
        return None
    s = _strip_pct(s)
    try:
        return float(s)
    except ValueError:
        return None


def _to_int_maybe(x: Any) -> int | None:
    f = _to_float_maybe(x)
    if f is None:
        return None
    if f != int(f):
        return None
    return int(f)


def _is_blank_or_no_data(s: Any) -> bool:
    if s is None:
        return True
    t = str(s).strip()
    return not t or t == NO_DATA


def run_qa(merged: pd.DataFrame, lines: list[str]) -> None:
    n = len(merged)
    lines.append(f"Rows: {n}")

    dup_codes = merged["school_code"].duplicated(keep=False)
    n_dup = int(dup_codes.sum())
    lines.append(f"Duplicate school_code rows: {n_dup}")
    if n_dup:
        lines.append(f"  Examples: {merged.loc[dup_codes, 'school_code'].head(10).tolist()}")

    bad_code = merged["school_code"].isna() | (merged["school_code"].astype(str).str.strip() == "")
    lines.append(f"Missing/empty school_code: {int(bad_code.sum())}")

    bad_name = merged["school_name"].map(_is_blank_or_no_data)
    lines.append(f"Missing/No Data school_name: {int(bad_name.sum())}")

    bad_city = merged["city"].map(_is_blank_or_no_data)
    lines.append(f"Missing/No Data city: {int(bad_city.sum())}")

    for col in ("latitude", "longitude"):
        bad = merged[col].map(_is_blank_or_no_data)
        lines.append(f"Missing/No Data {col}: {int(bad.sum())}")
        numeric = merged[col].map(_to_float_maybe)
        bad_parse = merged[col].notna() & (merged[col].astype(str).str.strip() != "") & numeric.isna()
        bad_parse = bad_parse & ~merged[col].map(_is_blank_or_no_data)
        lines.append(f"Non-numeric {col} (excluding No Data): {int(bad_parse.sum())}")
        ok = numeric.dropna()
        if len(ok):
            oob = (ok < 32) | (ok > 43) if col == "latitude" else (ok > -113) | (ok < -125)
            lines.append(f"Outside rough CA bbox {col}: {int(oob.sum())}")

    if "is_charter" in merged.columns:
        amb_pub = merged["is_charter"].isna() & merged["charter_raw"].notna()
        amb_pub = amb_pub & (merged["charter_raw"].astype(str).str.strip() != "") & (
            merged["charter_raw"].astype(str).str.strip() != NO_DATA
        )
        lines.append(f"Ambiguous charter_raw (not Y/N): {int(amb_pub.sum())}")

    if "charter_school_yn" in merged.columns:
        amb_frpm = merged["charter_school_yn"].notna()
        parsed = merged["charter_school_yn"].map(_parse_charter_frpm)
        amb_frpm = amb_frpm & parsed.isna()
        lines.append(f"Ambiguous charter_school_yn (not Yes/No): {int(amb_frpm.sum())}")

    both = merged["is_charter"].notna() & merged["charter_school_yn"].notna()
    if both.any():
        pf = merged.loc[both, "is_charter"]
        ff = merged.loc[both, "charter_school_yn"].map(_parse_charter_frpm)
        disagree = pf != ff
        lines.append(f"Charter flag disagree (public vs FRPM): {int(disagree.sum())}")
        if disagree.any():
            sub = merged.loc[disagree, ["school_code", "school_name", "charter_raw", "charter_school_yn"]]
            lines.append("  Sample rows:")
            lines.append(sub.head(15).to_string(index=False))

    if "school_name_frpm" in merged.columns:
        a = merged["school_name"].astype(str).str.strip()
        b = merged["school_name_frpm"].astype(str).str.strip()
        has_frpm = merged["school_name_frpm"].notna() & (b != "")
        mismatch = has_frpm & (a != b)
        lines.append(f"School name mismatch (public vs FRPM): {int(mismatch.sum())}")
        if mismatch.any():
            sub = merged.loc[mismatch, ["school_code", "school_name", "school_name_frpm"]]
            lines.append(sub.head(15).to_string(index=False))

    frpm_cols = [
        "enrollment_k12",
        "free_meal_count_k12",
        "percent_eligible_free_k12",
        "frpm_count_k12",
        "percent_eligible_frpm_k12",
    ]
    no_frpm = merged["enrollment_k12"].isna()
    if frpm_cols[0] in merged.columns:
        lines.append(f"Public rows with no FRPM match (null enrollment_k12): {int(no_frpm.sum())}")

    for c in frpm_cols:
        if c not in merged.columns:
            continue
        raw = merged[c]
        missing = raw.isna() | (raw.astype(str).str.strip() == "") | (raw.astype(str).str.strip() == NO_DATA)
        lines.append(f"Missing/empty/No Data {c}: {int(missing.sum())}")
        bad_parse = raw.notna() & ~missing & (raw.map(_to_float_maybe).isna())
        lines.append(f"Non-numeric {c}: {int(bad_parse.sum())}")
        nums = raw.map(_to_float_maybe).dropna()
        if len(nums):
            neg = nums < 0
            lines.append(f"Negative {c}: {int(neg.sum())}")
        if c in ("frpm_count_k12", "enrollment_k12", "free_meal_count_k12"):
            nums = raw.map(_to_int_maybe)
            nonint = raw.map(_to_float_maybe).notna() & nums.isna() & raw.notna() & ~missing
            lines.append(f"Non-integer {c} (when numeric expected): {int(nonint.sum())}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--public", type=Path, default=DEFAULT_PUBLIC)
    p.add_argument("--frpm", type=Path, default=DEFAULT_FRPM)
    p.add_argument("--output", type=Path, default=DEFAULT_MERGED)
    p.add_argument("--qa", type=Path, default=DEFAULT_QA)
    args = p.parse_args()

    pub = pd.read_csv(args.public, dtype=str, keep_default_na=False)
    missing_pub = [c for c in PUBLIC_RENAME if c not in pub.columns]
    if missing_pub:
        raise ValueError(f"Public CSV missing columns: {missing_pub}")
    pub = pub.rename(columns=PUBLIC_RENAME)
    pub["school_code"] = school_code_as_str(pub["school_id"])
    pub["is_charter"] = pub["charter_raw"].map(_parse_charter_public)

    frpm = pd.read_csv(args.frpm)
    frpm = normalize_frpm_columns(frpm)
    missing_keys = [k for k in FRPM_FULL_RENAME if k not in frpm.columns]
    if missing_keys:
        raise ValueError(f"FRPM CSV missing columns after normalize: {missing_keys}")
    frpm["School Code"] = school_code_as_str(frpm["School Code"])
    frpm = frpm.rename(columns=FRPM_FULL_RENAME)

    dup_mask = frpm["school_code"].duplicated(keep=False)
    n_dup_frpm = int(dup_mask.sum())
    n_dup_codes = int(frpm["school_code"].duplicated().sum())
    if n_dup_frpm:
        print(
            f"Warning: {n_dup_frpm} FRPM rows share duplicate school_code; "
            "keeping first per code.",
            file=sys.stderr,
        )
        frpm = frpm.drop_duplicates(subset=["school_code"], keep="first")

    merged = pub.merge(frpm, on="school_code", how="left", suffixes=("_public", "_frpm"))
    if "school_name_public" in merged.columns:
        merged = merged.rename(columns={"school_name_public": "school_name"})

    merged = _reorder_merged_columns(merged)

    lines: list[str] = []
    if n_dup_frpm:
        lines.append(
            f"FRPM duplicate school_code rows before dedup: {n_dup_frpm} "
            f"({n_dup_codes} extra rows dropped)"
        )
    run_qa(merged, lines)
    report = "\n".join(lines) + "\n"
    print(report)
    args.qa.parent.mkdir(parents=True, exist_ok=True)
    args.qa.write_text(report, encoding="utf-8")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output, index=False)
    print(f"Wrote {args.output}")
    print(f"Wrote {args.qa}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
