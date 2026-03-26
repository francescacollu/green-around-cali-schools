"""
Shared FRPM CSV helpers: header normalization, school code formatting, snake_case renames.
"""

from __future__ import annotations

import pandas as pd

# Nine columns used by merge scripts (canonical names after normalize).
FRPM_COLUMNS = [
    "School Code",
    "School Name",
    "School Type",
    "Charter School (Y/N)",
    "Enrollment (K-12)",
    "Free Meal Count (K-12)",
    "Percent (%) Eligible Free (K-12)",
    "FRPM Count (K-12)",
    "Percent (%) Eligible FRPM (K-12)",
]

RENAME_TO_SNAKE = {
    "School Code": "school_code",
    "School Name": "school_name",
    "School Type": "school_type",
    "Charter School (Y/N)": "charter_school_yn",
    "Enrollment (K-12)": "enrollment_k12",
    "Free Meal Count (K-12)": "free_meal_count_k12",
    "Percent (%) Eligible Free (K-12)": "percent_eligible_free_k12",
    "FRPM Count (K-12)": "frpm_count_k12",
    "Percent (%) Eligible FRPM (K-12)": "percent_eligible_frpm_k12",
}

# All normalized FRPM column names (after normalize_frpm_columns) to snake_case for full merges.
FRPM_FULL_RENAME = {
    **RENAME_TO_SNAKE,
    "Academic Year": "academic_year",
    "County Code": "county_code",
    "District Code": "district_code",
    "County Name": "county_name",
    "District Name": "district_name",
    "District Type": "district_type",
    "Educational Option Type": "educational_option_type",
    "Charter School Number": "charter_school_number",
    "Charter Funding Type": "charter_funding_type",
    "Independently Reporting Charter (IRC)": "independently_reporting_charter_irc",
    "Low Grade": "low_grade",
    "High Grade": "high_grade",
    "Enrollment (Ages 5-17)": "enrollment_ages_5_17",
    "Free Meal Count (Ages 5-17)": "free_meal_count_ages_5_17",
    "Percent (%) Eligible Free (Ages 5-17)": "percent_eligible_free_ages_5_17",
    "FRPM Count (Ages 5-17)": "frpm_count_ages_5_17",
    "Percent (%) Eligible FRPM (Ages 5-17)": "percent_eligible_frpm_ages_5_17",
    "California Longitudinal Pupil Achievement Data System (CALPADS) Fall 1 Certification Status": (
        "calpads_fall1_certification_status"
    ),
}


def normalize_frpm_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).replace("\n", " ").replace("\r", "") for c in out.columns]
    out.columns = out.columns.str.replace(r"\s+", " ", regex=True).str.strip()
    return out


def school_code_as_str(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.replace(r"\.0$", "", regex=True).str.strip()
    out: list[str] = []
    for x in s:
        if x.isdigit():
            out.append(x.zfill(7))
        else:
            out.append(x)
    return pd.Series(out, index=series.index, dtype=str)
