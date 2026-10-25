"""Classify baseball draft schools as College, JUCO, or High School.

This reproduces the rule-based classification used on earlier draft files.
Averages of ``age_june_1`` are computed *within each input file*, in MONTHS.

Usage:
    python classify_draft_schools.py input.xlsx
    python classify_draft_schools.py input.xlsx --output classified.xlsx

Requires: artifact_tool (for reading/writing Excel workbooks)
"""

import argparse
from collections import Counter, defaultdict
from math import isfinite
from pathlib import Path

from artifact_tool import Blob, SpreadsheetFile


# School-level overrides collected while reviewing earlier draft data.
EXPLICIT_HS_OVERRIDES = {
    "Home School (Oklahoma City, OK)",
    "Avon Old Farms School",
    "The Benjamin School (FL)",
    "Miller School of Albemarle",
    "Evangelical Christian School",
    "Fort Worth Christian School (TX)",
    "American Heritage School",
    "Westminster Christian School",
    "Liberty Christian School",
    "Mercersburg Academy",
    "St. Joseph Regional School",
    "Ezell Harding Christian School",
    "No School (Hazleton, PA)",
    "Memphis University School",
    "PJ Education School",
    "The Benjamin School",
    "Mobile Christian School",
}

EXTRA_JUCO_NAMES = {
    "Odessa College",
    "Mt. San Antonio College",
    "Yavapai College",
    "State College of Florida, Manatee-Sarasota",
    "State College of Florida Manatee - Sarasota",
    "Howard College ",
    "Howard College",
    "Spartanburg Methodist College",
    "San Diego Mesa College",
    "Eastern Oklahoma State College",
    "Seminole State College (OK)",
    "Gordon State College",
    "Parkland College",
    "Lackawanna College",
    "Ohlone College",
    "College of the Canyons",
    "Chipola College",
}


def is_no_school_base(name):
    return isinstance(name, str) and name.strip().lower() in {
        "no school", "none", ""
    }


def is_core_hs_token(name):
    if not isinstance(name, str):
        return False
    n = name.lower()
    return (
        " hs" in n
        or n.endswith("hs")
        or "high school" in n
        or " hs)" in n
        or " hs," in n
        or "charter school" in n
        or " charter " in n
        or "home school" in n
    )


def has_college_keyword(name):
    if not isinstance(name, str):
        return False
    n = name.lower()
    return (
        "university" in n
        or " college" in n
        or " community college" in n
        or " junior college" in n
        or " institute of technology" in n
        or " polytechnic" in n
        or " cc" in n
        or "cc," in n
        or " jc" in n
        or "jc," in n
    )


def is_academy_or_prep(name):
    if not isinstance(name, str):
        return False
    n = name.lower()
    return "academy" in n or " prep" in n or "preparatory" in n


def is_juco(name):
    if not isinstance(name, str):
        return False
    if name in EXTRA_JUCO_NAMES:
        return True
    n = name.lower()
    return (
        " community college" in n
        or " junior college" in n
        or " state jc" in n
        or "state cc" in n
        or " cc" in n
        or "cc," in n
        or " jc" in n
        or "jc," in n
    )


def average_ages_by_school(rows, school_col, age_col):
    """Calculate average draft age in months for each named school."""
    sums = defaultdict(float)
    counts = Counter()
    for row in rows:
        school = row[school_col]
        age = row[age_col]
        if isinstance(school, str) and isinstance(age, (int, float)):
            if not isinstance(age, bool) and isfinite(age):
                sums[school] += age
                counts[school] += 1
    return {school: sums[school] / n for school, n in counts.items()}


def classify_school(name, average_ages):
    """Return one of 'High School', 'JUCO', or 'College'."""
    if not isinstance(name, str):
        return "High School"
    if name in EXPLICIT_HS_OVERRIDES:
        return "High School"

    n = name.lower().strip()
    if is_no_school_base(name) or "home school" in n:
        return "High School"

    # Look for 2-year institution identifiers before broader college rules.
    if is_juco(name):
        return "JUCO"

    mean_age = average_ages.get(name)  # age_june_1 is in months
    if is_core_hs_token(name):
        return "High School"

    # Younger academy/preparatory-school draftees usually come from high school.
    if is_academy_or_prep(name):
        if has_college_keyword(name) and (mean_age is None or mean_age >= 236):
            return "College"
        if mean_age is not None and mean_age <= 230:
            return "High School"
        if mean_age is None:
            return "High School"
        return "College"

    if has_college_keyword(name):
        return "College"
    if mean_age is not None and mean_age <= 224:
        return "High School"
    return "College"


def classify_workbook(input_path, output_path):
    workbook = SpreadsheetFile.import_xlsx(Blob.load(str(input_path)))
    sheet = workbook.worksheets.get_item_at(0)
    data = sheet.get_range("A1").get_current_region().values
    if not data or len(data) < 2:
        raise ValueError("The first worksheet must contain a header and data rows.")

    headers = data[0]
    if "school_name" not in headers or "age_june_1" not in headers:
        raise ValueError("Required columns: school_name and age_june_1")

    school_col = headers.index("school_name")
    age_col = headers.index("age_june_1")
    school_ages = average_ages_by_school(data[1:], school_col, age_col)
    classifications = [classify_school(row[school_col], school_ages) for row in data[1:]]

    # Append new classification or replace an existing column of that name.
    target_col = (headers.index("school_type_3cat") if "school_type_3cat" in headers
                  else len(headers))
    sheet.get_range_by_indexes(0, target_col, 1, 1).values = [["school_type_3cat"]]
    sheet.get_range_by_indexes(1, target_col, len(classifications), 1).values = [
        [value] for value in classifications
    ]

    # Match the adjacent header's style and provide a readable column width.
    # Copy formatting BEFORE rewriting header, since 'all' also copies text.
    # The header value is rewritten after style copying, when adding a new col.
    if target_col == len(headers):
        prev_header = sheet.get_range_by_indexes(0, target_col - 1, 1, 1)
        this_header = sheet.get_range_by_indexes(0, target_col, 1, 1)
        this_header.copy_from(prev_header, "all")
        this_header.values = [["school_type_3cat"]]
    sheet.get_range_by_indexes(0, target_col, len(classifications) + 1, 1).format.column_width = 21

    SpreadsheetFile.export_xlsx(workbook).save(str(output_path))
    print(f"Saved: {output_path}")
    print(f"Rows classified: {len(classifications)}")
    print("Counts:", dict(Counter(classifications)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Source Excel file")
    parser.add_argument("--output", type=Path, help="Where to save the classified workbook")
    args = parser.parse_args()
    output = args.output or args.input.with_name(args.input.stem + "_classified.xlsx")
    classify_workbook(args.input, output)


if __name__ == "__main__":
    main()
