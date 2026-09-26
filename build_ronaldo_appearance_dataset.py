"""
Build Cristiano Ronaldo fixture-level appearance dataset automatically.

Purpose
-------
Create the dataset needed for an appearance-probability model:

    fixture occurs -> did Ronaldo appear?

The script:
1. downloads Transfermarkt detailed season pages,
2. extracts fixture/status rows,
3. classifies appearance vs non-appearance opportunities,
4. filters to Ronaldo's senior club + Portugal career,
5. merges against the existing canonical 1,333-appearance dataset,
6. writes raw, audit, and ML-ready fixture-level CSV files.

Important
---------
- Historical cutoff is fixed at 2026-08-28.
- Post-cutoff Sep 2026 matches are NOT included.
- "Not eligible" / not registered rows are preserved for audit but excluded
  from the model opportunity set.
- Portugal Olympic/youth rows are excluded.
- The canonical appearance dataset remains the positive-label authority.
"""

from __future__ import annotations

import argparse
import io
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from difflib import SequenceMatcher


PLAYER_ID = 8198
CUTOFF = pd.Timestamp("2026-08-28")
DOB = pd.Timestamp("1985-02-05")

SEASON_START_YEARS = list(range(2002, 2027))

BASE_URL = (
    "https://www.transfermarkt.com/cristiano-ronaldo/"
    "leistungsdatendetails/spieler/8198/plus/1/saison/{season}"
)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

CLUB_TENURES = [
    ("Sporting CP", pd.Timestamp("2002-07-01"), pd.Timestamp("2003-08-11")),
    ("Manchester United", pd.Timestamp("2003-08-12"), pd.Timestamp("2009-06-30")),
    ("Real Madrid", pd.Timestamp("2009-07-01"), pd.Timestamp("2018-07-09")),
    ("Juventus", pd.Timestamp("2018-07-10"), pd.Timestamp("2021-08-26")),
    ("Manchester United", pd.Timestamp("2021-08-27"), pd.Timestamp("2022-11-22")),
    ("Al-Nassr", pd.Timestamp("2023-01-01"), CUTOFF),
]

PORTUGAL_START = pd.Timestamp("2003-08-20")

TEAM_ALIASES = {
    "Sporting CP": "Sporting CP",
    "Sporting": "Sporting CP",
    "Man Utd": "Manchester United",
    "Manchester United": "Manchester United",
    "Real Madrid": "Real Madrid",
    "Juventus": "Juventus",
    "Juventus FC": "Juventus",
    "Al-Nassr": "Al-Nassr",
    "Al-Nassr FC": "Al-Nassr",
    "Portugal": "Portugal",
}

OPPONENT_ALIASES = {
    "Man City": "Manchester City",
    "Man Utd": "Manchester United",
    "Newcastle": "Newcastle United",
    "Tottenham": "Tottenham Hotspur",
    "West Ham": "West Ham United",
    "Wolves": "Wolverhampton Wanderers",
    "Leicester": "Leicester City",
    "Norwich": "Norwich City",
    "Leeds": "Leeds United",
    "Roma": "AS Roma",
    "Valladolid": "Real Valladolid",
    "Zaragoza": "Real Zaragoza",
    "Rep. of Ireland": "Republic of Ireland",
    "Republic of Ireland": "Republic of Ireland",
    "Bosnia-Herzegovina": "Bosnia & Herzegovina",
    "Bosnia and Herzegovina": "Bosnia & Herzegovina",
    "Türkiye": "Türkiye",
    "Turkey": "Türkiye",
    "FYR Macedonia": "North Macedonia",
    "Al-Ittihad": "Al-Ittihad FC",
    "Al-Ittihad FC": "Al-Ittihad FC",
    "Al-Khaleej": "Al Khaleej Saihat",
    "Al Khaleej": "Al Khaleej Saihat",
    "Al-Ahli": "Al-Ahli Jeddah",
    "Al-Ahli Jeddah": "Al-Ahli Jeddah",
    "Al-Hazm": "Al-Hazm",
    "Al-Taawoun": "Al Taawon",
    "Al-Taawon": "Al Taawon",
    "Al-Qadsiah": "Al-Qadisiyah FC",
    "Al-Qadisiyah": "Al-Qadisiyah FC",
    "Al-Fayha": "Al-Fayha",
    "Al-Hilal": "Al-Hilal",
    "Al-Fateh": "Al-Fateh",
    "Al-Gharafa": "Al-Gharafa",
    "Al-Wasl": "Al-Wasl FC",
    "Shabab Al-Ahli": "Shabab Al Ahli Dubai",
}

STATUS_PATTERNS = {
    "unused_sub": [
        r"\bon the bench\b",
        r"\bbench\b",
    ],
    "not_in_squad": [
        r"\bnot in squad\b",
        r"\bnot in the squad\b",
        r"\babsence\b",
    ],
    "injured": [
        r"\binjured\b",
        r"\binjury\b",
        r"\bill\b",
        r"\bvirus\b",
    ],
    "suspended": [
        r"\bsuspended\b",
        r"\bsuspension\b",
    ],
    "not_eligible": [
        r"\bnot eligible\b",
        r"\bno eligibility\b",
        r"\bnot registered\b",
        r"\bkeine spielberechtigung\b",
    ],
}

POSITION_CODES = {
    "GK", "CB", "LB", "RB", "DM", "CM", "AM",
    "LM", "RM", "LW", "RW", "SS", "CF",
}


def clean_text(x) -> str:
    if pd.isna(x):
        return ""
    s = str(x).replace("\xa0", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = [
            " ".join([clean_text(v) for v in tup if clean_text(v)])
            for tup in out.columns
        ]
    else:
        out.columns = [clean_text(c) for c in out.columns]
    return out


def normalize_name(name: str) -> str:
    s = clean_text(name)
    s = re.sub(r"\s*\(\d+\.\)\s*$", "", s).strip()
    s = re.sub(r"\s*\(\d+\)\s*$", "", s).strip()
    return OPPONENT_ALIASES.get(s, TEAM_ALIASES.get(s, s))


def parse_date(value, season_start: int) -> pd.Timestamp | pd.NaT:
    s = clean_text(value)
    if not s:
        return pd.NaT

    # Remove weekday / stray annotations.
    s = re.sub(r"\s+", " ", s)

    candidates = [
        dict(dayfirst=False),
        dict(dayfirst=True),
    ]

    for kwargs in candidates:
        try:
            dt = pd.to_datetime(s, errors="raise", **kwargs)
            if pd.isna(dt):
                continue

            # Transfermarkt rows without a year are rare, but handle them.
            if getattr(dt, "year", 0) == 1900:
                month = dt.month
                year = season_start if month >= 7 else season_start + 1
                dt = dt.replace(year=year)

            return pd.Timestamp(dt).normalize()
        except Exception:
            pass

    # Numeric mm/dd/yy or dd/mm/yy fallbacks.
    for fmt in ("%m/%d/%y", "%d/%m/%y", "%m/%d/%Y", "%d/%m/%Y"):
        try:
            return pd.Timestamp(pd.to_datetime(s, format=fmt)).normalize()
        except Exception:
            pass

    return pd.NaT


def season_label(date: pd.Timestamp) -> str:
    y = date.year if date.month >= 7 else date.year - 1
    return f"{y}/{str(y + 1)[-2:]}"


def infer_competition_from_context(context: str) -> str:
    text = clean_text(context).lower()

    mapping = [
        ("champions league qualifying", "Champions League Qualifying"),
        ("uefa champions league", "UEFA Champions League"),
        ("champions league", "UEFA Champions League"),
        ("europa league", "UEFA Europa League"),
        ("club world cup", "FIFA Club World Cup"),
        ("premier league", "Premier League"),
        ("fa cup", "FA Cup"),
        ("efl cup", "League Cup"),
        ("league cup", "League Cup"),
        ("community shield", "FA Community Shield"),
        ("laliga", "La Liga"),
        ("la liga", "La Liga"),
        ("copa del rey", "Copa del Rey"),
        ("supercopa", "Supercopa de Espana"),
        ("serie a", "Serie A"),
        ("italy cup", "Coppa Italia"),
        ("coppa italia", "Coppa Italia"),
        ("supercoppa", "Supercoppa Italiana"),
        ("saudi pro league", "Pro League"),
        ("king's cup", "King's Cup"),
        ("kings cup", "King's Cup"),
        ("saudi super cup", "Saudi Super Cup"),
        ("afc champions league elite", "AFC Champions League Elite"),
        ("afc champions league", "AFC Champions League"),
        ("arab club champions", "Arab Club Champions Cup"),
        ("world cup qualification", "World Cup - Qualification Europe"),
        ("world cup qualifiers", "World Cup - Qualification Europe"),
        ("world cup", "World Cup"),
        ("euro qualification", "Euros Qualifier"),
        ("euro qualifier", "Euros Qualifier"),
        ("european championship", "Euro Championship"),
        ("euro 20", "Euro Championship"),
        ("nations league", "UEFA Nations League"),
        ("friendlies", "International Friendly"),
        ("friendly", "International Friendly"),
        ("liga portugal", "Primeira Liga"),
        ("primeira liga", "Primeira Liga"),
        ("taça de portugal", "Taça de Portugal"),
    ]

    for needle, comp in mapping:
        if needle in text:
            return comp

    return clean_text(context)


def classify_status(row_text: str, pos_text: str, minutes_text: str) -> tuple[int, str, int]:
    full = f"{row_text} {pos_text} {minutes_text}".lower()

    for status, patterns in STATUS_PATTERNS.items():
        if any(re.search(p, full, flags=re.I) for p in patterns):
            eligible = 0 if status == "not_eligible" else 1
            return 0, status, eligible

    # If a football position is present or minutes are present, he appeared.
    pos = clean_text(pos_text).upper()
    if pos in POSITION_CODES:
        return 1, "appeared", 1

    if re.search(r"\b\d{1,3}'\b", minutes_text):
        return 1, "appeared", 1

    # Unknown status: preserve, don't silently label.
    return -1, "unresolved", 1


def find_col(columns: Iterable[str], patterns: Iterable[str]) -> str | None:
    cols = list(columns)
    for p in patterns:
        for c in cols:
            if re.search(p, c, flags=re.I):
                return c
    return None


def extract_rows_from_table(
    table: pd.DataFrame,
    season_start: int,
    source_url: str,
    table_index: int,
) -> list[dict]:
    df = flatten_columns(table)

    date_col = find_col(df.columns, [r"\bdate\b"])
    result_col = find_col(df.columns, [r"\bresult\b"])
    pos_col = find_col(df.columns, [r"\bpos\.?\b", r"\bposition\b"])
    venue_col = find_col(df.columns, [r"\bvenue\b"])
    md_col = find_col(df.columns, [r"\bmatchday\b", r"\bround\b"])

    home_col = find_col(df.columns, [r"\bhome team\b"])
    away_col = find_col(df.columns, [r"\baway team\b"])
    for_col = find_col(df.columns, [r"^for$", r"\bfor\b"])
    opp_col = find_col(df.columns, [r"\bopponent\b"])

    if date_col is None or result_col is None:
        return []

    if not ((home_col and away_col) or (for_col and opp_col)):
        return []

    rows = []

    # Competition often exists in a header-like first column or surrounding table text.
    context = " ".join(df.columns)

    for _, row in df.iterrows():
        date = parse_date(row.get(date_col, ""), season_start)
        if pd.isna(date) or date > CUTOFF:
            continue

        result = clean_text(row.get(result_col, ""))
        if not re.search(r"\d+\s*:\s*\d+", result):
            continue

        venue = clean_text(row.get(venue_col, "")).upper()
        pos = clean_text(row.get(pos_col, ""))

        row_values = [clean_text(v) for v in row.tolist()]
        row_text = " | ".join(row_values)

        minute_candidates = [
            v for v in row_values if re.fullmatch(r"\d{1,3}'", v)
        ]
        minutes_text = minute_candidates[-1] if minute_candidates else ""

        if home_col and away_col:
            home = normalize_name(row.get(home_col, ""))
            away = normalize_name(row.get(away_col, ""))

            known = set(TEAM_ALIASES.values())
            if home in known:
                team, opponent, inferred_venue = home, away, "H"
            elif away in known:
                team, opponent, inferred_venue = away, home, "A"
            else:
                continue

            if not venue:
                venue = inferred_venue

        else:
            team = normalize_name(row.get(for_col, ""))
            opponent = normalize_name(row.get(opp_col, ""))

            if team not in set(TEAM_ALIASES.values()):
                continue

        appeared, status, eligible = classify_status(
            row_text=row_text,
            pos_text=pos,
            minutes_text=minutes_text,
        )

        matchday = clean_text(row.get(md_col, "")) if md_col else ""

        rows.append(
            {
                "date": date,
                "season": season_label(date),
                "team": team,
                "opponent": opponent,
                "competition_raw_context": context,
                "competition": infer_competition_from_context(context),
                "venue": venue if venue in {"H", "A", "N"} else "",
                "result": result,
                "matchday": matchday,
                "position_or_status": pos,
                "minutes_raw": minutes_text,
                "appeared_source": appeared,
                "availability_status": status,
                "eligible_opportunity": eligible,
                "source_season": season_start,
                "source_table": table_index,
                "source_url": source_url,
            }
        )

    return rows


def fetch_html(url: str, cache_file: Path, pause: float) -> str:
    if cache_file.exists():
        return cache_file.read_text(encoding="utf-8", errors="ignore")

    r = requests.get(url, headers=HEADERS, timeout=45)
    if r.status_code != 200:
        raise RuntimeError(
            f"Transfermarkt returned HTTP {r.status_code} for {url}. "
            "If this is a bot block, rerun later or place saved HTML pages "
            f"into {cache_file.parent}."
        )

    html = r.text
    if "captcha" in html.lower() or len(html) < 20_000:
        raise RuntimeError(
            f"Transfermarkt response for {url} looks blocked/incomplete."
        )

    cache_file.write_text(html, encoding="utf-8")
    time.sleep(pause)
    return html


def scrape_transfermarkt(cache_dir: Path, pause: float = 1.5) -> pd.DataFrame:
    all_rows = []

    for season in SEASON_START_YEARS:
        url = BASE_URL.format(season=season)
        cache_file = cache_dir / f"transfermarkt_{season}.html"

        print(f"[download] season {season}/{str(season+1)[-2:]}")

        html = fetch_html(url, cache_file, pause=pause)

        try:
            tables = pd.read_html(io.StringIO(html))
        except ValueError:
            tables = []

        season_rows = []
        for i, table in enumerate(tables):
            season_rows.extend(
                extract_rows_from_table(
                    table=table,
                    season_start=season,
                    source_url=url,
                    table_index=i,
                )
            )

        print(f"           extracted candidate rows: {len(season_rows)}")
        all_rows.extend(season_rows)

    if not all_rows:
        raise RuntimeError("No fixture rows were extracted.")

    df = pd.DataFrame(all_rows)

    # Drop exact duplicates from repeated/summary tables.
    key = [
        "date", "team", "opponent", "competition",
        "result", "availability_status"
    ]
    df = (
        df.sort_values(["date", "team", "opponent"])
          .drop_duplicates(key, keep="first")
          .reset_index(drop=True)
    )

    return df


def within_club_tenure(team: str, date: pd.Timestamp) -> bool:
    for t, start, end in CLUB_TENURES:
        if team == t and start <= date <= end:
            return True
    return False


def apply_senior_eligibility_filters(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    club_mask = out.apply(
        lambda r: within_club_tenure(r["team"], r["date"]),
        axis=1,
    )

    portugal_mask = (
        (out["team"] == "Portugal")
        & (out["date"] >= PORTUGAL_START)
        & (out["date"] <= CUTOFF)
    )

    out["within_senior_tenure"] = (club_mask | portugal_mask).astype(int)

    # Exclude youth/Olympic contexts if they slipped through.
    youth_terms = r"olympic|u21|u-21|under 21|u23|u-23|under 23|youth"
    youth_mask = (
        out["competition_raw_context"]
        .astype(str)
        .str.contains(youth_terms, case=False, regex=True, na=False)
    )

    out["senior_fixture"] = (~youth_mask).astype(int)

    return out[
        (out["within_senior_tenure"] == 1)
        & (out["senior_fixture"] == 1)
        & (out["date"] <= CUTOFF)
    ].copy()


def compact_key(s: str) -> str:
    s = normalize_name(s).lower()
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, compact_key(a), compact_key(b)).ratio()


def reconcile_with_canonical(
    fixtures: pd.DataFrame,
    canonical: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    can = canonical.copy()
    can["date"] = pd.to_datetime(can["date"]).dt.normalize()
    can = can[can["date"] <= CUTOFF].copy()

    # Correct source of truth.
    required = {"match_id", "date", "team", "opponent"}
    missing = required - set(can.columns)
    if missing:
        raise ValueError(f"Canonical dataset missing columns: {sorted(missing)}")

    f = fixtures.copy()
    f["canonical_match_id"] = pd.NA
    f["canonical_opponent"] = pd.NA
    f["match_score"] = np.nan
    f["appeared"] = 0

    # Primary matching: same date + canonical team, then opponent similarity.
    used_fixture_indices = set()

    unmatched_canonical = []

    for _, cr in can.iterrows():
        candidates = f[
            (f["date"] == cr["date"])
            & (f["team"] == cr["team"])
        ].copy()

        if candidates.empty:
            unmatched_canonical.append(
                {
                    "match_id": cr["match_id"],
                    "date": cr["date"],
                    "team": cr["team"],
                    "opponent": cr["opponent"],
                    "reason": "no_same_date_team_fixture",
                }
            )
            continue

        candidates["score"] = candidates["opponent"].map(
            lambda x: similarity(x, cr["opponent"])
        )

        # Avoid assigning the same fixture twice.
        candidates = candidates[
            ~candidates.index.isin(used_fixture_indices)
        ]

        if candidates.empty:
            unmatched_canonical.append(
                {
                    "match_id": cr["match_id"],
                    "date": cr["date"],
                    "team": cr["team"],
                    "opponent": cr["opponent"],
                    "reason": "same_fixture_already_used",
                }
            )
            continue

        best_idx = candidates["score"].idxmax()
        best_score = float(candidates.loc[best_idx, "score"])

        if best_score < 0.60:
            unmatched_canonical.append(
                {
                    "match_id": cr["match_id"],
                    "date": cr["date"],
                    "team": cr["team"],
                    "opponent": cr["opponent"],
                    "reason": f"low_opponent_similarity_{best_score:.3f}",
                }
            )
            continue

        used_fixture_indices.add(best_idx)
        f.loc[best_idx, "canonical_match_id"] = cr["match_id"]
        f.loc[best_idx, "canonical_opponent"] = cr["opponent"]
        f.loc[best_idx, "match_score"] = best_score
        f.loc[best_idx, "appeared"] = 1

    unmatched_df = pd.DataFrame(unmatched_canonical)

    # Canonical positives override source-status inference.
    f.loc[f["canonical_match_id"].notna(), "appeared"] = 1

    # If source says appeared but canonical didn't match, leave for audit.
    f["source_positive_unmatched"] = (
        (f["appeared_source"] == 1)
        & (f["canonical_match_id"].isna())
    ).astype(int)

    return f, unmatched_df


def add_basic_pre_match_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values(["date", "team", "opponent"]).reset_index(drop=True).copy()

    out["fixture_id"] = [
        f"{d.date()}_{re.sub(r'[^A-Za-z0-9]+', '_', t).strip('_')}_"
        f"{re.sub(r'[^A-Za-z0-9]+', '_', o).strip('_')}_{i+1}"
        for i, (d, t, o) in enumerate(
            zip(out["date"], out["team"], out["opponent"])
        )
    ]

    out["age_years"] = (
        (out["date"] - DOB).dt.days / 365.2425
    ).round(4)

    out["is_national_team"] = (out["team"] == "Portugal").astype(int)
    out["month"] = out["date"].dt.month
    out["day_of_week"] = out["date"].dt.day_name()

    # Team-fixture cadence.
    out["days_since_previous_team_fixture"] = 0.0
    out["team_matches_last_7d"] = 0
    out["team_matches_last_14d"] = 0
    out["team_matches_last_30d"] = 0

    # Ronaldo availability state.
    out["days_since_ronaldo_last_appearance"] = 0.0
    out["ronaldo_appearances_last_5_team_matches"] = 0
    out["ronaldo_appearances_last_10_team_matches"] = 0
    out["ronaldo_appearance_rate_last_5_team_matches"] = 0.0
    out["ronaldo_appearance_rate_last_10_team_matches"] = 0.0
    out["consecutive_appearances_pre"] = 0
    out["consecutive_missed_pre"] = 0
    out["team_fixture_number_season_pre"] = 0
    out["ronaldo_team_appearances_season_pre"] = 0
    out["ronaldo_team_appearance_rate_season_pre"] = 0.0

    # Important: compute only among eligible model opportunities.
    histories = {}

    for idx, row in out.iterrows():
        team = row["team"]
        date = row["date"]
        season = row["season"]

        if team not in histories:
            histories[team] = {
                "dates": [],
                "appearance_history": [],
                "last_appearance_date": None,
                "appearance_streak": 0,
                "miss_streak": 0,
                "season": {},
            }

        st = histories[team]
        prior_dates = st["dates"]
        prior_apps = st["appearance_history"]

        if prior_dates:
            out.loc[idx, "days_since_previous_team_fixture"] = (
                date - prior_dates[-1]
            ).days

        for window in (7, 14, 30):
            count = sum(
                1 for d in prior_dates
                if 0 < (date - d).days <= window
            )
            out.loc[idx, f"team_matches_last_{window}d"] = count

        if st["last_appearance_date"] is not None:
            out.loc[idx, "days_since_ronaldo_last_appearance"] = (
                date - st["last_appearance_date"]
            ).days

        last5 = prior_apps[-5:]
        last10 = prior_apps[-10:]

        out.loc[idx, "ronaldo_appearances_last_5_team_matches"] = sum(last5)
        out.loc[idx, "ronaldo_appearances_last_10_team_matches"] = sum(last10)

        out.loc[idx, "ronaldo_appearance_rate_last_5_team_matches"] = (
            sum(last5) / len(last5) if last5 else 0.0
        )
        out.loc[idx, "ronaldo_appearance_rate_last_10_team_matches"] = (
            sum(last10) / len(last10) if last10 else 0.0
        )

        out.loc[idx, "consecutive_appearances_pre"] = st["appearance_streak"]
        out.loc[idx, "consecutive_missed_pre"] = st["miss_streak"]

        ss = st["season"].setdefault(
            season,
            {"fixtures": 0, "apps": 0},
        )

        out.loc[idx, "team_fixture_number_season_pre"] = ss["fixtures"]
        out.loc[idx, "ronaldo_team_appearances_season_pre"] = ss["apps"]
        out.loc[idx, "ronaldo_team_appearance_rate_season_pre"] = (
            ss["apps"] / ss["fixtures"] if ss["fixtures"] else 0.0
        )

        # Non-eligible rows do not update model-opportunity history.
        if int(row["eligible_opportunity"]) != 1:
            continue

        appeared = int(row["appeared"])

        st["dates"].append(date)
        st["appearance_history"].append(appeared)

        ss["fixtures"] += 1
        ss["apps"] += appeared

        if appeared:
            st["last_appearance_date"] = date
            st["appearance_streak"] += 1
            st["miss_streak"] = 0
        else:
            st["miss_streak"] += 1
            st["appearance_streak"] = 0

    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--canonical",
        required=True,
        help="Path to ronaldo_1000_ml_canonical_final.csv",
    )
    parser.add_argument(
        "--outdir",
        default="data/appearance",
    )
    parser.add_argument(
        "--pause",
        type=float,
        default=1.5,
        help="Delay between uncached web requests.",
    )
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    cache_dir = outdir / "raw_html"
    cache_dir.mkdir(parents=True, exist_ok=True)

    canonical = pd.read_csv(
        args.canonical,
        parse_dates=["date"],
    ).sort_values("date").reset_index(drop=True)

    if len(canonical) != 1333:
        raise ValueError(
            f"Expected canonical 1,333 appearances, got {len(canonical)}."
        )

    if "target_goals" in canonical.columns:
        total_goals = canonical["target_goals"].sum()
        if total_goals != 978:
            raise ValueError(
                f"Expected 978 canonical goals, got {total_goals}."
            )

    print("\n1) Scraping Transfermarkt season pages...")
    raw = scrape_transfermarkt(
        cache_dir=cache_dir,
        pause=args.pause,
    )

    raw.to_csv(
        outdir / "fixtures_transfermarkt_raw.csv",
        index=False,
    )

    print("\n2) Restricting to senior career / cutoff...")
    fixtures = apply_senior_eligibility_filters(raw)

    print("\n3) Reconciling to canonical 1,333 appearances...")
    reconciled, unmatched = reconcile_with_canonical(
        fixtures=fixtures,
        canonical=canonical,
    )

    reconciled.to_csv(
        outdir / "fixtures_reconciled.csv",
        index=False,
    )

    unmatched.to_csv(
        outdir / "unmatched_canonical_appearances.csv",
        index=False,
    )

    print("\n4) Building pre-match availability features...")
    engineered = add_basic_pre_match_features(reconciled)

    # Only legal binary opportunities for training.
    model_table = engineered[
        engineered["eligible_opportunity"] == 1
    ].copy()

    model_table = model_table[
        model_table["appeared"].isin([0, 1])
    ].copy()

    model_table.to_csv(
        outdir / "ronaldo_appearance_model_table.csv",
        index=False,
    )

    # Audit summary.
    summary = pd.DataFrame(
        [
            ("canonical_appearances", len(canonical)),
            ("canonical_goals", canonical.get("target_goals", pd.Series(dtype=float)).sum()),
            ("raw_fixture_rows", len(raw)),
            ("senior_fixture_rows", len(fixtures)),
            ("model_opportunities", len(model_table)),
            ("appeared_1", int((model_table["appeared"] == 1).sum())),
            ("appeared_0", int((model_table["appeared"] == 0).sum())),
            ("canonical_matches_reconciled", int(reconciled["canonical_match_id"].notna().sum())),
            ("unmatched_canonical", len(unmatched)),
            ("source_positive_unmatched", int(reconciled["source_positive_unmatched"].sum())),
            ("not_eligible_rows_preserved", int((reconciled["eligible_opportunity"] == 0).sum())),
            ("cutoff", str(CUTOFF.date())),
        ],
        columns=["metric", "value"],
    )

    summary.to_csv(
        outdir / "appearance_dataset_audit.csv",
        index=False,
    )

    print("\nAPPEARANCE DATASET AUDIT")
    print("=" * 78)
    print(summary.to_string(index=False))

    print("\nFiles written:")
    for p in [
        outdir / "fixtures_transfermarkt_raw.csv",
        outdir / "fixtures_reconciled.csv",
        outdir / "unmatched_canonical_appearances.csv",
        outdir / "ronaldo_appearance_model_table.csv",
        outdir / "appearance_dataset_audit.csv",
    ]:
        print(" -", p)

    print(
        "\nIMPORTANT: do not train the appearance model until "
        "`unmatched_canonical == 0` and source-positive unmatched rows "
        "have been reviewed."
    )


if __name__ == "__main__":
    main()
