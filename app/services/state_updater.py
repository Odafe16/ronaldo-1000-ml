# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 08:37:47 2026

@author: JONATHAN
"""

import copy

import pandas as pd


GOAL_EWMA_ALPHA = 0.30


def update_state_after_appearance(
    state,
    *,
    date,
    season,
    team,
    opponent,
    competition,
    venue,
    goals,
):
    """Update recursive state after a real Ronaldo appearance."""

    state = copy.deepcopy(state)

    date = pd.Timestamp(date)
    goals = int(goals)

    if goals < 0:
        raise ValueError("goals cannot be negative")

    # Career
    state["career_goals"].append(goals)
    state["career_appearances"] += 1
    state["career_goal_total"] += goals
    state["appearance_dates"].append(date)

    # Goal EWMA
    state["goal_ewma"] = (
        GOAL_EWMA_ALPHA * goals
        + (1.0 - GOAL_EWMA_ALPHA) * state["goal_ewma"]
    )

    # Scoring / goalless streak
    if goals > 0:
        state["consecutive_scoring"] += 1
        state["goalless_streak"] = 0
        state["last_goal_date"] = date
    else:
        state["consecutive_scoring"] = 0
        state["goalless_streak"] += 1

    # Season
    season_s = state["season_state"].setdefault(
        season,
        {"apps": 0, "goals": 0},
    )
    season_s["apps"] += 1
    season_s["goals"] += goals

    # Team
    team_s = state["team_state"].setdefault(
        team,
        {"apps": 0, "goals": 0},
    )
    team_s["apps"] += 1
    team_s["goals"] += goals

    # Opponent
    opponent_s = state["opponent_state"].setdefault(
        opponent,
        {
            "apps": 0,
            "goals": 0,
            "goal_history": [],
        },
    )
    opponent_s["apps"] += 1
    opponent_s["goals"] += goals
    opponent_s["goal_history"].append(goals)

    # Competition
    competition_s = state["competition_state"].setdefault(
        competition,
        {"apps": 0, "goals": 0},
    )
    competition_s["apps"] += 1
    competition_s["goals"] += goals

    # Venue
    venue_s = state["venue_state"].setdefault(
        venue,
        {"apps": 0, "goals": 0},
    )
    venue_s["apps"] += 1
    venue_s["goals"] += goals

    return state




def replay_match_updates(
    initial_state,
    future_fixtures,
    match_updates,
):
    """Replay completed real fixtures from the frozen forecast cutoff."""

    state = copy.deepcopy(initial_state)

    if match_updates.empty:
        return state, set()

    required_columns = {"fixture_id", "appeared", "goals"}
    missing = required_columns - set(match_updates.columns)

    if missing:
        raise ValueError(
            f"Match updates missing required columns: {sorted(missing)}"
        )

    fixtures = future_fixtures.copy()
    fixtures["date"] = pd.to_datetime(fixtures["date"])

    updates = match_updates.copy()

    if updates["fixture_id"].duplicated().any():
        duplicates = updates.loc[
            updates["fixture_id"].duplicated(),
            "fixture_id",
        ].tolist()

        raise ValueError(
            f"Duplicate fixture updates found: {duplicates}"
        )

    fixture_lookup = fixtures.set_index("fixture_id")

    unknown_ids = set(updates["fixture_id"]) - set(fixture_lookup.index)

    if unknown_ids:
        raise ValueError(
            f"Unknown fixture IDs in match updates: {sorted(unknown_ids)}"
        )

    # Replay in chronological fixture order rather than CSV row order.
    updates = updates.merge(
        fixtures[
            [
                "fixture_id",
                "date",
                "team",
                "opponent",
                "competition",
                "venue",
            ]
        ],
        on="fixture_id",
        how="left",
        validate="one_to_one",
    ).sort_values("date")

    completed_fixture_ids = set()

    for row in updates.itertuples(index=False):
        appeared = int(row.appeared)
        goals = int(row.goals)

        if appeared not in (0, 1):
            raise ValueError(
                f"{row.fixture_id}: appeared must be 0 or 1"
            )

        if goals < 0:
            raise ValueError(
                f"{row.fixture_id}: goals cannot be negative"
            )

        if appeared == 0 and goals != 0:
            raise ValueError(
                f"{row.fixture_id}: goals must be 0 when appeared is 0"
            )

        completed_fixture_ids.add(row.fixture_id)

        # A missed fixture advances the schedule, but not Ronaldo's
        # appearance-dependent recursive state.
        if appeared == 0:
            continue

        state = update_state_after_appearance(
            state,
            date=row.date,
            season="2026/27",
            team=row.team,
            opponent=row.opponent,
            competition=row.competition,
            venue=row.venue,
            goals=goals,
        )

    return state, completed_fixture_ids



def get_next_uncompleted_fixture(
    future_fixtures,
    completed_fixture_ids,
):
    """Return the earliest fixture that has not been recorded as completed."""

    fixtures = future_fixtures.copy()
    fixtures["date"] = pd.to_datetime(fixtures["date"])

    fixtures = fixtures.sort_values("date")

    remaining = fixtures[
        ~fixtures["fixture_id"].isin(completed_fixture_ids)
    ]

    if remaining.empty:
        return None

    return remaining.iloc[0]



def build_completed_fixture_states(
    initial_state,
    future_fixtures,
    match_updates,
):
    """
    Reconstruct the exact pre-match recursive state for every
    completed fixture.

    Returns a list of dictionaries containing:
    - fixture
    - pre_match_state
    - appeared
    - goals
    """

    fixtures = future_fixtures.copy()
    fixtures["date"] = pd.to_datetime(fixtures["date"])

    updates = match_updates.copy()

    if updates.empty:
        return []

    merged = updates.merge(
        fixtures,
        on="fixture_id",
        how="left",
        validate="one_to_one",
    ).sort_values("date")

    state = copy.deepcopy(initial_state)
    history = []

    for _, row in merged.iterrows():
        appeared = int(row["appeared"])
        goals = int(row["goals"])

        history.append(
            {
                "fixture": row.copy(),
                "pre_match_state": copy.deepcopy(state),
                "appeared": appeared,
                "goals": goals,
            }
        )

        if appeared == 0:
            continue

        state = update_state_after_appearance(
            state,
            date=row["date"],
            season="2026/27",
            team=row["team"],
            opponent=row["opponent"],
            competition=row["competition"],
            venue=row["venue"],
            goals=goals,
        )

    return history