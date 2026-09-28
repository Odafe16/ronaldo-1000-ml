# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 09:27:07 2026

@author: JONATHAN
"""

import pandas as pd

from app.services.fixture_features import build_fixture_features
from app.services.state_updater import build_completed_fixture_states


def build_prediction_history(
    initial_state,
    static_cache,
    future_fixtures,
    match_updates,
    predictor,
):
    completed = build_completed_fixture_states(
        initial_state,
        future_fixtures,
        match_updates,
    )

    rows = []

    for item in completed:
        fixture = item["fixture"]
        appeared = item["appeared"]
        actual_goals = item["goals"]

        features = build_fixture_features(
            fixture=fixture,
            state=item["pre_match_state"],
            static_cache=static_cache,
        )

        prediction = predictor.predict(features)

        rows.append(
            {
                "fixture_id": fixture["fixture_id"],
                "date": pd.Timestamp(fixture["date"]),
                "team": fixture["team"],
                "opponent": fixture["opponent"],
                "opponent_display": fixture["opponent_display"],
                "competition": fixture["competition"],
                "venue": fixture["venue"],
                "appeared": appeared,
                "actual_goals": actual_goals,
                "expected_goals": prediction["expected_goals"],
                "p_scores": prediction["p_scores"],
                "p_0": prediction["p_0"],
                "p_1": prediction["p_1"],
                "p_2": prediction["p_2"],
                "p_3_plus": prediction["p_3_plus"],
            }
        )

    return pd.DataFrame(rows)


def calculate_prediction_performance(history):
    evaluated = history[history["appeared"] == 1].copy()

    if evaluated.empty:
        return None

    matches = len(evaluated)

    actual_goals = int(evaluated["actual_goals"].sum())
    expected_goals = float(evaluated["expected_goals"].sum())

    actual_scoring_rate = float(
        (evaluated["actual_goals"] > 0).mean()
    )

    average_scoring_probability = float(
        evaluated["p_scores"].mean()
    )

    return {
        "matches": matches,
        "actual_goals": actual_goals,
        "expected_goals": expected_goals,
        "actual_scoring_rate": actual_scoring_rate,
        "average_scoring_probability": average_scoring_probability,
    }