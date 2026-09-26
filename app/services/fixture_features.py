# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 10:12:16 2026

@author: JONATHAN
"""

import pickle
from pathlib import Path

import pandas as pd

from app.services.goal_predictor import GoalPredictor
from app.services.recursive_features import build_recursive_features
from app.services.schedule_features import build_schedule_features


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = ROOT / "artifacts"


def load_fixture_artifacts():
    with open(ARTIFACT_DIR / "initial_recursive_state.pkl", "rb") as file:
        state = pickle.load(file)

    with open(ARTIFACT_DIR / "goal_static_cache.pkl", "rb") as file:
        static_cache = pickle.load(file)

    return state, static_cache


def build_fixture_features(fixture, state, static_cache):
    fixture_id = fixture["fixture_id"]
    fixture_date = pd.Timestamp(fixture["date"])

    if fixture_id not in static_cache:
        raise ValueError(f"No static features for fixture: {fixture_id}")

    recursive = build_recursive_features(
        state=state,
        date=fixture_date,
        season="2026/27",
        team=fixture["team"],
        opponent=fixture["opponent"],
        competition=fixture["competition"],
        venue=fixture["venue"],
    )

    scheduling = build_schedule_features(
        fixture_date,
        state["appearance_dates"],
    )

    feature_row = {
        **static_cache[fixture_id],
        **scheduling,
        **recursive,
    }

    predictor = GoalPredictor()

    missing = set(predictor.full_features) - set(feature_row)
    unexpected = set(feature_row) - set(predictor.full_features)

    if missing or unexpected:
        raise ValueError(
            f"Feature mismatch. Missing: {missing}; "
            f"Unexpected: {unexpected}"
        )

    return pd.DataFrame(
        [feature_row],
        columns=predictor.full_features,
    )