# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 09:35:32 2026

@author: JONATHAN
"""

import json
from pathlib import Path

import joblib
import numpy as np
from catboost import CatBoostRegressor
from xgboost import XGBRegressor


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = ROOT / "artifacts"


class GoalPredictor:
    def __init__(self):
        with open(
            ARTIFACT_DIR / "preprocessing_spec.json",
            encoding="utf-8",
        ) as file:
            preprocessing_spec = json.load(file)

        with open(
            ARTIFACT_DIR / "feature_spec.json",
            encoding="utf-8",
        ) as file:
            feature_spec = json.load(file)

        with open(
            ARTIFACT_DIR / "main_model_spec.json",
            encoding="utf-8",
        ) as file:
            model_spec = json.load(file)

        self.full_features = preprocessing_spec["feature_sets"]["full"]
        self.context_features = preprocessing_spec["feature_sets"]["context"]
        self.categorical_features = feature_spec["categorical_features"]

        self.weights = model_spec["selected_ensemble"]

        self.preprocessor = joblib.load(
            ARTIFACT_DIR / "goal_xgboost_preprocessor.joblib"
        )

        self.xgboost = XGBRegressor()
        self.xgboost.load_model(
            str(ARTIFACT_DIR / "goal_xgboost_production.json")
        )

        self.catboost = CatBoostRegressor()
        self.catboost.load_model(
            str(ARTIFACT_DIR / "goal_catboost_production.cbm")
        )

    def predict(self, fixture):
        missing = [
            feature
            for feature in self.full_features
            if feature not in fixture.columns
        ]

        if missing:
            raise ValueError(
                f"Missing required features: {missing}"
            )

        xgb_input = fixture[self.context_features].copy()
        xgb_input = self.preprocessor.transform(xgb_input)

        cat_input = fixture[self.full_features].copy()

        for column in self.categorical_features:
            cat_input[column] = cat_input[column].astype(str)

        lambda_xgb = float(self.xgboost.predict(xgb_input)[0])
        lambda_cat = float(self.catboost.predict(cat_input)[0])

        expected_goals = (
            self.weights["xgboost_weight"] * lambda_xgb
            + self.weights["catboost_weight"] * lambda_cat
        )

        if not np.isfinite(expected_goals) or expected_goals <= 0:
            raise ValueError("Model returned invalid expected goals.")

        p0 = np.exp(-expected_goals)
        p1 = expected_goals * p0
        p2 = expected_goals**2 / 2 * p0

        return {
            "expected_goals": expected_goals,
            "xgboost_lambda": lambda_xgb,
            "catboost_lambda": lambda_cat,
            "p_0": p0,
            "p_1": p1,
            "p_2": p2,
            "p_3_plus": max(0.0, 1 - p0 - p1 - p2),
            "p_scores": 1 - p0,
        }