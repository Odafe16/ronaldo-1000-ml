# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 10:09:02 2026

@author: JONATHAN
"""

import numpy as np
import pandas as pd

from app.services.recursive_helpers import (
    safe_rate,
    rolling_sum,
    rolling_mean,
    rolling_scoring_rate,
    rolling_multi_goal_rate,
)


def build_recursive_features(
    state,
    date,
    season,
    team,
    opponent,
    competition,
    venue,
):
    """Build the 35 recursive pre-match features without changing state."""

    date = pd.Timestamp(date)

    career_history = state["career_goals"]
    career_apps = state["career_appearances"]
    career_goal_total = state["career_goal_total"]

    season_s = state["season_state"].get(
        season, {"apps": 0, "goals": 0}
    )
    team_s = state["team_state"].get(
        team, {"apps": 0, "goals": 0}
    )
    opponent_s = state["opponent_state"].get(
        opponent, {"apps": 0, "goals": 0, "goal_history": []}
    )
    competition_s = state["competition_state"].get(
        competition, {"apps": 0, "goals": 0}
    )
    venue_s = state["venue_state"].get(
        venue, {"apps": 0, "goals": 0}
    )

    features = {
        "career_appearances_pre": career_apps,
        "career_goals_pre": career_goal_total,
        "career_goals_per_game_pre": safe_rate(
            career_goal_total, career_apps
        ),
        "no_prior_career_goal_pre": int(career_goal_total == 0),
        "first_career_appearance": int(career_apps == 0),

        "goals_last_3": rolling_sum(career_history, 3),
        "goals_per_game_last_5": rolling_mean(career_history, 5),
        "goals_per_game_last_10": rolling_mean(career_history, 10),
        "goals_per_game_last_20": rolling_mean(career_history, 20),
        "goals_ewma_pre": float(state["goal_ewma"]),
        "scoring_rate_last_5": rolling_scoring_rate(career_history, 5),
        "scoring_rate_last_10": rolling_scoring_rate(career_history, 10),
        "multi_goal_rate_last_10": rolling_multi_goal_rate(
            career_history, 10
        ),
        "consecutive_scoring_games_pre": int(
            state["consecutive_scoring"]
        ),
        "goalless_streak_games_pre": int(state["goalless_streak"]),
        "days_since_last_goal": (
            0.0
            if state["last_goal_date"] is None
            else float((date - state["last_goal_date"]).days)
        ),

        "season_appearances_pre": season_s["apps"],
        "season_goals_pre": season_s["goals"],
        "season_goals_per_game_pre": safe_rate(
            season_s["goals"], season_s["apps"]
        ),

        "team_appearances_with_ronaldo_pre": team_s["apps"],
        "team_goals_by_ronaldo_pre": team_s["goals"],
        "team_ronaldo_goals_per_game_pre": safe_rate(
            team_s["goals"], team_s["apps"]
        ),

        "opponent_prior_meetings_pre": opponent_s["apps"],
        "ronaldo_goals_vs_opponent_pre": opponent_s["goals"],
        "ronaldo_goals_per_game_vs_opponent_pre": safe_rate(
            opponent_s["goals"], opponent_s["apps"]
        ),
        "ronaldo_scoring_rate_vs_opponent_pre": safe_rate(
            sum(g > 0 for g in opponent_s["goal_history"]),
            opponent_s["apps"],
        ),
        "ronaldo_goals_last_3_vs_opponent_pre": rolling_sum(
            opponent_s["goal_history"], 3
        ),
        "first_time_opponent_pre": int(opponent_s["apps"] == 0),
        "opponent_meetings_log_pre": float(
            np.log1p(opponent_s["apps"])
        ),

        "competition_appearances_pre": competition_s["apps"],
        "competition_goals_pre": competition_s["goals"],
        "competition_goals_per_game_pre": safe_rate(
            competition_s["goals"], competition_s["apps"]
        ),

        "venue_appearances_pre": venue_s["apps"],
        "venue_goals_pre": venue_s["goals"],
        "venue_goals_per_game_pre": safe_rate(
            venue_s["goals"], venue_s["apps"]
        ),
    }

    return features