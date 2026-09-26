# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 10:01:55 2026

@author: JONATHAN
"""

import pandas as pd


def calculate_days_rest(date, appearance_dates):
    date = pd.Timestamp(date)

    previous_dates = [
        pd.Timestamp(d)
        for d in appearance_dates
        if pd.Timestamp(d) < date
    ]

    if not previous_dates:
        return 0.0

    return float((date - max(previous_dates)).days)


def count_recent_appearances(date, appearance_dates, window_days):
    date = pd.Timestamp(date)
    start = date - pd.Timedelta(days=window_days)

    return sum(
        start <= pd.Timestamp(d) < date
        for d in appearance_dates
    )


def build_schedule_features(date, appearance_dates):
    return {
        "days_rest": calculate_days_rest(date, appearance_dates),
        "matches_last_7d": count_recent_appearances(
            date, appearance_dates, 7
        ),
        "matches_last_14d": count_recent_appearances(
            date, appearance_dates, 14
        ),
        "matches_last_30d": count_recent_appearances(
            date, appearance_dates, 30
        ),
    }