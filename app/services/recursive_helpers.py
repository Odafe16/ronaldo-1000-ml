# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 10:06:24 2026

@author: JONATHAN
"""

def safe_rate(numerator, denominator):
    return (
        float(numerator) / float(denominator)
        if denominator > 0
        else 0.0
    )


def rolling_sum(values, window):
    vals = list(values)[-window:]
    return float(sum(vals))


def rolling_mean(values, window):
    vals = list(values)[-window:]

    return (
        float(sum(vals)) / len(vals)
        if len(vals) > 0
        else 0.0
    )


def rolling_scoring_rate(values, window):
    vals = list(values)[-window:]

    return (
        float(sum(g > 0 for g in vals)) / len(vals)
        if len(vals) > 0
        else 0.0
    )


def rolling_multi_goal_rate(values, window):
    vals = list(values)[-window:]

    return (
        float(sum(g >= 2 for g in vals)) / len(vals)
        if len(vals) > 0
        else 0.0
    )