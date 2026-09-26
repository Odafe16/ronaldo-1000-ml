# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 08:53:18 2026

@author: JONATHAN
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

from app.services.fixture_features import (
    load_fixture_artifacts,
    build_fixture_features,
)
from app.services.goal_predictor import GoalPredictor

import altair as alt



FORECAST_DIR = ROOT / "outputs" / "forecast"

st.set_page_config(
    page_title="Ronaldo | Road to 1000",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data
def load_forecast():
    summary = pd.read_csv(
        FORECAST_DIR / "appearance_sensitivity_summary.csv"
    )

    monthly = pd.read_csv(
        FORECAST_DIR / "monthly_milestone_probabilities.csv"
    )

    return summary, monthly


forecast, monthly_forecast = load_forecast()

st.sidebar.title("RONALDO → 1000")

scenario = st.sidebar.selectbox(
    "Forecast scenario",
    ["Baseline", "Conservative", "Optimistic"],
)

selected = forecast.loc[
    forecast["scenario"] == scenario
].iloc[0]

st.title("Cristiano Ronaldo Road to 1,000 Goals ⚽")

st.markdown(
    """
    An ML-powered forecast tracking Cristiano Ronaldo's journey
    toward **1,000 career goals**.
    """
)

st.caption(
    "Historical data cutoff: 28 August 2026 · "
    "Forecast horizon: 28 May 2027"
)

st.divider()

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Career Goals",
    "978",
)

col2.metric(
    "Goals Remaining",
    "22",
)

col3.metric(
    "Milestone Target",
    "1,000",
)

col4.metric(
    "Progress",
    "97.8%",
)

st.progress(978 / 1000, text="978 of 1,000 career goals")

st.divider()

st.subheader("Milestone Forecast")

st.caption(
    f"{scenario} scenario · 1,000 Monte Carlo simulations"
)

completion_probability = selected["completion_probability"] * 100

median_date = pd.to_datetime(
    selected["median_date"]
).strftime("%d %B %Y")


# Main forecast results
col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Probability of Reaching 1,000",
        f"{completion_probability:.1f}%",
    )

with col2:
    st.metric(
        "Median Milestone Date",
        median_date,
    )


# Supporting simulation information
completed = int(selected["completed"])
simulations = int(selected["simulations"])

st.caption(
    f"{completed:,} of {simulations:,} simulated career paths reached "
    "1,000 goals within the forecast window."
)

st.info(
    "The forecast window ends on 28 May 2027. "
    "The median milestone date is calculated only from simulations "
    "that reached 1,000 goals within that period."
)


st.divider()

st.subheader("Road to 1,000")

st.write(
    "Explore the cumulative probability of Ronaldo reaching "
    "1,000 career goals by each monthly deadline."
)

road_chart = monthly_forecast.copy()

road_chart["deadline"] = pd.to_datetime(
    road_chart["deadline"]
)

road_chart["probability_pct"] = (
    road_chart["probability"] * 100
)

road_chart = road_chart.sort_values("deadline")

scenario_selection = alt.selection_point(
    fields=["scenario"],
    value=scenario,
)

line_chart = (
    alt.Chart(road_chart)
    .mark_line(point=True)
    .encode(
        x=alt.X(
            "deadline:T",
            title="Forecast deadline",
        ),
        y=alt.Y(
            "probability_pct:Q",
            title="Probability of reaching 1,000 (%)",
            scale=alt.Scale(domain=[0, 100]),
        ),
        color=alt.Color(
            "scenario:N",
            title="Scenario",
        ),
        opacity=alt.condition(
            scenario_selection,
            alt.value(1.0),
            alt.value(0.25),
        ),
        strokeWidth=alt.condition(
            scenario_selection,
            alt.value(4),
            alt.value(2),
        ),
        tooltip=[
            alt.Tooltip(
                "deadline:T",
                title="Deadline",
                format="%d %b %Y",
            ),
            alt.Tooltip(
                "scenario:N",
                title="Scenario",
            ),
            alt.Tooltip(
                "probability_pct:Q",
                title="Probability",
                format=".1f",
            ),
        ],
    )
    .add_params(scenario_selection)
    .properties(height=400)
)

st.altair_chart(
    line_chart,
    use_container_width=True,
)
st.caption(
    "Each line represents an appearance scenario. "
    "Probabilities are cumulative and apply only to the "
    "modeled fixture calendar ending 28 May 2027."
)


# =========================================================
# NEXT FIXTURE PREDICTION
# =========================================================

st.divider()

st.subheader("Next Fixture Prediction")

st.caption(
    "Pre-match forecast using the historical data cutoff "
    "of 28 August 2026. This is a frozen prediction, "
    "not a live match update."
)


@st.cache_data
def load_future_fixtures():
    return pd.read_csv(
        FORECAST_DIR / "future_fixtures.csv"
    )


@st.cache_resource
def load_goal_predictor():
    return GoalPredictor()


@st.cache_data
def load_prediction_artifacts():
    return load_fixture_artifacts()


future_fixtures = load_future_fixtures()

# Only the first fixture has been validated against the
# original notebook forecast so far.
fixture = future_fixtures.iloc[4]

state, static_cache = load_prediction_artifacts()

fixture_input = build_fixture_features(
    fixture=fixture,
    state=state,
    static_cache=static_cache,
)

predictor = load_goal_predictor()
prediction = predictor.predict(fixture_input)

fixture_date = pd.to_datetime(
    fixture["date"]
).strftime("%d %B %Y")

st.markdown(
    f"## {fixture['team']} vs {fixture['opponent_display']}"
)

venue_labels = {
    "H": "Home",
    "A": "Away",
    "N": "Neutral",
}

venue_display = venue_labels.get(
    fixture["venue"],
    fixture["venue"],
)

st.caption(
    f"{fixture_date} · "
    f"{fixture['competition']} · "
    f"{venue_display}"
)

st.markdown("#### Goal Forecast")

col1, col2, col3 = st.columns([1.4, 1, 1])

col1.metric(
    "Probability of Scoring",
    f"{prediction['p_scores']:.1%}",
)

col2.metric(
    "Expected Goals",
    f"{prediction['expected_goals']:.2f}",
)

col3.metric(
    "Probability of 2+ Goals",
    f"{1 - prediction['p_0'] - prediction['p_1']:.1%}",
)

st.caption(
    "Goal probabilities are conditional on Ronaldo appearing in the match."
)


goal_distribution = pd.DataFrame({
    "Goals": ["0", "1", "2", "3+"],
    "Probability": [
        prediction["p_0"] * 100,
        prediction["p_1"] * 100,
        prediction["p_2"] * 100,
        prediction["p_3_plus"] * 100,
    ],
})

goal_chart = (
    alt.Chart(goal_distribution)
    .mark_bar()
    .encode(
        x=alt.X(
            "Goals:N",
            title="Goals scored",
            sort=["0", "1", "2", "3+"],
        ),
        y=alt.Y(
            "Probability:Q",
            title="Probability (%)",
        ),
        tooltip=[
            alt.Tooltip(
                "Goals:N",
                title="Goals",
            ),
            alt.Tooltip(
                "Probability:Q",
                title="Probability",
                format=".1f",
            ),
        ],
    )
    .properties(
        height=280,
        title="Predicted Goal Distribution",
    )
)

st.altair_chart(
    goal_chart,
    use_container_width=True,
)


st.caption(
    "Goal probabilities use a Poisson distribution with "
    "the XGBoost–CatBoost ensemble's expected goals. "
    "Predictions are conditional on Ronaldo appearing."
)


#fixture by fixture milestone forecast chart

st.divider()
st.subheader("Fixture-by-Fixture Milestone Forecast — Pilot Simulation")
st.markdown(
    """
    This view shows **when the 1,000th goal occurred across the saved
    1,000-path pilot simulation**, mapped onto Ronaldo's 45 projected
    fixtures through 28 May 2027.
    """
)

st.info(
    "This fixture-level view comes from the earlier pilot Monte Carlo run "
    "(949 of 1,000 paths reached the milestone). It is separate from the "
    "Conservative, Baseline, and Optimistic sensitivity experiment above."
)

st.info(
    "This fixture-level view is derived from the earlier 1,000-path "
    "pilot Monte Carlo run. It is separate from the Baseline, "
    "Conservative, and Optimistic sensitivity experiment shown above."
)

@st.cache_data
def load_fixture_milestone_forecast():
    data = pd.read_csv(
        FORECAST_DIR / "fixture_milestone_probabilities.csv",
        parse_dates=["date"],
    )

    assert len(data) == 45, "Expected 45 forecast fixtures."

    assert data["milestone_count"].sum() == 949, (
        "Milestone counts do not match the saved simulations."
    )

    assert abs(
        data["cumulative_milestone_probability"].iloc[-1] - 0.949
    ) < 1e-9, "Final cumulative probability is incorrect."

    return data


fixture_forecast = load_fixture_milestone_forecast()

most_frequent_fixture = fixture_forecast.loc[
    fixture_forecast["milestone_count"].idxmax()
]

st.markdown("#### Most Frequent Simulated Milestone Fixture")

col1, col2 = st.columns([1.5, 1])

with col1:
    st.metric(
        "Opponent",
        most_frequent_fixture["opponent"],
    )

with col2:
    st.metric(
        "Fixture Date",
        f"{most_frequent_fixture['date']:%d %b %Y}",
    )

st.caption(
    f"Ronaldo reached 1,000 goals at this fixture in "
    f"{int(most_frequent_fixture['milestone_count'])} of the "
    f"1,000 saved pilot simulations "
    f"({most_frequent_fixture['milestone_probability']:.1%}). "
    "This identifies the most frequent simulated milestone fixture, "
    "not a guaranteed outcome."
)

# 1. Prepare the fixture selector
fixture_options = fixture_forecast.copy()

fixture_options["fixture_label"] = (
    fixture_options["date"].dt.strftime("%d %b %Y")
    + " — "
    + fixture_options["opponent"]
)

# 2. Let the user select a fixture
st.markdown("#### Explore a Fixture")

selected_index = st.selectbox(
    "Select a fixture to inspect its milestone probability",
    options=fixture_options.index,
    format_func=lambda i: fixture_options.loc[i, "fixture_label"],
    help=(
        "This selector changes only the fixture-level milestone simulation "
        "below. It does not change the Next Fixture goal prediction above."
    ),
)

# 3. Get the selected fixture BEFORE creating the chart
selected_fixture = fixture_options.loc[selected_index]


# 4. Prepare chart data
chart_source = fixture_forecast.copy()

chart_source["probability_pct"] = (
    chart_source["cumulative_milestone_probability"] * 100
)

# 5. Draw the cumulative probability line
# 5. Draw the cumulative probability line
line = (
    alt.Chart(chart_source)
    .mark_line(point=True)
    .encode(
        x=alt.X(
            "date:T",
            title="Fixture date",
        ),
        y=alt.Y(
            "probability_pct:Q",
            title="Cumulative milestone probability (%)",
            scale=alt.Scale(domain=[0, 100]),
        ),
        tooltip=[
            alt.Tooltip(
                "date:T",
                title="Fixture date",
                format="%d %b %Y",
            ),
            alt.Tooltip(
                "opponent:N",
                title="Opponent",
            ),
            alt.Tooltip(
                "milestone_probability:Q",
                title="Milestone at fixture",
                format=".1%",
            ),
            alt.Tooltip(
                "probability_pct:Q",
                title="Milestone by fixture",
                format=".1f",
            ),
        ],
    )
    .properties(
        height=380,
        title="Cumulative Probability of Reaching 1,000",
    )
)

# 6. Highlight the selected fixture
selected_point = chart_source.loc[
    chart_source["fixture_index"] == selected_fixture["fixture_index"]
]

highlight = alt.Chart(selected_point).mark_circle(
    size=140,
    color="red",
).encode(
    x="date:T",
    y="probability_pct:Q",
   tooltip=[
    alt.Tooltip("date:T", title="Date"),
    alt.Tooltip("opponent:N", title="Opponent"),
    alt.Tooltip(
        "milestone_probability:Q",
        title="Milestone at this fixture",
        format=".1%",
    ),
    alt.Tooltip(
        "probability_pct:Q",
        title="Milestone by this fixture (%)",
        format=".1f",
    ),
],
)
    
selected_rule = (
    alt.Chart(selected_point)
    .mark_rule(
        strokeDash=[5, 5],
        opacity=0.5,
    )
    .encode(
        x="date:T",
    )
)

st.altair_chart(
    line + selected_rule + highlight,
    use_container_width=True,
)

st.caption(
    "Cumulative probability of Ronaldo reaching 1,000 career goals "
    "by each fixture, based on 1,000 saved pilot Monte Carlo simulations. "
    "This is not the probability of scoring in an individual match."
)

# 7. Display the selected fixture's probabilities
cumulative_count = fixture_forecast.loc[
    fixture_forecast["fixture_index"]
    <= selected_fixture["fixture_index"],
    "milestone_count",
].sum()

st.markdown("##### At This Fixture")

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Milestone Probability",
        f"{selected_fixture['milestone_probability']:.1%}",
    )

with col2:
    st.metric(
        "Simulation Paths",
        f"{int(selected_fixture['milestone_count'])} / 1,000",
    )

st.markdown("##### By This Fixture")

col3, col4 = st.columns(2)

with col3:
    st.metric(
        "Cumulative Milestone Probability",
        f"{selected_fixture['cumulative_milestone_probability']:.1%}",
    )

with col4:
    st.metric(
        "Cumulative Simulation Paths",
        f"{int(cumulative_count)} / 1,000",
    )    


table_data = fixture_forecast[
    [
        "date",
        "opponent",
        "milestone_count",
        "milestone_probability",
        "cumulative_milestone_probability",
    ]
].copy()

table_data["cumulative_milestone_count"] = (
    table_data["milestone_count"].cumsum()
)

with st.expander("View Complete Fixture Forecast Table"):
    st.dataframe(
        table_data[
    [
        "date",
        "opponent",
        "milestone_count",
        "milestone_probability",
        "cumulative_milestone_count",
        "cumulative_milestone_probability",
    ]
],
        column_config={
            "date": st.column_config.DateColumn(
                "Fixture Date",
                format="DD MMM YYYY",
            ),
            "opponent": "Opponent",
            "milestone_count": "Milestone Simulations",
            "milestone_probability": st.column_config.ProgressColumn(
                "Milestone at Fixture",
                min_value=0,
                max_value=1,
                format="percent",
            ),
            "cumulative_milestone_probability": st.column_config.ProgressColumn(
                "Milestone by Fixture",
                min_value=0,
                max_value=1,
                format="percent",
            ),
            "cumulative_milestone_count": "Reached 1,000 By Here",
        },
        hide_index=True,
        use_container_width=True,
    )
    
st.caption(
    "At this fixture = probability of reaching 1,000 goals specifically "
    "in the selected match. By this fixture = probability of reaching "
    "1,000 goals on or before the selected match. "
    "Both use all 1,000 saved pilot simulations."
)



st.divider()

st.subheader("About the Forecast")

st.markdown(
    """
    This project combines two machine-learning components:

    - **Goal model:** a 60% XGBoost and 40% CatBoost ensemble estimates
      Ronaldo's expected goals when he appears in a fixture.
    - **Appearance model:** estimates the probability that Ronaldo appears
      in each eligible fixture.

    Goal counts are modeled using a **Poisson distribution**, while Monte
    Carlo simulation propagates appearance and scoring uncertainty across
    the future fixture schedule until Ronaldo reaches 1,000 career goals
    or the forecast window ends.

    All predictions use information available through **28 August 2026**.
    """
)

st.caption(
    "Forecasts are probabilistic model estimates, not guarantees. "
    "Future injuries, transfers, fixture changes, team selection, and other "
    "events after the historical cutoff are not incorporated."
)