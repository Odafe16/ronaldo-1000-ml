# Cristiano Ronaldo — Road to 1,000 Goals ⚽

An end-to-end machine learning project forecasting Cristiano Ronaldo's journey toward **1,000 career goals**.

The project models both **goal scoring** and **match appearance probability**, then combines them through Monte Carlo simulation to estimate when Ronaldo could reach the 1,000-goal milestone.

> **Historical data cutoff:** 28 August 2026  
> **Goals at cutoff:** 978  
> **Goals remaining:** 22  
> **Forecast horizon:** 28 May 2027

## Live Project

**Interactive Dashboard:**  
https://cr7milestoneforecast.streamlit.app

The dashboard provides:

- Goal predictions for the next projected fixture
- Probability distributions for 0, 1, 2, and 3+ goals
- Conservative, Baseline, and Optimistic appearance scenarios
- Monthly milestone probabilities
- Fixture-by-fixture milestone analysis
- Monte Carlo estimates of the 1,000th-goal date


## Project Overview

Cristiano Ronaldo entered the forecast period with **978 official career goals**, leaving him 22 goals away from the 1,000-goal milestone.

Rather than simply extrapolating his historical scoring rate, this project treats the problem as two connected uncertainties:

1. **Will Ronaldo appear in the fixture?**
2. **If he appears, how many goals will he score?**

Separate machine learning components model these two processes. Their predictions are then combined with the projected fixture schedule in a Monte Carlo simulation that generates possible future career paths.

The goal model predicts Ronaldo's expected goals per appearance using an ensemble of:

- **XGBoost — 60%**
- **CatBoost — 40%**

The predicted expected-goals value is converted into probabilities for different goal counts using a **Poisson distribution**.

The appearance model separately estimates Ronaldo's probability of participating in each eligible fixture.

Together, these models allow the simulation to account for both **appearance uncertainty** and **goal-scoring uncertainty** instead of assuming Ronaldo plays and scores at a fixed rate.



## Model Performance

The goal-scoring models were evaluated using a **temporal holdout**, preserving the chronological nature of the problem rather than randomly mixing past and future matches.

The final held-out test period contained **78 appearances** from **9 January 2025 to 28 August 2026**, during which Ronaldo scored **62 goals**.

| Model | Poisson Deviance | MAE | RMSE |
|---|---:|---:|---:|
| XGBoost | 0.8177 | 0.5946 | 0.6995 |
| CatBoost | 0.7949 | 0.5461 | 0.6856 |
| Ensemble | 0.8032 | 0.5752 | 0.6910 |

The production forecast uses a **60% XGBoost / 40% CatBoost ensemble**.

Across the held-out period:

- Observed scoring rate: **0.795 goals per appearance**
- Ensemble predicted rate: **0.806 goals per appearance**
- Predicted probability of scoring: **55.3%**
- Observed scoring frequency: **62.8%**
- Predicted probability of 2+ goals: **19.4%**
- Observed 2+ goal frequency: **16.7%**

The out-of-fold Poisson dispersion was approximately **0.979**, supporting the use of a Poisson count model for the simulation layer.

## Milestone Forecast

Starting from **978 career goals**, the simulation projects Ronaldo's career across **45 scheduled fixtures** through 28 May 2027.

Under the **Baseline appearance scenario**:

- **942 of 1,000 simulations** reached 1,000 goals within the forecast window
- Estimated completion probability: **94.2%**
- Median milestone date among successful simulations: **25 February 2027**

The project also evaluates **Conservative** and **Optimistic** appearance scenarios to measure how changes in Ronaldo's availability affect the milestone forecast.

> The 94.2% Baseline result comes from the controlled appearance-sensitivity experiment. The dashboard's fixture-by-fixture view uses a separate earlier 1,000-path pilot simulation, in which 949 paths reached the milestone. These simulation runs are presented separately rather than treated as identical results.


## Data & Feature Engineering

The modeling dataset is built around Ronaldo's historical match appearances, with each row representing a match in which he appeared.

The final career appearance index contains **1,333 appearances** from **14 August 2002 through 28 August 2026**, covering both club and international football.

Across those appearances, Ronaldo scored **978 goals**.

### Historical Coverage

The dataset covers appearances for:

- Sporting CP
- Manchester United
- Real Madrid
- Juventus
- Al-Nassr
- Portugal

It includes domestic leagues, domestic and international cups, continental club competitions, and international tournaments.

### Feature Design

The models use historical and match-context information available **before each fixture**.

Feature groups include:

- Match and competition context
- Team and opponent information
- Venue
- Ronaldo's historical scoring form
- Recent goal-scoring performance
- Career and team appearance history
- Team recent form
- Opponent-related historical performance
- Head-to-head information
- Schedule and fixture context
- Rolling and expanding historical statistics

### Preventing Data Leakage

Because this is a forecasting problem, feature construction follows a strict chronological process.

Rolling statistics, scoring form, team performance, opponent history, and other historical features are calculated using only information available **before the match being predicted**.

The project therefore avoids using future match outcomes when constructing historical training examples.

Model evaluation also uses **temporal splits** rather than random train/test splitting so that later matches are predicted from earlier information.

For future forecasting, recursive state is updated chronologically as simulated fixtures are processed, allowing future features to evolve without introducing information from matches that have not yet occurred.


## Monte Carlo Forecasting

Predicting goals for a single match is only one part of the problem. Reaching 1,000 career goals depends on what happens across an entire sequence of future fixtures.

The project therefore uses **Monte Carlo simulation** to propagate uncertainty across Ronaldo's projected schedule.

For each future fixture, the simulation:

1. Estimates Ronaldo's probability of appearing.
2. Simulates whether he appears in the match.
3. If he appears, constructs the required pre-match features from the current historical state.
4. Uses the XGBoost–CatBoost ensemble to estimate expected goals (`λ`).
5. Samples an integer goal count from a **Poisson distribution**.
6. Updates Ronaldo's career goals, appearances, and recursive historical features.
7. Continues chronologically to the next fixture.
8. Stops when Ronaldo reaches 1,000 goals or the forecast schedule ends.

This produces many possible future career paths rather than a single deterministic prediction.

### Appearance Scenarios

To examine the effect of availability, the main forecasting experiment runs **1,000 simulations per scenario** under three appearance assumptions:

| Scenario | Appearance Adjustment |
|---|---:|
| Conservative | -5 percentage points |
| Baseline | No adjustment |
| Optimistic | +5 percentage points |

The same simulation seeds are shared across the three scenarios, allowing the effect of the appearance adjustment to be compared under controlled random conditions.

### Forecast Interpretation

The resulting probabilities should be interpreted as **model-based estimates**, not guarantees.

The forecast is frozen at the historical cutoff of **28 August 2026**. Events occurring after that date — such as injuries, transfers, fixture changes, suspensions, tactical decisions, or changes in player performance — are not incorporated unless the forecast is rebuilt with new information.


## Tech Stack

The project is implemented in Python using:

- **Pandas & NumPy** — data processing and numerical operations
- **Scikit-learn** — preprocessing and supporting ML utilities
- **XGBoost** — gradient-boosted goal regression
- **CatBoost** — categorical-aware goal regression
- **Joblib / Pickle** — model and forecasting artifact persistence
- **Altair** — interactive visualizations
- **Streamlit** — interactive forecasting dashboard
- **Jupyter Notebook** — experimentation, analysis, model evaluation, and simulation development

## Project Structure

```text
ronaldo-1000-ml/
│
├── app/
│   ├── streamlit_app.py
│   └── services/
│       ├── fixture_features.py
│       ├── goal_predictor.py
│       ├── recursive_features.py
│       ├── recursive_helpers.py
│       └── schedule_features.py
│
├── artifacts/
│   ├── goal_xgboost_production.json
│   ├── goal_catboost_production.cbm
│   ├── goal_xgboost_preprocessor.joblib
│   ├── feature_spec.json
│   ├── preprocessing_spec.json
│   ├── main_model_spec.json
│   └── ...
│
├── data/
│   └── ...
│
├── models/
│   └── ...
│
├── notebooks/
│   └── ...
│
├── outputs/
│   └── forecast/
│       ├── future_fixtures.csv
│       ├── appearance_sensitivity_summary.csv
│       ├── monthly_milestone_probabilities.csv
│       ├── fixture_milestone_probabilities.csv
│       └── ...
│
├── reports/
│   └── ...
│
├── src/
│   └── ...
│
├── tests/
│   └── ...
│
├── build_ronaldo_appearance_dataset.py
├── requirements.txt
└── README.md
```

### Application Architecture

The deployed application keeps prediction logic separate from the Streamlit interface.

`app/services/` contains the reusable forecasting components responsible for:

- Loading trained model artifacts
- Constructing fixture features
- Building schedule-based features
- Maintaining recursive forecasting state
- Generating XGBoost and CatBoost predictions
- Combining model outputs into the production goal forecast

`app/streamlit_app.py` acts primarily as the presentation layer, loading the saved forecast outputs and exposing the model results through the interactive dashboard.


## Running the Project Locally

### 1. Clone the Repository

```bash
git clone https://github.com/Odafe16/ronaldo-1000-ml.git
cd ronaldo-1000-ml
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Run the Dashboard

```bash
python -m streamlit run app/streamlit_app.py
```

Streamlit will start a local server and provide a URL for opening the dashboard in your browser.

## Deployment

The interactive dashboard is deployed with **Streamlit Community Cloud** and connected directly to this GitHub repository.

Changes pushed to the deployment branch can be picked up by Streamlit and deployed to the public application.

**Live dashboard:**  
https://cr7milestoneforecast.streamlit.app


## Limitations

This project is a forecasting system, and its outputs should not be interpreted as certain future outcomes.

Key limitations include:

- The forecast uses information available only through **28 August 2026**.
- Future injuries, suspensions, transfers, retirements, tactical changes, and unexpected availability changes are not known to the model.
- The future fixture schedule may change after the forecast cutoff.
- Goal predictions depend on historical patterns and the available pre-match features.
- The Poisson simulation assumes the predicted expected-goals rate provides an appropriate basis for generating match-level goal counts.
- Appearance probabilities introduce an additional source of uncertainty into long-term milestone forecasts.
- The forecast horizon ends on **28 May 2027**, so simulations that have not reached 1,000 goals by then are treated as unfinished rather than assumed never to reach the milestone.

The project is therefore best interpreted as:

> **Given the historical information available at the cutoff and the projected fixture schedule, what range of paths could take Cristiano Ronaldo from 978 to 1,000 career goals?**

## Author

**Ojo Jonathan Felix**

Computer Science student and developer interested in machine learning, data science, software engineering, and building data-driven products.

- GitHub: https://github.com/Odafe16
- LinkedIn: https://www.linkedin.com/in/jonathan-ojo-a34070393/
- Live Project: https://cr7milestoneforecast.streamlit.app

---

If you found this project interesting, feel free to explore the repository, experiment with the dashboard, or connect with me.