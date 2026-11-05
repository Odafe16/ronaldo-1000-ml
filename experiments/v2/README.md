# Version 2 — Experimental Model Validation

This directory contains additional experiments evaluating the goal-scoring models and milestone forecasting system developed for my [Ronaldo's 1,000th Goal | Machine Learning Forecast] project.

The objective was to investigate whether changes to model complexity could improve predictive performance, examine model reliability across chronological validation periods, and assess the internal consistency of the existing Monte Carlo milestone forecast.

Version 1 remains the production system.

## Experiments

| Notebook                                 | Focus                                                                                           |
| ---------------------------------------- | ----------------------------------------------------------------------------------------------- |
| `01_model_complexity.ipynb`              | XGBoost and CatBoost complexity experiments, ensemble comparisons, and chronological evaluation |
| `02_model_validation.ipynb`              | Calibration, error analysis, historical baselines, and statistical uncertainty                  |
| `03_milestone_forecast_validation.ipynb` | Monte Carlo probability integrity, appearance sensitivity, and milestone-date uncertainty       |

## Key Findings

### Model Complexity

The best experimental ensemble achieved a macro-averaged Poisson deviance of **1.103669**, compared with **1.106496** for the Version 1 ensemble.

Although the experimental configuration improved performance in three of four validation folds, the improvement was small, and the bootstrap confidence interval included zero.

The evidence was insufficient to justify replacing the existing production model.

### Model Validation

Both ensembles outperformed the historical training-mean baseline on the observed chronological validation folds.

However, uncertainty estimates did not establish a statistically conclusive advantage over the baseline.

The analysis also identified variation in prediction bias across validation periods.

### Milestone Forecast Validation

The saved Monte Carlo forecast passed probability consistency and simulation-integrity checks.

The fixture-level forecast produced a **94.9% probability** of reaching 1,000 goals within its 45-fixture horizon.

Appearance-sensitivity experiments produced completion probabilities of:

* Conservative: **87.8%**
* Baseline: **94.2%**
* Optimistic: **96.4%**

These results demonstrate sensitivity to appearance assumptions.

Milestone-date quantiles are conditional on reaching the milestone within the simulated horizon. Historical milestone-date predictive accuracy was not evaluated.

## Final Decision

**Experimental validation completed — no model promotion.**

The experiments strengthened the evaluation and documentation of the existing forecasting system but did not provide sufficient evidence to replace Version 1.

## Directory Structure

```text
experiments/v2/
├── notebooks/
│   ├── 01_model_complexity.ipynb
│   ├── 02_model_validation.ipynb
│   └── 03_milestone_forecast_validation.ipynb
├── reports/
│   ├── experiment_summary.json
│   ├── model_complexity_report.md
│   ├── model_validation_report.md
│   ├── model_validation_summary.json
│   ├── milestone_forecast_validation_summary.json
│   ├── milestone_forecast_validation_summary.md
│   └── experimental CSV outputs
└── README.md
```

Detailed experimental results and validation summaries are available in the `reports/` directory.
