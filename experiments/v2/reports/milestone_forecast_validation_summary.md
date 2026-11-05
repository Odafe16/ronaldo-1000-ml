# V2 Milestone Forecast Validation

## Objective

I evaluated the internal consistency, probability distributions,
and appearance sensitivity of my existing appearance-aware
Monte Carlo milestone forecast.

I used saved Version 1 forecast outputs without modifying
the original forecasting system.

## Forecast Probability Integrity

- Forecast fixtures: 45
- Monte Carlo simulations: 1,000
- Completed paths: 949
- Unfinished paths: 51
- Completion probability: 94.9%

I verified that fixture milestone probabilities were nonnegative,
that cumulative probabilities were monotonic, and that the
cumulative distribution matched the sum of individual
fixture probabilities.

All fixture probability checks passed.

## Appearance Sensitivity

| Scenario | Completed paths | Completion probability | Conditional median date |
|---|---:|---:|---|
| Baseline | 942/1000 | 94.2% | 2027-02-25 |
| Conservative | 878/1000 | 87.8% | 2027-04-02 |
| Optimistic | 964/1000 | 96.4% | 2027-02-18 |

I independently reconstructed the scenario summaries from
individual saved simulation paths.

All 21 scenario-summary comparisons passed.

The conservative scenario produced fewer milestone completions
and a later conditional median date than the baseline and
optimistic scenarios.

This demonstrates sensitivity to appearance assumptions,
not historical predictive accuracy.

## Individual Simulation Integrity

I verified that:

- Completed paths contain milestone dates.
- Unfinished paths have no milestone dates.
- Completed paths have zero remaining goals.
- Unfinished paths have positive remaining goals.
- Remaining goal counts are nonnegative.
- Appearance counts are nonnegative.
- Scenario labels and completion outcomes are present.

All eight integrity checks passed.

## Baseline Monte Carlo Comparison

The fixture-level forecast estimated a completion probability
of 94.9%, compared with 94.2% in
the separate baseline sensitivity experiment.

The observed difference was
0.70 percentage points.

Under an independent-run approximation, the estimated
standard error of this difference was
1.02 percentage points.

The standardized difference was
0.690.

The results are statistically compatible with a common
underlying probability under this approximation.

However, I did not independently establish that both
experiments used identical simulation settings.

## Interpretation of Milestone Dates

The reported milestone-date quantiles are conditional
on reaching 1,000 goals within the simulated fixture horizon.

Unfinished paths remain in the denominator when calculating
completion probabilities.

I do not interpret conditional milestone-date quantiles
as unconditional prediction intervals.

## Limitations

1. I validated saved simulation outputs rather than rerunning
   the complete recursive simulation engine.
2. I did not perform a leakage-safe historical milestone-date
   backtest.
3. The saved simulation paths do not contain every intermediate
   match-level state.
4. Monte Carlo standard errors measure simulation sampling
   uncertainty, not model uncertainty.
5. Forecast outcomes depend on the fixture schedule, forecast
   cutoff, goal model, appearance model, and scenario assumptions.
6. The two baseline experiments may differ in simulation settings.

## Conclusion

The saved milestone forecast passed the probability,
scenario-summary, and individual-path consistency checks.

The results also demonstrate meaningful sensitivity to
appearance assumptions.

These findings support the internal consistency of the
saved forecasting outputs, but they do not establish
historical milestone-date predictive accuracy.

I retain the existing Version 1 forecasting system without
promoting a new milestone forecasting model.
