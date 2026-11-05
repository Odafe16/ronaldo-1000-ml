# Version 2 — Model Validation Report

## 1. Objective

I evaluated whether my experimental Version 2 goal-scoring
ensemble demonstrates sufficiently reliable improvements
over Version 1 to justify model promotion.

I preserved the frozen Version 1 implementation and
restricted this analysis to the Version 2 experiment workspace.

## 2. Validation Protocol

- Strategy: expanding-window chronological validation
- Validation folds: 4
- Out-of-fold appearances: 340
- Historical final-test observations used: No
- Baseline training boundaries: Verified
- Primary metric: mean Poisson deviance

I evaluated Version 1, Version 2, and a historical-average
baseline using identical validation appearances.

## 3. Model Performance

| Model | Macro Poisson deviance |
|---|---:|
| Historical-average baseline | 1.115870 |
| Version 1 | 1.106496 |
| Version 2 | 1.103669 |

Version 2 reduced macro Poisson deviance by
0.26%
relative to Version 1 and
1.09%
relative to the historical-average baseline.

Both improvements are modest.

## 4. Temporal Generalization

Version 2 outperformed Version 1 in three of four
chronological folds.

It also outperformed the historical-average baseline
in three of four folds.

Fold 3 remained problematic: both ensembles performed
worse than the baseline and underestimated the observed
scoring rate.

I interpret this as evidence that predictive performance
is not consistent across historical periods.

## 5. Calibration and Scoring Distribution

The five-bin descriptive calibration analysis produced:

- Version 1 calibration error: 0.099481
- Version 2 calibration error: 0.095535

These results suggest a small descriptive improvement
in calibration, but they do not establish statistical
significance.

The observed scoring distribution varied substantially
between folds, particularly in the frequency of
scoreless appearances and multi-goal matches.

## 6. Recent Scoring Form

Recent goals per appearance over the previous five matches
showed little overall association with next-match goals:

- Pearson correlation: -0.011949
- Spearman correlation: 0.005414

I therefore found insufficient evidence to justify
increasing model sensitivity to recent scoring form
solely because predictions appeared conservative.

## 7. Paired Robustness

Version 2 improved:

- 192 of 340 appearances relative to Version 1
- 181 of 340 appearances relative to the historical baseline

The observed improvements were not uniform across folds.

## 8. Bootstrap Uncertainty

I used 5,000 fold-stratified paired bootstrap repetitions
to examine Version 2 against the historical baseline.

- Observed macro deviance change: -0.012200
- Exploratory 95% interval: (-0.036410, 0.011130)
- Fraction of resamples favoring Version 2:
  0.8438

The interval includes zero, so the direction of
improvement remains uncertain.

The bootstrap resamples individual appearances within
folds and does not fully preserve temporal dependence.

Version 2 was also selected using these validation
folds, so this analysis is not independent confirmation
of its superiority.

## 9. Limitations

1. The validation dataset contains only 340 appearances.
2. Individual chronological folds are relatively small.
3. Goal outcomes exhibit substantial match-level variability.
4. Scoring distributions differ across historical periods.
5. Model selection and evaluation share validation data.
6. The bootstrap does not fully account for temporal dependence.
7. No independent forward evaluation has yet been completed.

## 10. Promotion Decision

**Decision: Do not promote Version 2.**

Although Version 2 achieves slightly better observed
Poisson deviance, the improvement is modest and uncertain.

I will retain Version 1 as the frozen production model
and preserve Version 2 as an experimental candidate.

Before considering promotion, I would require stronger
evidence from an independent forward evaluation using
a predefined comparison protocol.

I will not use the reserved historical final-test
observations for further hyperparameter tuning.

## 11. Conclusion

This experiment showed that increasing model complexity
does not automatically produce meaningful improvements
in goal-scoring forecasts.

My main finding is that Version 2 offers a small
observed advantage, but not enough evidence to justify
replacing Version 1.

The next priority is independent evaluation rather
than additional tuning against the existing folds.
