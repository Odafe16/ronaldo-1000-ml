# Version 2 — Model Complexity Experiments

## 1. Objective

I investigated whether controlled changes to XGBoost and CatBoost
model complexity could improve the goal-scoring predictions in my
Ronaldo 1,000-goal forecasting project.

I preserved the original Version 1 models, data contracts,
chronological validation folds, and final-test boundary.

All experiments were conducted separately under `experiments/v2/`.

## 2. Experimental Methodology

I evaluated candidate configurations using four expanding-window
chronological validation folds containing 340 validation appearances.

My primary selection metric was macro-averaged Poisson deviance.

I also examined pooled Poisson deviance, MAE, RMSE, prediction bias,
prediction variability, and fold-level stability.

I changed one hyperparameter at a time before evaluating selected
candidates within the original 60% XGBoost / 40% CatBoost ensemble.

The final-test period was not used for hyperparameter selection.

## 3. Original Baseline

| Component | Configuration |
|---|---|
| XGBoost | 15 iterations, depth 2 |
| CatBoost | 100 iterations, depth 2 |
| Ensemble | 60% XGBoost, 40% CatBoost |
| Macro Poisson deviance | 1.106496 |

## 4. XGBoost Experiments

### 4.1 Tree Depth

| max_depth | macro_poisson_deviance | pooled_poisson_deviance | macro_mae | macro_rmse | macro_bias |
| --- | --- | --- | --- | --- | --- |
| 2.000000 | 1.109154 | 1.098756 | 0.712476 | 0.869278 | -0.015777 |
| 3.000000 | 1.110045 | 1.099673 | 0.711300 | 0.869660 | -0.008993 |
| 4.000000 | 1.113508 | 1.102080 | 0.713058 | 0.871132 | -0.011300 |

Increasing tree depth did not improve overall macro Poisson deviance.
I retained depth 2.

### 4.2 Regularization

| candidate | min_child_weight | reg_lambda | macro_deviance | pooled_deviance | macro_mae | macro_bias |
| --- | --- | --- | --- | --- | --- | --- |
| lambda_10 | 10 | 10 | 1.109054 | 1.098606 | 0.712389 | -0.015577 |
| original | 10 | 20 | 1.109154 | 1.098756 | 0.712476 | -0.015777 |
| lambda_5 | 10 | 5 | 1.109157 | 1.098647 | 0.712534 | -0.015968 |
| child_weight_5 | 5 | 20 | 1.109196 | 1.098782 | 0.712483 | -0.015776 |
| child_weight_2 | 2 | 20 | 1.109196 | 1.098782 | 0.712483 | -0.015776 |

Reducing L2 regularization from 20 to 10 produced a very small
improvement, but the effect was insufficient to justify changing
the baseline configuration.

### 4.3 Boosting Iterations

| n_estimators | macro_deviance | pooled_deviance | macro_mae | macro_rmse | macro_bias | prediction_std |
| --- | --- | --- | --- | --- | --- | --- |
| 30.000000 | 1.107376 | 1.096395 | 0.708606 | 0.868314 | 0.002390 | 0.056586 |
| 15.000000 | 1.109154 | 1.098756 | 0.712476 | 0.869278 | -0.015777 | 0.029689 |
| 50.000000 | 1.109832 | 1.098545 | 0.705113 | 0.869362 | 0.025949 | 0.082072 |
| 100.000000 | 1.120768 | 1.110028 | 0.705923 | 0.874222 | 0.056696 | 0.133109 |
| 200.000000 | 1.137584 | 1.132856 | 0.718085 | 0.882754 | 0.085374 | 0.203060 |

Increasing XGBoost from 15 to 30 iterations improved individual-model
performance. However, this improvement did not carry over to the
original ensemble when CatBoost remained unchanged.

## 5. CatBoost Experiments

### 5.1 Tree Depth

| depth | macro_deviance | pooled_deviance | macro_mae | macro_rmse | macro_bias | prediction_std |
| --- | --- | --- | --- | --- | --- | --- |
| 2.000000 | 1.112413 | 1.102292 | 0.698919 | 0.868244 | 0.012699 | 0.123679 |
| 3.000000 | 1.124673 | 1.110185 | 0.703087 | 0.871734 | -0.009295 | 0.146843 |
| 4.000000 | 1.126931 | 1.113147 | 0.702643 | 0.873570 | 0.034603 | 0.144087 |

Deeper CatBoost trees substantially worsened macro Poisson deviance.
I retained depth 2.

### 5.2 Boosting Iterations

| iterations | macro_deviance | pooled_deviance | macro_mae | macro_rmse | macro_bias | prediction_std |
| --- | --- | --- | --- | --- | --- | --- |
| 150.000000 | 1.107921 | 1.098969 | 0.696770 | 0.865840 | 0.005427 | 0.141846 |
| 50.000000 | 1.112247 | 1.105657 | 0.699296 | 0.870297 | 0.055050 | 0.099894 |
| 100.000000 | 1.112413 | 1.102292 | 0.698919 | 0.868244 | 0.012699 | 0.123679 |
| 200.000000 | 1.129687 | 1.113024 | 0.703800 | 0.872176 | -0.015866 | 0.134055 |

Increasing CatBoost from 100 to 150 iterations produced the strongest
individual CatBoost result among the configurations tested.

Performance deteriorated at 200 iterations.

## 6. Ensemble Comparison

| configuration | macro_deviance | pooled_deviance | macro_mae | macro_bias |
| --- | --- | --- | --- | --- |
| Original XGB + Cat150 | 1.103669 | 1.093719 | 0.705683 | -0.007296 |
| XGB30 + Cat150 | 1.104279 | 1.094198 | 0.703361 | 0.003605 |
| Original ensemble | 1.106496 | 1.096307 | 0.706974 | -0.004387 |

The strongest ensemble combined the original 15-tree XGBoost model
with the 150-iteration CatBoost candidate.

The original 60/40 ensemble weights were preserved.

## 7. Leading Version 2 Candidate

| Component | Configuration |
|---|---|
| XGBoost | Original 15 iterations |
| CatBoost | 150 iterations |
| Ensemble weights | 60% XGBoost / 40% CatBoost |
| Macro Poisson deviance | 1.103669 |
| Relative macro improvement | 0.2555% |

The candidate improved macro Poisson deviance in three of four
chronological validation folds.

## 8. Prediction Behavior

| metric | original_ensemble | v2_candidate |
| --- | --- | --- |
| Mean predicted goals | 0.766466 | 0.765868 |
| Prediction standard deviation | 0.065636 | 0.072295 |
| Minimum prediction | 0.599467 | 0.592044 |
| Maximum prediction | 0.889190 | 0.916793 |
| Mean absolute error | 0.701696 | 0.699930 |

The updated ensemble produced slightly more variable predictions,
while its average predicted goals remained close to the original.

The improvement was not driven by a large shift in overall
prediction magnitude.

## 9. Robustness Assessment

I compared the original and updated ensembles on the same
340 validation appearances.

- Appearances improved: 192
- Appearances worsened: 148
- Observed macro deviance change: -0.002827
- Bootstrap iterations: 5000
- Exploratory bootstrap 95% interval:
  (-0.006242, 0.000641)
- Bootstrap fraction favoring Version 2:
  0.9508

The bootstrap resampled appearances within each validation fold.

The interval includes zero, so the analysis does not establish
a reliable generalization improvement.

The bootstrap also does not fully account for temporal dependence.

## 10. Limitations

1. I tested multiple configurations on the same validation folds,
   introducing model-selection uncertainty.
2. The observed improvement is modest.
3. The bootstrap is exploratory and does not fully capture
   temporal dependence between appearances.
4. I have not performed an independent final-test evaluation
   of the Version 2 candidate.
5. Improvements in goal-count prediction do not automatically
   establish improvements in milestone-date forecasting.

## 11. Conclusion

I identified a promising experimental configuration by increasing
CatBoost iterations from 100 to 150 while preserving the original
XGBoost model and ensemble weights.

The candidate reduced macro Poisson deviance from
1.106496 to 1.103669, an improvement of
approximately 0.26%.

However, the uncertainty analysis and repeated use of the same
validation folds prevent me from treating this as a confirmed
generalization improvement.

I therefore retain this configuration as an experimental
Version 2 candidate.

Version 1 remains frozen and unchanged.

## 12. Reproducibility

Notebook:
`experiments/v2/notebooks/01_model_complexity.ipynb`

Experiment outputs:
`experiments/v2/reports/`

Generated:
2026-11-04 09:09 UTC
