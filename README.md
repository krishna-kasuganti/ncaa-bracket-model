# NCAA Tournament Win Probability Model

## Project Overview

An NCAA men's basketball tournament win probability model built from scratch. Given any individual tournament matchup, the model outputs the probability that a specified team wins. It is built on feature engineering from raw box-score and ranking data, a temporally clean train/validation/test split, and evaluation using both classification accuracy and probability calibration metrics.

## Motivation

Built as a personal project to develop practical ML engineering skills. The emphasis throughout is on rigorous methodology: no data leakage, walk-forward temporal validation, and a fully held-out test set touched exactly once.

## Data

Data comes from the [Kaggle March Machine Learning Mania](https://www.kaggle.com/competitions/march-machine-learning-mania-2025) dataset. Raw files live in `data/raw/`. The five files that feed the model are:

| File | Role |
|---|---|
| `MRegularSeasonDetailedResults.csv` | Full box-score stats for every regular-season game since 2003 |
| `MNCAATourneyCompactResults.csv` | Tournament game outcomes — the target variable |
| `MNCAATourneySeeds.csv` | Each team's tournament seed by year |
| `MMasseyOrdinals.csv` | Weekly ratings from dozens of external systems including KenPom and Sagarin |
| `MTeams.csv` | Team ID to name lookup |

## Project Structure

The project is organized as three sequential Jupyter notebooks, each building on the last.

`notebooks/01_eda.ipynb` loads each raw file, explores schema and distributions, and identifies which third-party rating systems have sufficient historical coverage to be useful features.

`notebooks/02_feature_engineering.ipynb` is the core of the project. It constructs per-team season averages from box scores, engineers efficiency ratings and strength of schedule, builds tournament history time-series features, joins in seeds and external rankings, and assembles the final matchup table. The notebook ends by saving train, validation, and test splits to `data/processed/` as Parquet files.

`notebooks/03_modeling.ipynb` loads those splits, fits a logistic regression baseline, tunes an XGBoost classifier across several hyperparameter configurations, and evaluates the final model on the held-out 2024 test set.

## Feature Engineering

Each row in the training data represents a tournament matchup. Both teams' full feature vectors are stacked side by side into a single flat row. The target is binary: 1 if TeamA wins, 0 otherwise. TeamA/TeamB assignment is randomized at construction time to prevent the model from learning a spurious ordering signal.

Features fall into five groups.

**Box-score averages.** Per-team regular-season means of scoring, field goal attempts and makes, three-point shooting, free throws, offensive rebounds, assists, turnovers, steals, blocks, and fouls — computed from the completed regular season before the tournament begins.

**Efficiency ratings.** Offensive efficiency (points scored per 100 estimated possessions), defensive efficiency (points allowed per 100 estimated possessions), and the net margin between them. Possessions are estimated using the standard formula: FGA − OR + 0.44·FTA + TO.

**Strength of schedule.** The average end-of-season win rate of each team's regular-season opponents, serving as a proxy for schedule difficulty.

**Tournament history.** Four time-series features derived from prior NCAA tournament results: round reached the previous year, rolling 3-year average round reached, number of tournament appearances in the prior 5 years, and a flag for whether the team won the national championship the prior year. All are constructed with `.shift(1)` to ensure no same-season information is included.

**External rankings.** Pre-tournament ordinal rankings from four systems selected for longest historical coverage in the dataset: KenPom (POM), Sagarin (SAG), Massey (MOR), and Dunkel (DUN). Rankings are taken from the last published snapshot before the tournament begins.

## Methodology

The train/validation/test split is strictly temporal. Training data covers the 2003–2022 tournament seasons (1,248 matchups). The 2023 tournament (67 matchups) served as a validation set used only for hyperparameter selection. The 2024 tournament (67 matchups) was held out entirely and touched exactly once for final evaluation.

All features are constructed exclusively from information that would have been available before the tournament began in a given year. Regular-season statistics reflect the completed regular season. Massey rankings use the last pre-tournament snapshot. Tournament history features use prior-year data only. There is no forward-looking bias in the feature set.

## Model

A logistic regression with standard scaling serves as the baseline. The final model is XGBoost with the following hyperparameters, tuned sequentially on the 2023 validation set:

```
n_estimators      = 1000
learning_rate     = 0.05
max_depth         = 6
subsample         = 0.8
early_stopping_rounds = 20
```

Three metrics are reported: accuracy (for comparison against publicly available bracket benchmarks), log loss (the standard probabilistic scoring rule), and Brier score (mean squared error of predicted probabilities), which penalizes both miscalibration and wrong-direction predictions.

## Results

| Set | Accuracy | Log Loss | Brier Score |
|---|---|---|---|
| Logistic Regression — validation (2023) | 59.7% | 0.656 | 0.221 |
| XGBoost — validation (2023) | 73.1% | 0.592 | 0.202 |
| XGBoost — test (2024, touched once) | 67.2% | 0.631 | 0.220 |

The gap between validation and test accuracy reflects mild overfitting to the 2023 tournament through hyperparameter tuning — an inherent limitation of having a single held-out validation season with 67 games. For context, the ESPN Tournament Challenge average historically places around 66–68% of game picks correctly, putting the model's test performance at the upper end of that range.

## Known Limitations

Three limitations are documented honestly.

Defensive efficiency uses the team's own possession estimate in the denominator rather than the opponent's, because opponent-level possession counts are not available in the same row. Since per-game possession counts are nearly equal between opponents, the approximation is small but introduces a systematic bias for teams with unusually high or low turnover rates.

The `appearances_5yr` feature uses `min_periods=1` in the rolling window. For teams in their first few seasons of D1 history, the count is computed over fewer than five prior years with no indicator that the window is shorter, making early-career programs appear more directly comparable to established programs than they are.

The DUN and SAG Massey systems are missing data for several seasons entirely. For those years, median imputation assigns the same constant to all teams, so the ranking columns carry zero discriminating power. The model sees the feature but cannot use it.

## How to Run

```bash
pip install -r requirements.txt
```

Run the notebooks in order: `01_eda.ipynb`, `02_feature_engineering.ipynb`, `03_modeling.ipynb`. Notebook 02 saves the processed splits to `data/processed/`; notebook 03 reads them directly and will use stale data if notebook 02 is not run first after any changes to the raw data or feature pipeline.

## Environment

Python 3.11, VS Code, virtual environment. Key dependencies: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `fastparquet`.
