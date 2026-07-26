# MSCS 634 - Lab Assignment 4

**Name:** Chun-Shuo Sheu
**Course:** MSCS 634 - Big Data & Data Mining
**Lab:** Lab 4 - Regression Analysis with Regularization Techniques

## Purpose

The purpose of this lab is to explore a progression of regression techniques on the Diabetes Dataset from `sklearn` and to see how **regularization** controls the overfitting that flexible models introduce. The lab builds five families of models:

- **Simple Linear Regression** - one predictor (BMI).
- **Multiple Regression** - all 10 predictors.
- **Polynomial Regression** - degrees 1 through 5, to expose the bias-variance trade-off.
- **Ridge Regression (L2)** - shrinks coefficients smoothly toward zero.
- **Lasso Regression (L1)** - drives coefficients exactly to zero, performing feature selection.

Every model is evaluated on the same held-out test set using **MAE**, **MSE**, **RMSE**, and **R²** so the results are directly comparable, and each is also scored on the training set so that the train-test gap makes overfitting visible.

## How to Reproduce

```bash
python3 -m venv venv
source venv/bin/activate
pip install numpy pandas matplotlib scikit-learn jupyter
jupyter notebook MSCS_634_Lab_4.ipynb
```

## Results

An 80/20 train-test split (`random_state=42`) gives 353 training and 89 test samples. Test-set performance:

| Rank | Model | MAE | MSE | RMSE | R² |
|---|---|---|---|---|---|
| 1 | Lasso, degree 2 (α = 2.98) | **41.16** | **2560.42** | **50.60** | **0.517** |
| 2 | Ridge, degree 2 (α = 92.5) | 41.41 | 2649.35 | 51.47 | 0.500 |
| 3 | Multiple Linear (10 features) | 42.79 | 2900.19 | 53.85 | 0.453 |
| 4 | Polynomial degree 2, unregularized | 43.58 | 3096.03 | 55.64 | 0.416 |
| 5 | Simple Linear (bmi only) | 52.26 | 4061.83 | 63.73 | 0.233 |
| 6 | Polynomial degree 3, unregularized | 161.19 | 77808.80 | 278.94 | -13.686 |

Train vs. test R², which is where the overfitting shows up:

| Model | Train R² | Test R² | Gap |
|---|---|---|---|
| Simple Linear (bmi) | 0.366 | 0.233 | 0.132 |
| Multiple Linear | 0.528 | 0.453 | 0.075 |
| Polynomial degree 2 | 0.606 | 0.416 | 0.191 |
| Polynomial degree 3 | 0.877 | -13.686 | 14.563 |
| Ridge degree 2 | 0.563 | 0.500 | 0.063 |
| Lasso degree 2 | 0.531 | 0.517 | **0.015** |

## Key Insights

- **Lasso on degree-2 polynomial features is the best model**, winning on all four metrics (R² = 0.517) while using only **16 of 65** terms. It also has the smallest train-test gap of any model in the lab.
- **Regularization did more than repair the damage - it beat every unregularized model.** Plain multiple regression reached R² = 0.453 and the unregularized degree-2 model actually did worse at 0.416, but Ridge and Lasso on that same degree-2 feature space reached 0.500 and 0.517. The interaction and squared terms genuinely carry a little signal; only a constrained model could extract it without also absorbing the noise. Neither ingredient worked alone - it took expanding the feature space *and then* penalizing it.
- **Unregularized polynomial expansion was actively harmful.** Test error was already at its minimum at degree 1 and degraded from there. By degree 3 (285 features, 353 samples) test R² collapsed to -13.7, worse than predicting the mean. At degrees 4-5 the model interpolates the training data exactly - training RMSE of 0.0 and R² of 1.0 - and the test predictions become meaningless.
- **Regularized models fit the training data *worse* on purpose.** Ridge and Lasso have lower training R² (0.563, 0.531) than the unregularized degree-2 model (0.606) yet much better test R². That trade is the bias-variance trade-off made concrete.
- **The serum features are heavily collinear.** In the unregularized multiple regression, `s1` receives a coefficient of about **-931** despite having a *positive* raw correlation with the target (r = 0.21) - the model is playing it against `s2` and `s5` to cancel shared variance. Ridge stabilizes these coefficients, and Lasso removes `s1`, `s2` and `s4` from the model outright.
- **BMI and `s5` are the dominant predictors.** They lead the raw correlation ranking (0.59 and 0.57) and hold the two largest surviving Lasso coefficients, followed by `bp` and `s3`.
- **Ridge and Lasso differ structurally, not just numerically.** At any finite α, Ridge keeps all 65 coefficients alive and merely shrinks them; Lasso eliminates them one at a time (by α ≈ 10 only 4 survive, and by α ≈ 50 none do). The coefficient-path plots make this contrast obvious.
- **There is a hard ceiling on accuracy.** The best model still leaves roughly 48% of the variance unexplained, with a typical error of ~50 units against a target standard deviation of 77. Disease progression depends on factors this dataset does not measure - which is exactly why the high-capacity models could only overfit.

## Challenges and Decisions

- **Choosing where to apply regularization.** Ridge and Lasso are pointless on a model that is not overfitting, and hopeless on the degree-3 model with its 285 features. I applied them to the **degree-2** feature space, which is where the unregularized model first began to overfit and where regularization had something real to fix.
- **Selecting α honestly.** Reading α off the test-error curve would leak test information into model selection and inflate the reported scores. I used **5-fold cross-validation on the training set only** (`GridSearchCV` over 60 log-spaced α values), then scored the test set exactly once per model. The difference is not cosmetic: the test-optimal values were α = 26.5 (Ridge) and 2.18 (Lasso), while CV chose 92.5 and 2.98. The notebook still plots the full test curve, but for illustration only.
- **Scaling before penalizing.** L1 and L2 penalties are scale-sensitive, so a term like `bmi²` would be penalized differently from `bmi` purely because of magnitude. All polynomial features are passed through `StandardScaler` before Ridge/Lasso, and the same scaling sits inside the polynomial pipelines via `make_pipeline`.
- **Interpreting a non-monotonic degree curve.** Degree 5 posts a lower test RMSE (261) than degree 4 (423), which looks like a partial recovery but is not one. Once the model interpolates the training set exactly, the fit is determined by the minimum-norm solution rather than by any signal in the data, so test error in that regime is essentially arbitrary. I noted this in the notebook rather than letting the number imply a real trend.
- **Readable comparison plots.** The degree-3 model's test R² of -13.7 flattened every other bar in the comparison chart, and its RMSE of 278.9 did the same. I clipped the R² axis at -1.0 with an annotated arrow marking the off-scale bar, and used a log scale for the RMSE and MAE charts, so the differences among the competitive models stay visible.
- **Lasso convergence.** Coordinate descent on 65 correlated polynomial features does not always converge within the default iteration budget, so `max_iter` was raised to 100,000 and `ConvergenceWarning` suppressed to keep the notebook output readable.
- **Data cleaning was a no-op.** The dataset ships with no missing values, no duplicate rows, and predictors already mean-centered and scaled to unit L2 norm. I verified all of this explicitly rather than assuming it. Worth noting: the *target* is **not** scaled, so all error metrics are in the original disease-progression units.
- **Reproducibility.** A fixed `random_state=42` was used for the train-test split and for every estimator that accepts one, so the results are reproducible run-to-run.
