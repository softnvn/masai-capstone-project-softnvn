# /analytics — Titanic profiling, data story and predictive modelling

Part of the **Zepto Data & AI Platform** capstone. This module shows Zepto's analyst-to-data-scientist workflow on the
classic Titanic dataset: profile, clean, visualise, then build, tune and ship a predictive pipeline.
All numbers below come from executed notebooks (outputs are saved in the  files).

## Files

| File | What it is |
|---|---|
| `01_eda.ipynb` | **The only raw load** (`sns.load_dataset('titanic')`), saves `titanic.csv`, profiling, cleaning, univariate, bivariate, data story, z-score check |
| `02_modeling.ipynb` | Reads the same `titanic.csv`, stratified split, `Pipeline`/`ColumnTransformer`, 3 classifiers, imbalance comparison, `GridSearchCV` + OOB, regression, comparison table, `joblib` save/reload |
| `titanic_utils.py` | Shared row-level cleaning rules and colours, imported by both notebooks so the data is cleaned the same way once |
| `titanic.csv` | Committed offline fallback, written by `df.to_csv("titanic.csv", index=False)` right after loading (891 × 15) |
| `titanic_best_pipeline.joblib` | Complete fitted pipeline (preprocessing + tuned Random Forest), usable on raw rows |
| `model_comparison.csv` | The final comparison table as data |
| `figures/*.png` | Charts saved by the notebooks (supporting artifacts only; every chart is interpreted in text below) |

## How to run

```bash
cd analytics
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace 01_eda.ipynb        # needs internet the first time
jupyter nbconvert --to notebook --execute --inplace 02_modeling.ipynb   # works fully offline
# or open them interactively:  jupyter notebook
```
If `sns.load_dataset` cannot reach the internet, `01_eda.ipynb` automatically falls back to `pd.read_csv("titanic.csv")`.
Random seeds are fixed (`random_state=42`), so re-running reproduces every number below.

**Single-load guarantee:** `sns.load_dataset(...)` is called exactly once in the module (first code cell of `01_eda.ipynb`).
`02_modeling.ipynb` starts from `pd.read_csv("titanic.csv")` and applies the same `drop_sparse_rows` rule.

---

# Part A — Profiling, cleaning and the data story

## Task 1 — Profile
`df.shape` = **(891, 15)**. `df.info()` shows 4 columns with nulls (`age` 714 non-null, `embarked` 889, `deck` 203, `embark_town` 889).
`df.describe()`: mean age 29.70 (std 14.53, range 0.42–80), mean fare 32.20 (std 49.69, median 14.45, max 512.33), 38.38% survived.
The full `info()`/`describe()` output is in the notebook.


#### Missing values measured, and the strategy each one gets

| Column | Missing | % missing | Rule that applies | Decision |
|---|---|---|---|---|
| `embarked` | 2 | **0.22%** | < 5% → drop rows | Drop the 2 rows. |
| `embark_town` | 2 | **0.22%** | < 5% → drop rows | Same 2 passengers as `embarked` (it is the spelled-out port), so the same 2-row drop fixes both. |
| `age` | 177 | **19.87%** | 5%–30% → impute | Impute with the **median age of the passenger's (sex, pclass) group**. |
| `deck` | 688 | **77.22%** | > 30% → too sparse to impute | Keep the column, encode missing as its **own category `"Missing"`** (justified below). |

**Class balance:** 549 died (61.62%) vs 342 survived (38.38%), about 1.6 : 1. That is a moderate imbalance. It matters later for the stratified split and for the imbalance-handling comparison in `02_modeling.ipynb`.

**Why a group median for `age` and not a global one?** Age differs a lot by class and sex. In the table below the medians range from 21.5 (3rd-class women) to 40 (1st-class men). A single global median (28) would push every missing 1st-class man about 12 years too young. Median beats mean because age is slightly right-skewed.

## Task 2 — Missing-value handling

**Why `deck` becomes a `"Missing"` category instead of being dropped or imputed.** At 77.22% missing, any imputed deck would mostly be invented. But the missingness is *not random*, and that is itself information:
- Deck is known for 80.8% of 1st-class passengers but only 8.7% of 2nd-class and 2.4% of 3rd-class passengers. Cabin records mostly survived for the wealthy.
- Passengers with a recorded deck survived at **66.7%**, against **29.9%** for those without one.

Dropping the column would throw that signal away. Imputing would destroy it by making "unknown" look like a real deck. An explicit `"Missing"` level keeps the signal honestly. (In `02_modeling.ipynb` I leave `deck` out of the features anyway. The deck-known/unknown split is almost a copy of `pclass`, so it adds little beyond class. That trade-off is stated there.)

After cleaning: **889 rows × 15 columns, zero missing values.**

## Task 3 — Univariate analysis (`figures/01_univariate_age_fare.png`)

#### Univariate findings

**IQR outlier counts** (outliers lie outside [Q1 − 1.5·IQR, Q3 + 1.5·IQR]):

| Column | Q1 | Q3 | IQR | Bounds | Outliers |
|---|---|---|---|---|---|
| `age` (observed values only, before imputation) | 20.12 | 38.00 | 17.88 | [−6.69, 64.81] | **11** |
| `age` (cleaned, after group-median imputation) | 21.50 | 36.00 | 14.50 | [−0.25, 57.75] | **32** |
| `fare` (cleaned) | 7.90 | 31.00 | 23.10 | [−26.76, 65.66] | **114** |

`age` is reported twice on purpose. Imputing 177 values at just a few medians piles mass in the middle of the distribution. That narrows the IQR (17.88 → 14.50), so the same elderly passengers are flagged more often (11 → 32). The raw figure of **11** is the honest description of the age distribution. None of these outliers are errors: an 80-year-old passenger is real, so I keep them all. Of the 889 fares, **114** fall above the upper fence of 65.66. Those are real luxury tickets (up to 512.33), not errors, and they are kept. The model's `StandardScaler` handles the scale.

**Skewness of `fare`:** mean = **32.10**, median = **14.45**, mode = **8.05**. The ordering is **mode < median < mean**, which is the signature of a **right-skewed (positively skewed)** distribution. Most passengers paid a low 3rd-class fare, so the mode and median sit low, while a long tail of expensive 1st-class tickets drags the mean up to more than twice the median. The sample skewness of **+4.80** confirms it. `age` is close to symmetric by comparison: its histogram has a mild right tail and a small bump of infants.

## Task 4 — Bivariate analysis (`figures/02_correlation_heatmap.png`)

#### Bivariate findings

**Survival rates, computed with boolean masks:**
- **(a) By sex:** female **74.04%** vs male **18.89%**.
- **(b) By class:** 1st **62.62%**, 2nd **47.28%**, 3rd **24.24%**.
- **(c) By sex & class:** female 1st / 2nd / 3rd = **96.74% / 92.11% / 50.00%**; male 1st / 2nd / 3rd = **36.89% / 15.74% / 13.54%**.
- Extra `|` / `&` checks: `(sex == female) | (age < 16)` survived at **71.59%**, `(sex == male) & (age >= 16)` at only **16.39%**.

**Two strongest correlations**, ranked by |r| over all 15 off-diagonal pairs of the 6×6 matrix (`survived, pclass, age, sibsp, parch, fare`; `adult_male` and `alone` excluded):

1. **`pclass` ↔ `fare`: r = −0.548.** The strongest pair. A higher class *number* means a cheaper ticket, so 3rd class pays the least. The sign is negative only because class 1 is the "top" class. The relationship is structural (ticket price was set by class). That is also why a model gets partly redundant information from the two columns.
2. **`sibsp` ↔ `parch`: r = +0.415.** Passengers travelling with siblings or a spouse also tend to travel with parents or children, because families board together. Both columns measure "family on board", which is why Chart 4 combines them into `family_size`.

Third place is `pclass` ↔ `age` (r = −0.411; older passengers were in better classes). It is a close third, and group-median imputation reinforces it slightly. Among correlations with the target, `survived` ↔ `pclass` is the strongest (r = −0.336), followed by `survived` ↔ `fare` (+0.255). Sex does not appear in this matrix only because it is not numeric. As shown above, it is the single strongest driver of survival.

## Task 5 — Multivariate data story

#### Chart 1 — Survival rate by class and sex (`figures/03_story_class_sex.png`)

Sex and class together explain most of who survived. Nearly every woman in 1st and 2nd class survived (97% and 92%), but only half of 3rd-class women did. Men had worse odds in every class, dropping from 37% in 1st class to around 14–16% in 2nd and 3rd. The "women first" rule was applied strongly, but it was weakened by class: a 3rd-class woman had lower odds than a 1st-class woman, though still better odds than any man.

#### Chart 2 — Survival share by age band, split by sex (`figures/04_story_age_sex.png`)

This chart uses observed ages only, so imputed values cannot create a false spike. Age mattered mainly for boys. Boys aged 0–12 survived at 56.8%, almost the same as girls (59.4%), but for males aged 13–18 survival falls to 8.8% and stays around 10–19% for every older band. Among females, survival stays high (roughly 60–100%) at every age. So "children first" in practice meant young boys were saved alongside women, and once a boy was treated as a man his odds collapsed.

#### Chart 3 — Fare by class and outcome (`figures/05_story_fare_class.png`)

Within 1st class, survivors paid noticeably more than those who died (median fare 77.34 vs 44.75), and 2nd class shows the same pattern (21.00 vs 13.00). A higher fare likely meant a better cabin higher up and closer to the boat deck. In 3rd class the two medians are almost identical (8.52 vs 8.05), so paying more within steerage did not buy a better chance. Fare adds information beyond class mainly *inside* the upper classes. The log-style y-axis keeps the 512-pound outliers from flattening everything else.

#### Chart 4 — Survival rate by family size (`figures/06_story_family_size.png`)

Survival is not monotonic in family size. Passengers travelling alone survived at only 30.1% (n = 535, most of them 3rd-class men). Small families of 2–4 did best, peaking at 72.4% for families of four. Large families of 5 or more fell to 13.6–20.0%, probably because keeping a big group together in the evacuation was hard, and these groups were mostly in 3rd class. This non-linear pattern is one reason tree-based models can outperform a linear model on this data.

#### Chart 5 — Survival % by port and class (`figures/07_story_port_class.png`)

Cherbourg (C) passengers survived at 55.4% overall, against 38.9% for Queenstown (Q) and 33.7% for Southampton (S). The heatmap shows this is mostly class composition, not the port itself: 50.6% of Cherbourg boarders were in 1st class, against 93.5% of Queenstown boarders in 3rd class. Within 3rd class, Southampton is still clearly the worst (19%, n = 353), consistent with its many lone adult men. Port therefore acts mostly as a proxy for class and wealth. The Q/1st and Q/2nd cells have n = 2 and n = 3, so they carry no meaningful information.

#### The data story in one paragraph
Survival on the Titanic followed a clear order: **sex first, then class, then age and family context.** Women survived at 74% and men at 19% (Chart 1). Class then split each group: a 1st-class woman almost always survived and a 3rd-class man almost never did. Among children, boys were saved almost as often as girls, but once past about 12 a male's odds were no better than an adult man's (Chart 2). Money helped within the upper classes, where survivors paid higher fares (Chart 3). Travelling alone or in a very large group hurt, while small families did best (Chart 4). Port of embarkation looks important but mostly reflects who boarded where (Chart 5). **The predictive features to carry into modelling are therefore `sex`, `pclass`, `age`, `fare`, and family size (`sibsp`/`parch`), with `embarked` as a weak proxy.**

## Task 6 — Exploratory z-score check (`figures/08_zscore_before_after.png`)

| | age (before) | fare (before) | age (after) | fare (after) |
|---|---|---|---|---|
| mean | 29.0654 | 32.0967 | 0.0000 | 0.0000 |
| std (ddof=0) | 13.2627 | 49.6695 | 1.0000 | 1.0000 |
| min | 0.42 | 0.00 | −2.1598 | −0.6462 |
| max | 80.00 | 512.33 | 3.8404 | 9.6686 |

After z = (x − mean) / std, both `age` and `fare` have mean **0.0000** and population std **1.0000**, matching `StandardScaler` exactly (asserted with `np.allclose`). The histograms have **identical shapes before and after**: z-scoring is a linear shift-and-rescale, so `fare` is still heavily right-skewed (its max is now +9.67 SDs) and `age` still has its imputation spike. Standardisation fixes scale, not skew. Because this check is fitted on the full cleaned data, it is **EDA only**. `02_modeling.ipynb` fits its own `StandardScaler` on the training split alone.

---

# Part B — Predictive modelling (`02_modeling.ipynb`)

## Features and missing values in the pipeline

**Features used:** `pclass, age, sibsp, parch, fare` (numeric) and `sex, embarked` (categorical). The Part A story showed these carry the survival signal. Deliberately **excluded**:
- `alive`: it is the target spelled out as yes/no, so including it would be direct target leakage.
- `who`, `adult_male`, `class`, `embark_town`, `alone`: exact re-encodings of `sex`/`age`, `pclass`, `embarked`, and `sibsp + parch`.
- `deck`: 77% missing. EDA showed its "Missing" level mostly duplicates `pclass` (97.6% of 3rd class has no deck).

**Missing-value choice for modelling.** The only feature still missing here is `age` (177 NaN, 19.87%). Inside the pipeline I impute it with the **training-split median**, a plain median rather than the EDA's (sex, pclass) group median. It is simpler to express as a leak-free `SimpleImputer`, and the scaled `pclass`/`sex` features already sit next to it in every model. `embarked` gets a most-frequent imputer as a safety net for new raw data, even though its two NaN rows were already dropped by the <5% rule.

## Task 7 — Stratified split

**Why stratify?** Task 1 measured a **61.62% / 38.38%** died/survived split, a 1.6 : 1 imbalance. With only 178 test rows, a plain random split could easily give a test set with, say, 34% or 43% survivors. Accuracy, precision and recall would then shift for reasons that have nothing to do with the model, and the train and test sets would describe different populations. `stratify=y` keeps the class ratio fixed in both parts: **38.26% survivors in train and 38.20% in test**, against 38.25% overall. The split happens **before** any imputer, encoder or scaler is fitted.

## Task 8 — Train-only preprocessing

**Leakage check.** The imputer learned the age median (**28.0**) from the 711 training rows only. The scaled training features have mean exactly 0, while the test features do **not** (e.g. −0.072, +0.117). That is what we want to see: the scaler never looked at the test set, so the test set is only transformed. In the rest of the notebook each model is wrapped as `Pipeline([("prep", ColumnTransformer), ("clf", estimator)])`. Calling `.fit(X_train, y_train)` fits the preprocessor on train only, and `.predict(X_test)` applies it in transform-only mode, so the contract is enforced by structure, not by memory.

## Task 9 — Three classifiers and the decision tree (`figures/09_decision_tree.png`)

**Reading the tree.** The root split is `sex_male <= 0.5`, i.e. **sex is the most informative single feature**, exactly as the EDA predicted. Women (left branch, 188 of 253 survived) are split next on `pclass`: 1st/2nd-class women are almost all "survived" (127/134), while for 3rd-class women the tree falls back on `fare` and `embarked`. Men (right, only 84/458 survived) are split first on **age** (`age <= −1.99` in z-units, age ≤ about 3.5 years: little boys, 11 of 13 survived), then by class and fare. The tree is capped at `max_depth=4` with `min_samples_leaf=5` so it stays readable and does not memorise the training set. Thresholds are shown in standardised units because the tree sits after the `StandardScaler`.

## Task 10 — Evaluation (`figures/10_confusion_matrices.png`, `figures/11_roc_curves.png`)

**Test-set results (178 passengers, 68 survivors):**

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | TN / FP / FN / TP |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8146 | 0.7966 | 0.6912 | 0.7402 | **0.8610** | 98 / 12 / 21 / 47 |
| Decision Tree (depth 4) | 0.7978 | 0.7759 | 0.6618 | 0.7143 | 0.8510 | 97 / 13 / 23 / 45 |
| Random Forest (default) | 0.8146 | 0.7778 | **0.7206** | **0.7481** | 0.8283 | 96 / 14 / 19 / 49 |

All three models are much better than the 61.8% "everyone died" baseline. **Logistic Regression ranks passengers best** (highest AUC 0.861). **The untuned Random Forest finds the most survivors** (recall 0.721, 49 true positives) at a small cost in precision. Its training accuracy of 0.985 against 0.815 on test shows it overfits when depth is unlimited, which explains its lowest AUC (0.828) and is why it gets tuned in Task 12. The depth-4 Decision Tree is the weakest on every metric. It trades accuracy for interpretability.

## Task 11 — Imbalance handling

**Class balance:** the training split has 439 died and 272 survived (38.26% minority, 1.61 : 1). SMOTE balanced *only the training data* to 439 : 439. The test set keeps its real 110 : 68 distribution.

| Logistic Regression variant | Precision | Recall | F1 |
|---|---|---|---|
| (a) baseline | **0.7966** | 0.6912 | 0.7402 |
| (b) `class_weight='balanced'` | 0.7083 | **0.7500** | 0.7286 |
| (c) SMOTE (train fold only) | 0.7463 | 0.7353 | **0.7407** |

**Conclusion.** Both techniques do what they are designed to do: they shift the decision boundary toward the minority class, so **recall rises** (0.691 → 0.750 / 0.735) and **precision falls** (0.797 → 0.708 / 0.746). **SMOTE is the best overall strategy here.** It achieves the highest F1 (0.7407) and AUC (0.8668) by gaining 4.4 points of recall for only 5 points of precision. `class_weight='balanced'` over-corrects: it buys the most recall but loses 8.8 precision points, so its F1 actually falls below the baseline. The differences are small (F1 within 0.012) because a 1.6 : 1 imbalance is mild. The choice should therefore follow the business cost: if missing a survivor (a false negative) is the expensive error, use SMOTE or class weights; otherwise the baseline is already well calibrated. SMOTE sits inside an `imblearn` `Pipeline`, so it runs only during `.fit()` on the training rows and never touches the test data.

## Task 12 — GridSearchCV + OOB score

**Tuning result.** Grid: 3 × 4 × 3 = 36 combinations × 5 stratified folds = 180 fits, scored on F1. The folds come from the training split only.

- **Best parameters:** `n_estimators = 100`, `max_depth = 8`, `max_features = None` (all 9 features considered at each split).
- **Best mean CV F1:** 0.7755.
- **OOB score:** **0.8270**. The forest was built as `RandomForestClassifier(oob_score=True, ...)`, so each tree is scored on the ~37% of training rows its bootstrap sample left out. This gives a free validation accuracy.

Limiting depth to 8 is the key change. It stops the default forest from memorising the training data, and on the test set every metric improved except recall, which stayed the same: **accuracy 0.8146 → 0.8258, precision 0.7778 → 0.8033, F1 0.7481 → 0.7597, AUC 0.8283 → 0.8388** (recall held at 0.7206). The OOB accuracy (0.827) and the test accuracy (0.826) agree closely, which suggests the test result is not a lucky split. `max_features=None` beating `sqrt` suggests that with only 9 features, giving every tree access to `sex` matters more than extra decorrelation.

## Task 13 — Regression side-task (`figures/12_regression_residuals.png`)

**Regression: predicting `fare`** from `pclass, age, sibsp, parch, survived, sex, embarked` (9 predictors after encoding; random 80/20 split; same train-only preprocessing pattern).

| MAE | RMSE | R² | Adjusted R² |
|---|---|---|---|
| 21.10 | 41.70 | 0.3482 | 0.3132 |

The model explains about **35% of the variance in fare** (31% after adjusting for 9 predictors on 178 test rows). The largest coefficient is `pclass` (−26.7 per SD). Family size (`parch` +8.9, `sibsp` +7.3) comes next, because a family ticket's fare covers everyone on it, and then Cherbourg embarkation (+9.7). RMSE is about double MAE, so the error is dominated by a few very large misses (the luxury fares).

**Heteroscedasticity: yes, clearly present.** The residuals do not form an even band around zero; they **fan out as the fitted value grows**. The residual standard deviation is about 6–15 for fitted fares below 66, but jumps to **85.6** in the top quintile (66–101), where the 1st-class luxury tickets sit, including one residual above +400. The low end also shows a systematic pattern: fitted values go *negative* (as low as −5.5) for cheap 3rd-class fares, where the residuals are all positive. That is a sign that a linear model on raw fare is mis-specified. A log-transformed target (`log1p(fare)`) or a tree-based regressor would be the natural fix. I kept the plain multivariate linear regression because the task asks for it.

## Task 14 — Model comparison table and recommendation

#### Final model comparison

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.8146 | 0.7966 | 0.6912 | 0.7402 | **0.8610** |
| Decision Tree (depth 4) | 0.7978 | 0.7759 | 0.6618 | 0.7143 | 0.8510 |
| Random Forest (default) | 0.8146 | 0.7778 | 0.7206 | 0.7481 | 0.8283 |
| **Random Forest (tuned)** | **0.8258** | **0.8033** | **0.7206** | **0.7597** | 0.8388 |

| Regression model (target = `fare`) | MAE | RMSE | R² | Adjusted R² |
|---|---|---|---|---|
| Linear Regression | 21.0998 | 41.7022 | 0.3482 | 0.3132 |

*The two groups are on different scales and are not comparable. Classification metrics are proportions in [0, 1] (higher is better). MAE/RMSE are in fare units (lower is better), and R²/Adj. R² are shares of fare variance explained. The notebook table keeps them as two separate column groups for the same reason.*

**Recommendation.** I would deploy the **tuned Random Forest** (`max_depth=8, max_features=None, n_estimators=100`). On the held-out test set it has the best accuracy (**0.8258**), precision (**0.8033**) and F1 (**0.7597**), and it ties for the best recall (**0.7206**). Its OOB accuracy of **0.8270** matches its test accuracy, so the gain over the default forest is not a lucky split. Logistic Regression is a close runner-up: it has the best ranking quality (AUC **0.8610** vs 0.8388) and is fully interpretable. I would pick it instead if the use case needed calibrated probabilities or explainable coefficients rather than the best hard yes/no decisions. The depth-4 Decision Tree is useful as the explanatory diagram above but trails on every metric (F1 0.7143), so it is not a deployment candidate.

## Task 15 — Saved pipeline

`titanic_best_pipeline.joblib` is the **complete fitted `Pipeline`**: the `ColumnTransformer` (median imputer + `StandardScaler` for numeric features, most-frequent imputer + `OneHotEncoder` for `sex`/`embarked`) plus the tuned `RandomForestClassifier`. It is not a bare estimator. After `joblib.load`, it (1) reproduces the in-memory predictions on the test set exactly (asserted), with the same 0.8258 accuracy, and (2) scores brand-new **raw** passengers directly from strings, raw fares, and even a missing age (`NaN`, filled by the stored training median). A 1st-class woman gets p(survive) = 0.98 and a 3rd-class man 0.10. Those outputs are consistent with everything the EDA showed.

To use it elsewhere:
```python
import joblib, pandas as pd
model = joblib.load("titanic_best_pipeline.joblib")
model.predict(pd.DataFrame([{"pclass": 2, "age": 30, "sibsp": 0, "parch": 0, "fare": 13.0, "sex": "female", "embarked": "S"}]))
```
