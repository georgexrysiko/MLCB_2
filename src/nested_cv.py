import warnings
import numpy as np
import pandas as pd
import optuna
from sklearn.model_selection import StratifiedKFold
import sklearn.base as skbase
from sklearn.metrics import roc_auc_score

from functions import compute_metrics, build_pipeline, bootstrap_median_ci

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")

class RepeatedNestedCV:
    """
    Repeated Nested Cross-Validation for binary classification.

    Parameters
    ----------
    estimators : dict  {name: unfitted_estimator}
    param_spaces : dict  {name: callable(trial) -> dict}
        Each value is an Optuna suggest function that returns a hyperparam dict.
        Pass None or {} to skip inner-loop tuning for that estimator.
    R : int   Number of repetitions (default 10)
    N : int   Outer folds (default 5)
    K : int   Inner folds (default 3)
    n_trials : int   Optuna trials per inner fold (default 50)
    inner_metric : str  Metric optimised in inner loop (default "AUC")
    base_seed : int  Root seed for reproducibility (default 42)
    """

    def __init__(
        self,
        estimators: dict,
        param_spaces: dict,
        categorical_cols: list,
        numerical_cols: list,
        R: int = 10,
        N: int = 5,
        K: int = 3,
        n_trials: int = 50,
        inner_metric: str = "AUC",
        base_seed: int = 42,
    ):
        self.estimators = estimators # {name: estimator}
        self.param_spaces  = param_spaces # {name: optuna_space_fn | None}
        self.R = R
        self.N = N
        self.K = K
        self.n_trials = n_trials
        self.inner_metric  = inner_metric
        self.base_seed = base_seed
        self.categorical_cols = categorical_cols
        self.numerical_cols   = numerical_cols

        # Populated after fit()
        self.raw_scores_   = {} # {name: list of metric dicts} len = R*N
        self.results_      = None # DataFrame summary

    def fit(self, X: pd.DataFrame, y: pd.Series, tune: bool = True):
        """
        Run rnCV.

        Parameters
        ----------
        X : feature DataFrame
        y : binary target Series
        tune : bool
            If False, skip the inner loop (no hyperparameter tuning).
            Useful for the baseline comparison step.
        """
        self.raw_scores_ = {name: [] for name in self.estimators}

        for r in range(self.R):
            outer_seed = self.base_seed + r * 100
            outer_cv = StratifiedKFold(n_splits=self.N, shuffle=True, random_state=outer_seed)

            for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
                # Change to .iloc for DataFrames
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

                for name, base_estimator in self.estimators.items():
                    if tune and self.param_spaces.get(name):
                        best_params = self._inner_loop(
                            name, base_estimator, X_train, y_train,
                            inner_seed=outer_seed + fold_idx + 1,
                        )
                        estimator = self._clone_with_params(base_estimator, best_params)
                    else:
                        estimator = skbase.clone(base_estimator)

                    # Pass columns to the pipeline builder
                    pipe = build_pipeline(estimator, self.categorical_cols, self.numerical_cols)
                    pipe.fit(X_train, y_train)

                    y_pred = pipe.predict(X_test)
                    y_proba = pipe.predict_proba(X_test)[:, 1]

                    metrics = compute_metrics(y_test, y_pred, y_proba)
                    self.raw_scores_[name].append(metrics)

        self.results_ = self._summarise()
        return self

    def summary(self) -> pd.DataFrame:
        """Return the results DataFrame (median ± 95 % CI for each metric)."""
        if self.results_ is None:
            raise RuntimeError("Call fit() first.")
        return self.results_

    def get_scores(self, name: str) -> pd.DataFrame:
        """Return all RxN metric values for a given estimator as a DataFrame."""
        return pd.DataFrame(self.raw_scores_[name])

    # Internal: Optuna inner loop
    def _inner_loop(self, name, base_estimator, X_train, y_train, inner_seed):
        """Run Optuna over K inner folds; return best hyperparameters."""
        space_fn = self.param_spaces[name]

        def objective(trial):
            params = space_fn(trial)
            estimator = self._clone_with_params(base_estimator, params)

            pipe = build_pipeline(estimator, self.categorical_cols, self.numerical_cols)

            inner_cv  = StratifiedKFold(
                n_splits=self.K, shuffle=True, random_state=inner_seed
            )
            fold_scores = []
            for tr_idx, val_idx in inner_cv.split(X_train, y_train):
                Xtr, Xval = X_train.iloc[tr_idx], X_train.iloc[val_idx]
                ytr, yval = y_train.iloc[tr_idx], y_train.iloc[val_idx]
                pipe.fit(Xtr, ytr)
                y_proba = pipe.predict_proba(Xval)[:, 1]
                score = roc_auc_score(yval, y_proba)
                fold_scores.append(score)
            return np.mean(fold_scores)

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=inner_seed),
        )
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)
        return study.best_params

    # Internal: clone estimator with given params
    @staticmethod
    def _clone_with_params(base_estimator, params):
        import sklearn.base as skbase
        est = skbase.clone(base_estimator)
        est.set_params(**params)
        return est

    def _summarise(self) -> pd.DataFrame:
        metric_names = list(next(iter(self.raw_scores_.values()))[0].keys())
        rows = []
        for name, scores_list in self.raw_scores_.items():
            df_scores = pd.DataFrame(scores_list)
            row = {"Estimator": name}
            for m in metric_names:
                vals = df_scores[m].values
                med = np.median(vals)
                lo, hi = bootstrap_median_ci(vals, seed=self.base_seed)
                row[f"{m}_median"] = round(med, 4)
                row[f"{m}_CI_lo"] = round(lo,  4)
                row[f"{m}_CI_hi"] = round(hi,  4)
            rows.append(row)
        return pd.DataFrame(rows).set_index("Estimator")
