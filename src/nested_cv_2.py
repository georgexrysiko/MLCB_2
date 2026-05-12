import warnings
import numpy as np
import pandas as pd
import optuna
import sklearn.base as skbase
from sklearn.model_selection import StratifiedKFold
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
from functions import compute_metrics, build_pipeline, bootstrap_median_ci

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")


def _select_features(estimator, X_train, y_train, cat_cols, num_cols, n_features, seed):
    """
    Model-agnostic feature selection via permutation importance.
    Evaluated on the FULL pipeline so original categorical features are scored.
    """
    pipe = build_pipeline(skbase.clone(estimator), cat_cols, num_cols)
    pipe.fit(X_train, y_train)

    # Pass the ENTIRE fitted pipeline and the raw X_train dataframe
    result = permutation_importance(
        pipe,
        X_train,
        y_train,
        n_repeats=5,
        random_state=seed,
        scoring="roc_auc",
    )

    feature_names = list(X_train.columns)
    importances = result.importances_mean
    top_indices = np.argsort(importances)[::-1][:n_features]
    
    # Return the string names of the top features
    top_names = [feature_names[i] for i in top_indices]
    return top_names


class RepeatedNestedCVFeatureSelection:
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
        use_feature_selection: bool = False,
        n_features: int = 8,
    ):
        self.estimators = estimators
        self.param_spaces = param_spaces
        self.categorical_cols = categorical_cols
        self.numerical_cols = numerical_cols
        self.R = R
        self.N = N
        self.K = K
        self.n_trials = n_trials
        self.inner_metric = inner_metric
        self.base_seed = base_seed
        self.use_feature_selection = use_feature_selection
        self.n_features = n_features

        self.raw_scores_ = {}
        self.results_ = None
        self.feature_selection_ = {}

    def fit(self, X: pd.DataFrame, y: pd.Series, tune: bool = True):
        feature_names = list(X.columns)

        self.raw_scores_ = {name: [] for name in self.estimators}

        if self.use_feature_selection:
            self.feature_selection_ = {
                name: {f: 0 for f in feature_names}
                for name in self.estimators
            }

        for r in range(self.R):
            outer_seed = self.base_seed + r
            outer_cv   = StratifiedKFold(n_splits=self.N, shuffle=True, random_state=outer_seed)

            # Using .iloc to keep Pandas DataFrames intact
            for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
                X_train_full = X.iloc[train_idx]
                X_test_full = X.iloc[test_idx]
                y_train = y.iloc[train_idx]
                y_test = y.iloc[test_idx]

                for name, base_estimator in self.estimators.items():

                    # ── Step 1: Feature selection (outer train fold only) ──
                    if self.use_feature_selection:
                        sel_names = _select_features(
                            base_estimator,
                            X_train_full, y_train,
                            self.categorical_cols, self.numerical_cols,
                            self.n_features,
                            seed=outer_seed + fold_idx,
                        )
                        # Slice DataFrames using column names
                        X_train = X_train_full[sel_names]
                        X_test = X_test_full[sel_names]

                        for fname in sel_names:
                            self.feature_selection_[name][fname] += 1
                    else:
                        X_train = X_train_full
                        X_test = X_test_full

                    # Filter the column lists to only include what's currently in X_train
                    cur_cats = [c for c in self.categorical_cols if c in X_train.columns]
                    cur_nums = [c for c in self.numerical_cols if c in X_train.columns]

                    # ── Step 2: Hyperparameter tuning (inner loop) ─────────
                    if tune and self.param_spaces.get(name):
                        best_params = self._inner_loop(
                            name, base_estimator, X_train, y_train,
                            cur_cats, cur_nums,
                            inner_seed=outer_seed + fold_idx + 1,
                        )
                        estimator = self._clone_with_params(base_estimator, best_params)
                    else:
                        estimator = skbase.clone(base_estimator)

                    # ── Step 3: Train & Evaluate ───────────────────────────
                    pipe = build_pipeline(estimator, cur_cats, cur_nums)
                    pipe.fit(X_train, y_train)

                    y_pred  = pipe.predict(X_test)
                    y_proba = pipe.predict_proba(X_test)[:, 1]

                    self.raw_scores_[name].append(compute_metrics(y_test, y_pred, y_proba))

        self.results_ = self._summarise()
        return self

    def summary(self) -> pd.DataFrame:
        if self.results_ is None:
            raise RuntimeError("Call fit() first.")
        return self.results_

    def get_scores(self, name: str) -> pd.DataFrame:
        return pd.DataFrame(self.raw_scores_[name])

    def get_selection_frequency(self, name: str) -> pd.DataFrame:
        if not self.use_feature_selection:
            raise RuntimeError("use_feature_selection=False — no data recorded.")

        counts = self.feature_selection_[name]
        total  = self.R * self.N
        df = pd.DataFrame({
            "selected_count": counts,
            "selected_%": {k: round(v / total * 100, 1) for k, v in counts.items()},
        }).sort_values("selected_%", ascending=False)
        return df

    def _inner_loop(self, name, base_estimator, X_train, y_train, cur_cats, cur_nums, inner_seed):
        space_fn = self.param_spaces[name]

        def objective(trial):
            params = space_fn(trial)
            estimator = self._clone_with_params(base_estimator, params)
            pipe = build_pipeline(estimator, cur_cats, cur_nums)

            inner_cv = StratifiedKFold(n_splits=self.K, shuffle=True, random_state=inner_seed)
            fold_scores = []
            for tr_idx, val_idx in inner_cv.split(X_train, y_train):
                Xtr, Xval = X_train.iloc[tr_idx], X_train.iloc[val_idx]
                ytr, yval = y_train.iloc[tr_idx], y_train.iloc[val_idx]
                pipe.fit(Xtr, ytr)
                y_proba = pipe.predict_proba(Xval)[:, 1]
                fold_scores.append(roc_auc_score(yval, y_proba))
            return np.mean(fold_scores)

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=inner_seed),
        )
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)
        return study.best_params

    @staticmethod
    def _clone_with_params(base_estimator, params):
        est = skbase.clone(base_estimator)
        est.set_params(**params)
        return est

    def _summarise(self) -> pd.DataFrame:
        metric_names = list(next(iter(self.raw_scores_.values()))[0].keys())
        rows = []
        for name, scores_list in self.raw_scores_.items():
            df_s = pd.DataFrame(scores_list)
            row = {"Estimator": name}
            for m in metric_names:
                vals = df_s[m].values
                med  = np.median(vals)
                lo, hi = bootstrap_median_ci(vals, seed=self.base_seed)
                row[f"{m}_median"] = round(med, 4)
                row[f"{m}_CI_lo"] = round(lo,  4)
                row[f"{m}_CI_hi"] = round(hi,  4)
            rows.append(row)
        return pd.DataFrame(rows).set_index("Estimator")