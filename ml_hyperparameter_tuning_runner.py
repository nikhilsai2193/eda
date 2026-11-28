"""One-at-a-time chronological hyperparameter tuning for the heating project.

The notebook imports this module so the large-data runner is reusable without
duplicating the implementation in several notebook cells. It deliberately
uses a single PredefinedSplit: train rows are the fitting fold and validation
rows are the scoring fold. The final test period is never passed to the search.
"""

from __future__ import annotations

import gc
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score
from sklearn.model_selection import PredefinedSplit, RandomizedSearchCV

# Keep matplotlib's import (via the existing feature module) from trying to
# create a cache under a read-only home directory during notebook runs.
_MPL_CACHE = Path('/private/tmp/ml_hyperparameter_tuning_mpl_cache')
_MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault('MPLCONFIGDIR', str(_MPL_CACHE))

from ml_pipeline_final import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    TARGET,
    apply_rare_levels,
    build_model_frame,
    fit_rare_levels,
    make_catboost,
    make_sklearn_pipeline,
)


RANDOM_STATE = 42
DEFAULT_OUTPUT_ROOT = Path('Project1/streamlit_ml_dashboard_final/artifacts/hyperparameter_runs')


def _safe_name(value: str) -> str:
    return re.sub(r'[^A-Za-z0-9_.-]+', '_', str(value)).strip('_')


def _chronological_positions(n_rows: int, max_rows: int | None) -> np.ndarray:
    if max_rows is None or n_rows <= max_rows:
        return np.arange(n_rows, dtype=int)
    return np.unique(np.linspace(0, n_rows - 1, int(max_rows), dtype=int))


def _make_estimator(model_name: str, params: dict | None = None):
    params = params or {}
    if model_name == 'CatBoost':
        return make_catboost(params)
    # RandomizedSearchCV exposes sklearn-pipeline parameters as
    # ``model__parameter``. The factory accepts estimator-level names, so
    # remove that prefix before constructing the fresh estimator used by the
    # validation and final refits.
    estimator_params = {
        str(key).removeprefix('model__'): value for key, value in params.items()
    }
    return make_sklearn_pipeline(model_name, estimator_params)


def _predict_log(estimator, X):
    return np.asarray(estimator.predict(X), dtype=float)


def _mae_hours_scorer(estimator, X, y_log):
    prediction_hours = np.maximum(np.expm1(_predict_log(estimator, X)), 0)
    actual_hours = np.maximum(np.expm1(np.asarray(y_log, dtype=float)), 0)
    return -mean_absolute_error(actual_hours, prediction_hours)


def _rmse_hours_scorer(estimator, X, y_log):
    prediction_hours = np.maximum(np.expm1(_predict_log(estimator, X)), 0)
    actual_hours = np.maximum(np.expm1(np.asarray(y_log, dtype=float)), 0)
    return -float(np.sqrt(mean_squared_error(actual_hours, prediction_hours)))


def _evaluate_predictions(model_name, split, actual_hours, actual_log, prediction_log, run_id, best_params):
    actual_hours = np.asarray(actual_hours, dtype=float)
    actual_log = np.asarray(actual_log, dtype=float)
    prediction_log = np.asarray(prediction_log, dtype=float)
    prediction_hours = np.maximum(np.expm1(prediction_log), 0)
    return {
        'run_id': run_id,
        'model_family': model_name,
        'split': split,
        'n_rows': int(len(actual_hours)),
        'MAE_hours': float(mean_absolute_error(actual_hours, prediction_hours)),
        'RMSE_hours': float(np.sqrt(mean_squared_error(actual_hours, prediction_hours))),
        'R2_hours': float(r2_score(actual_hours, prediction_hours)),
        'MedianAE_hours': float(median_absolute_error(actual_hours, prediction_hours)),
        'Log_RMSE': float(np.sqrt(mean_squared_error(actual_log, prediction_log))),
        'best_params_json': json.dumps(best_params, sort_keys=True, default=str),
    }


def _append_csv(path: Path, rows):
    incoming = pd.DataFrame(rows)
    if path.exists():
        incoming = pd.concat([pd.read_csv(path), incoming], ignore_index=True)
    incoming.to_csv(path, index=False)


def run_model_search(
    clean: pd.DataFrame,
    model_name: str,
    param_distributions: dict,
    n_iter: int = 20,
    search_rows_per_split: int | None = 250_000,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
):
    """Tune one model, refit it, evaluate it, save artifacts, and release memory.

    ``search_rows_per_split`` reduces only the search workload while retaining
    evenly spaced chronological coverage. The validation and final test
    metrics use all rows in their respective periods.
    """
    started = time.time()
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + _safe_name(model_name)

    frame = train = validation = test = fit_frame = None
    search = validation_model = final_model = None
    train_model = validation_model_frame = fit_model_frame = test_model_frame = None
    X_search = y_search_log = None

    try:
        frame = build_model_frame(clean)
        n_rows = len(frame)
        train_end = int(n_rows * 0.70)
        validation_end = int(n_rows * 0.85)
        train = frame.iloc[:train_end].copy()
        validation = frame.iloc[train_end:validation_end].copy()
        test = frame.iloc[validation_end:].copy()

        y_train_hours = train[TARGET].to_numpy(dtype=float)
        y_validation_hours = validation[TARGET].to_numpy(dtype=float)
        y_test_hours = test[TARGET].to_numpy(dtype=float)
        y_train_log = np.log1p(y_train_hours)
        y_validation_log = np.log1p(y_validation_hours)
        y_test_log = np.log1p(y_test_hours)

        # Rare-level pooling is learned from the search-training period only.
        train_levels = fit_rare_levels(train)
        train_model = apply_rare_levels(train, train_levels)
        validation_model_frame = apply_rare_levels(validation, train_levels)
        train_positions = _chronological_positions(len(train), search_rows_per_split)
        validation_positions = _chronological_positions(len(validation), search_rows_per_split)
        search_train = train_model.iloc[train_positions]
        search_validation = validation_model_frame.iloc[validation_positions]
        X_search = pd.concat([
            search_train[FEATURE_COLUMNS],
            search_validation[FEATURE_COLUMNS],
        ], ignore_index=True)
        y_search_log = np.concatenate([
            y_train_log[train_positions],
            y_validation_log[validation_positions],
        ])
        fold_assignment = np.concatenate([
            np.full(len(search_train), -1, dtype=int),
            np.zeros(len(search_validation), dtype=int),
        ])
        chronological_cv = PredefinedSplit(test_fold=fold_assignment)

        estimator = _make_estimator(model_name)
        total_combinations = int(np.prod([len(values) for values in param_distributions.values()]))
        fit_kwargs = {'cat_features': CATEGORICAL_FEATURES} if model_name == 'CatBoost' else {}
        search = RandomizedSearchCV(
            estimator=estimator,
            param_distributions=param_distributions,
            n_iter=min(int(n_iter), total_combinations),
            scoring={'MAE_hours': _mae_hours_scorer, 'RMSE_hours': _rmse_hours_scorer},
            refit=False,
            cv=chronological_cv,
            random_state=RANDOM_STATE,
            n_jobs=1,
            pre_dispatch=1,
            error_score='raise',
            return_train_score=False,
            verbose=2,
        )
        search.fit(X_search, y_search_log, **fit_kwargs)
        # With refit=False, choose explicitly from validation results. The
        # scorer values are negative because sklearn maximizes scores, so the
        # rank columns already encode the correct direction.
        cv_results = pd.DataFrame(search.cv_results_)
        best_cv_row = (
            cv_results
            .sort_values(['rank_test_MAE_hours', 'rank_test_RMSE_hours'])
            .iloc[0]
        )
        best_params = dict(best_cv_row['params'])

        # Unbiased validation result: refit the selected parameters on train only.
        validation_model = _make_estimator(model_name, best_params)
        validation_model.fit(train_model[FEATURE_COLUMNS], y_train_log, **fit_kwargs)
        validation_prediction_log = _predict_log(validation_model, validation_model_frame[FEATURE_COLUMNS])

        # Final test result: refit selected parameters on train plus validation.
        fit_frame = pd.concat([train, validation], ignore_index=True)
        fit_levels = fit_rare_levels(fit_frame)
        fit_model_frame = apply_rare_levels(fit_frame, fit_levels)
        test_model_frame = apply_rare_levels(test, fit_levels)
        y_fit_log = np.log1p(fit_frame[TARGET].to_numpy(dtype=float))
        final_model = _make_estimator(model_name, best_params)
        final_model.fit(fit_model_frame[FEATURE_COLUMNS], y_fit_log, **fit_kwargs)
        test_prediction_log = _predict_log(final_model, test_model_frame[FEATURE_COLUMNS])

        evaluation = pd.DataFrame([
            _evaluate_predictions(model_name, 'validation', y_validation_hours, y_validation_log, validation_prediction_log, run_id, best_params),
            _evaluate_predictions(model_name, 'test', y_test_hours, y_test_log, test_prediction_log, run_id, best_params),
        ])
        model_path = output_root / f'{_safe_name(model_name)}_{run_id}.joblib'
        joblib.dump({
            'model': final_model,
            'model_family': model_name,
            'best_params': best_params,
            'feature_columns': FEATURE_COLUMNS,
            'categorical_features': CATEGORICAL_FEATURES,
            'rare_levels': fit_levels,
            'target': TARGET,
            'target_transform': 'log1p; inverse transform with expm1',
        }, model_path, compress=3)
        (output_root / f'{_safe_name(model_name)}_{run_id}_best_params.json').write_text(
            json.dumps(best_params, indent=2, default=str)
        )
        pd.DataFrame(search.cv_results_).to_csv(
            output_root / f'{_safe_name(model_name)}_{run_id}_cv_results.csv', index=False
        )
        _append_csv(output_root / 'evaluation_results.csv', evaluation)
        _append_csv(output_root / 'best_parameters.csv', [{
            'run_id': run_id,
            'model_family': model_name,
            'best_params_json': json.dumps(best_params, sort_keys=True, default=str),
            'search_rows_train': len(search_train),
            'search_rows_validation': len(search_validation),
            'n_iter': search.n_iter,
            'runtime_seconds': time.time() - started,
        }])
        print(f'Completed {model_name}: {model_path}')
        print(evaluation.round(3).to_string(index=False))
        return {
            'run_id': run_id,
            'model_family': model_name,
            'best_params': best_params,
            'model_path': str(model_path),
            'evaluation': evaluation,
            'runtime_seconds': time.time() - started,
        }
    finally:
        # Explicitly release large references between one-model runs.
        del frame, train, validation, test, fit_frame
        del search, validation_model, final_model
        del train_model, validation_model_frame, fit_model_frame, test_model_frame
        del X_search, y_search_log
        gc.collect()


def compare_completed_runs(output_root: str | Path = DEFAULT_OUTPUT_ROOT):
    """Return all runs and the top three distinct model families by validation score."""
    evaluation_path = Path(output_root) / 'evaluation_results.csv'
    if not evaluation_path.exists():
        return pd.DataFrame(), pd.DataFrame()
    evaluations = pd.read_csv(evaluation_path)
    validation = (
        evaluations.loc[evaluations['split'].eq('validation')]
        .sort_values(['MAE_hours', 'RMSE_hours'])
        .drop_duplicates('model_family')
    )
    return evaluations, validation.head(3)
