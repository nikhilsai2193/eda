"""NYC heating complaint closure-time ML dashboard.

The dashboard combines the complete leakage-aware artifact export from
``ml_pipeline_final.run_final_ml`` with the latest one-model-at-a-time
hyperparameter results written by ``ml_hyperparameter_tuning_runner.py``.
It only presents saved predictions and explanations; it does not retrain
models inside Streamlit.
"""

from __future__ import annotations

import json
import os
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


MPL_CACHE = Path("/private/tmp/streamlit_ml_mpl_cache")
MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE))


st.set_page_config(
    page_title="NYC Heating Closure-Time ML",
    page_icon="♨️",
    layout="wide",
    initial_sidebar_state="expanded",
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FULL_ARTIFACT_DIR = PROJECT_ROOT / "Project1" / "streamlit_ml_dashboard_final" / "artifacts"
TUNING_DIR = Path(__file__).resolve().parent / "artifacts" / "hyperparameter_runs"

PALETTE = {
    "blue": "#0072B2", "orange": "#D55E00", "teal": "#009E73",
    "gold": "#E69F00", "purple": "#CC79A7", "navy": "#17324D",
    "gray": "#6B7280", "light_blue": "#56B4E9",
}

NUMERIC_FEATURES = [
    "created_year", "created_month", "created_weekday", "created_hour",
    "month_sin", "month_cos", "weekday_sin", "weekday_cos", "hour_sin",
    "hour_cos", "latitude", "longitude", "city_volume_before",
    "building_same_day_before", "building_prior_365d",
]
CATEGORICAL_FEATURES = [
    "borough", "incident_zip", "community_board", "council_district",
    "police_precinct", "open_data_channel", "problem_detail",
    "additional_details", "street_name", "heat_season", "season",
    "building_history_bucket",
]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
MODEL_LABELS = {
    "CatBoost": "1. CatBoost",
    "Gradient Boosting": "2. Gradient Boosting",
    "LightGBM": "3. LightGBM",
}


def _mtime(path: Path) -> int:
    try:
        return path.stat().st_mtime_ns
    except FileNotFoundError:
        return 0


@st.cache_resource(show_spinner="Loading fitted models…")
def load_bundle(version: int):
    del version
    return joblib.load(FULL_ARTIFACT_DIR / "model_bundle.joblib")


@st.cache_data(show_spinner=False)
def load_csv(path_string: str, version: int) -> pd.DataFrame:
    del version
    path = Path(path_string)
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_parquet(path_string: str, version: int) -> pd.DataFrame:
    del version
    path = Path(path_string)
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


def _clean_text(value) -> str:
    if pd.isna(value) or str(value).strip() == "":
        return "__MISSING__"
    return str(value).strip().upper()


def _normalise_zip(value) -> str:
    try:
        return str(int(float(value)))
    except (TypeError, ValueError):
        return "__MISSING__"


def _rare_pool(frame: pd.DataFrame, rare_levels: dict) -> pd.DataFrame:
    result = frame.copy()
    for column, allowed in rare_levels.items():
        result[column] = result[column].astype(str).fillna("__MISSING__")
        result.loc[~result[column].isin(set(allowed)), column] = "__RARE__"
    return result


def _catboost_frame(frame: pd.DataFrame, numeric_medians: dict,
                    numeric_features: list, categorical_features: list,
                    feature_columns: list) -> pd.DataFrame:
    result = frame[feature_columns].copy()
    for column in numeric_features:
        result[column] = pd.to_numeric(result[column], errors="coerce").fillna(float(numeric_medians[column]))
    for column in categorical_features:
        result[column] = result[column].astype(str).fillna("__MISSING__")
    return result


def _build_input_row(bundle, complaint_date: date, hour: int, borough: str,
                     zip_code: str, community_board: str,
                     council_district: str, police_precinct: str, channel: str,
                     problem_detail: str, latitude: float, longitude: float,
                     city_volume_before: int, building_same_day_before: int,
                     building_prior_365d: int) -> pd.DataFrame:
    dt = pd.Timestamp(complaint_date).replace(hour=int(hour))
    month = int(dt.month)
    weekday = int(dt.dayofweek)
    prior = float(building_prior_365d)
    if prior <= 0:
        history_bucket = "0"
    elif prior <= 4:
        history_bucket = "1-4"
    elif prior <= 19:
        history_bucket = "5-19"
    elif prior <= 99:
        history_bucket = "20-99"
    else:
        history_bucket = "100+"
    row = {
        "created_year": int(dt.year), "created_month": month,
        "created_weekday": weekday, "created_hour": int(hour),
        "month_sin": float(np.sin(2 * np.pi * month / 12)),
        "month_cos": float(np.cos(2 * np.pi * month / 12)),
        "weekday_sin": float(np.sin(2 * np.pi * weekday / 7)),
        "weekday_cos": float(np.cos(2 * np.pi * weekday / 7)),
        "hour_sin": float(np.sin(2 * np.pi * hour / 24)),
        "hour_cos": float(np.cos(2 * np.pi * hour / 24)),
        "latitude": float(latitude), "longitude": float(longitude),
        "city_volume_before": float(city_volume_before),
        "building_same_day_before": float(building_same_day_before),
        "building_prior_365d": prior,
        "borough": _clean_text(borough), "incident_zip": _normalise_zip(zip_code),
        "community_board": _clean_text(community_board),
        "council_district": _clean_text(council_district),
        "police_precinct": _clean_text(police_precinct),
        "open_data_channel": _clean_text(channel),
        "problem_detail": _clean_text(problem_detail),
        # These columns are retained internally because the saved models were
        # trained with them, but they are intentionally not user inputs.
        "additional_details": "__MISSING__",
        "street_name": "__MISSING__",
        "heat_season": "OFF-SEASON" if month in [6, 7, 8, 9] else "HEATING SEASON",
        "season": f"{dt.year if month >= 10 else dt.year - 1}-{(dt.year + 1 if month >= 10 else dt.year) % 100:02d}",
        "building_history_bucket": history_bucket,
    }
    return pd.DataFrame([row], columns=bundle["feature_columns"])


def predict_hours(bundle, model_name: str, row: pd.DataFrame) -> float:
    selected = bundle["models"][model_name]
    model = selected["model"]
    info = selected["info"]
    pooled = _rare_pool(row, info["rare_levels"])
    if info["kind"] == "catboost":
        x = _catboost_frame(
            pooled, info["numeric_medians"], bundle["numeric_features"],
            bundle["categorical_features"], bundle["feature_columns"],
        )
        log_prediction = model.predict(x)
    else:
        log_prediction = model.predict(pooled[bundle["feature_columns"]])
    return float(max(0.0, np.expm1(float(log_prediction[0]))))


def local_shap_values(bundle, model_name: str, row: pd.DataFrame) -> pd.DataFrame:
    """Compute local SHAP in log-hour space for a submitted filing-time row."""
    import shap

    selected = bundle["models"][model_name]
    model = selected["model"]
    info = selected["info"]
    pooled = _rare_pool(row, info["rare_levels"])
    if info["kind"] == "catboost":
        from catboost import Pool
        x = _catboost_frame(
            pooled, info["numeric_medians"], bundle["numeric_features"],
            bundle["categorical_features"], bundle["feature_columns"],
        )
        values = model.get_feature_importance(
            Pool(x, cat_features=bundle["categorical_features"]), type="ShapValues",
        )[:, :-1][0]
        names = bundle["feature_columns"]
    else:
        preprocessor = model.named_steps["preprocess"]
        estimator = model.named_steps["model"]
        transformed = preprocessor.transform(pooled[bundle["feature_columns"]])
        names = list(preprocessor.get_feature_names_out())
        if hasattr(transformed, "toarray"):
            transformed = transformed.toarray()
        if estimator.__class__.__name__ in {"LinearRegression", "Ridge"}:
            explainer = shap.LinearExplainer(estimator, transformed)
        else:
            explainer = shap.TreeExplainer(estimator)
        values = explainer(transformed).values
        if values.ndim == 3:
            values = values[:, :, 0]
        values = values[0]

    base_names = []
    for name in names:
        value = str(name).lower()
        if "__" in value:
            value = value.split("__", 1)[1]
        matched = next(
            (feature for feature in bundle["feature_columns"] if value == feature or value.startswith(feature + "_")),
            value,
        )
        base_names.append(matched)
    result = pd.DataFrame({"feature": base_names, "shap_value": values})
    result = result.groupby("feature", as_index=False)["shap_value"].sum()
    result["direction"] = np.where(
        result["shap_value"] >= 0,
        "Increases predicted log-hours",
        "Decreases predicted log-hours",
    )
    return result.sort_values("shap_value", key=np.abs, ascending=False)


def apply_theme(fig: go.Figure) -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        font=dict(family="Arial", size=13, color=PALETTE["navy"]),
        title_font=dict(size=17), hoverlabel=dict(bgcolor="white"),
        margin=dict(l=55, r=25, t=70, b=55),
    )
    return fig


def metric_chart(comparison: pd.DataFrame, metric: str) -> go.Figure:
    plot = comparison.copy()
    plot["split_label"] = plot["split"].str.title()
    fig = px.bar(
        plot, x="display_model", y=metric, color="split_label", barmode="group",
        color_discrete_map={"Validation": PALETTE["blue"], "Test": PALETTE["orange"]},
        labels={metric: f"{metric.replace('_hours', '')} (hours)", "display_model": "Regression model"},
        title=f"{metric.replace('_hours', '')} comparison: chronological validation vs final test",
        text_auto=".2f",
    )
    fig.update_layout(legend_title_text="Evaluation split", height=420)
    return fig


def tuning_chart(tuning: pd.DataFrame, metric: str) -> go.Figure | None:
    if tuning.empty or not {"model_family", "split", metric}.issubset(tuning.columns):
        return None
    plot = tuning.copy()
    validation = plot.loc[plot["split"].eq("validation")].sort_values(metric).drop_duplicates("model_family")
    chosen_ids = validation["run_id"].tolist()
    plot = plot.loc[plot["run_id"].isin(chosen_ids)].copy()
    plot["display_model"] = plot["model_family"]
    plot["split_label"] = plot["split"].str.title()
    return apply_theme(px.bar(
        plot, x="display_model", y=metric, color="split_label", barmode="group",
        color_discrete_map={"Validation": PALETTE["blue"], "Test": PALETTE["orange"]},
        text_auto=".2f", labels={metric: f"{metric.replace('_hours', '')} (hours)"},
        title=f"Latest hyperparameter-run {metric.replace('_hours', '')}: best run per model family",
    ))


def model_importance(bundle, model_name: str) -> pd.DataFrame:
    selected = bundle["models"][model_name]
    model = selected["model"]
    names = bundle["feature_columns"]
    estimator = model
    values = None
    method = "model importance"
    if hasattr(model, "named_steps"):
        estimator = model.named_steps.get("model", model)
        try:
            names = list(model.named_steps["preprocess"].get_feature_names_out())
        except Exception:
            pass
    if hasattr(model, "get_feature_importance"):
        values = np.asarray(model.get_feature_importance(), dtype=float)
    elif hasattr(estimator, "feature_importances_"):
        values = np.asarray(estimator.feature_importances_, dtype=float)
    elif type(estimator).__name__ == "HistGradientBoostingRegressor" and hasattr(estimator, "_predictors"):
        values = np.zeros(len(names), dtype=float)
        for stage in estimator._predictors:
            for tree in stage:
                nodes = tree.nodes
                split_nodes = nodes[nodes["is_leaf"] == 0]
                for feature_index, gain in zip(split_nodes["feature_idx"], split_nodes["gain"]):
                    if 0 <= int(feature_index) < len(values):
                        values[int(feature_index)] += max(float(gain), 0.0)
        method = "HistGradientBoosting split-gain importance"
    if values is None or len(values) != len(names):
        return pd.DataFrame(columns=["feature", "importance", "method"])
    return pd.DataFrame({"feature": names, "importance": values, "method": method}).sort_values("importance", ascending=False)


def _load_full_artifacts():
    version = _mtime(FULL_ARTIFACT_DIR / "metadata.json")
    bundle = load_bundle(version)
    files = {
        "comparison": "model_comparison.csv", "baselines": "baseline_results.csv",
        "subgroups": "subgroup_metrics.csv", "calibration": "calibration.csv",
        "shap_global": "shap_global.csv", "shap_local": "shap_local.csv",
        "shap_global_by_model": "shap_global_by_model.csv",
        "shap_local_by_model": "shap_local_by_model.csv",
        "permutation": "permutation_importance.csv",
        "eda_borough": "eda_borough.csv", "feature_catalog": "feature_catalog.csv",
    }
    tables = {key: load_csv(str(FULL_ARTIFACT_DIR / filename), version) for key, filename in files.items()}
    tables["predictions"] = load_parquet(str(FULL_ARTIFACT_DIR / "test_predictions_all.parquet"), version)
    tables["eda_sample"] = load_parquet(str(FULL_ARTIFACT_DIR / "eda_sample.parquet"), version)
    metadata_path = FULL_ARTIFACT_DIR / "metadata.json"
    tables["metadata"] = json.loads(metadata_path.read_text())
    tuning_path = TUNING_DIR / "evaluation_results.csv"
    tables["tuning"] = load_csv(str(tuning_path), _mtime(tuning_path))
    return bundle, tables


if not (FULL_ARTIFACT_DIR / "model_bundle.joblib").exists():
    st.error("The complete ML artifact bundle is missing. Run the final ML export before starting this app.")
    st.stop()

bundle, data = _load_full_artifacts()
comparison = data["comparison"].copy()
baselines = data["baselines"]
subgroups = data["subgroups"]
predictions = data["predictions"]
eda_sample = data["eda_sample"]
eda_borough = data["eda_borough"]
tuning = data["tuning"]
metadata = data["metadata"]

available_labels = comparison.loc[comparison["split"].eq("test"), "model"].drop_duplicates().tolist()
available_models = [label.split(". ", 1)[-1] for label in available_labels]
label_by_model = {label.split(". ", 1)[-1]: label for label in available_labels}
default_label = bundle.get("selected_model", available_labels[0])
default_model = default_label.split(". ", 1)[-1]

with st.sidebar:
    st.header("Dashboard controls")
    selected_model = st.selectbox(
        "Model", available_models,
        index=available_models.index(default_model) if default_model in available_models else 0,
    )
    subgroup_dimension = st.selectbox(
        "Subgroup error view", ["Borough", "Heat season", "Year", "Workload band", "Duration bucket"],
    )
    top_k = st.slider("Top features to display", min_value=5, max_value=20, value=10, step=1)
    use_log_axis = st.checkbox("Use log1p scale for EDA distribution", value=True)

selected_label = label_by_model[selected_model]
selected_comparison = comparison.loc[comparison["model"].eq(selected_label)].copy()
selected_predictions = predictions.loc[predictions["model"].eq(selected_label)].copy()
best_test = selected_comparison.loc[selected_comparison["split"].eq("test")].iloc[0]

st.title("NYC 311 Heating Complaint Closure-Time Prediction")
st.subheader("Can we predict how long a heating complaint takes to close at the moment it is filed?")
st.markdown(
    "This dashboard combines filing-time EDA with leakage-aware regression. "
    "The target is `log1p(Resolution Hours)` and all displayed errors are converted back to hours."
)

cols = st.columns(4)
cols[0].metric("Selected model", selected_model)
cols[1].metric("Final-test MAE", f"{best_test['MAE_hours']:.2f} h")
cols[2].metric("Final-test RMSE", f"{best_test['RMSE_hours']:.2f} h")
cols[3].metric("Test requests", f"{metadata['test_rows']:,}")
st.caption(
    f"Chronological test period: {metadata['test_start'][:10]}–{metadata['test_end'][:10]}. "
    "Core diagnostic artifacts are from the full final ML export; the latest Section 18 tuning results are shown separately."
)

tab_eda, tab_ml, tab_explain, tab_predict = st.tabs(["EDA", "ML Results", "Explainability", "Predict"])

with tab_eda:
    st.header("EDA")
    left, right = st.columns(2)
    with left:
        eda_display = eda_sample.copy()
        if use_log_axis:
            eda_display["display_hours"] = np.log1p(eda_display["resolution_hours"].clip(lower=0))
            x_label = "log1p(observed closure time [hours])"
            chart_title = "EDA 1 — Closure-time distribution on log1p scale"
        else:
            cap = float(eda_display["resolution_hours"].quantile(0.995))
            eda_display["display_hours"] = eda_display["resolution_hours"].clip(upper=cap)
            x_label = "Observed closure time (hours; clipped at 99.5th percentile)"
            chart_title = "EDA 1 — Closure-time distribution"
        hist = px.histogram(
            eda_display, x="display_hours", nbins=70,
            color_discrete_sequence=[PALETTE["blue"]],
            labels={"display_hours": x_label, "count": "Requests"}, title=chart_title,
        )
        hist.update_layout(height=430)
        st.plotly_chart(apply_theme(hist), width="stretch")
        st.caption("Answers: Is the target skewed, and why is log1p modeling appropriate?")
    with right:
        eda_plot = eda_borough.loc[eda_borough["borough"].ne("__MISSING__")].copy()
        season_fig = px.bar(
            eda_plot, x="borough", y="median_hours", color="heat_season", barmode="group",
            color_discrete_map={"HEATING SEASON": PALETTE["orange"], "OFF-SEASON": PALETTE["teal"]},
            labels={"borough": "Borough", "median_hours": "Median closure time (hours)", "heat_season": "Season"},
            title="EDA 2 — Median closure time by borough and heat season",
            hover_data={"requests": ":,", "median_hours": ":.2f"},
        )
        season_fig.update_layout(height=430)
        st.plotly_chart(apply_theme(season_fig), width="stretch")
        st.caption("Answers: Do geography and heating-season context relate to typical closure time?")

with tab_ml:
    st.header("ML Results")
    st.caption("Validation selects the candidate; the final test period is evaluated once after refitting on train + validation.")
    ml1, ml2 = st.columns(2)
    with ml1:
        comparison_plot = comparison.copy()
        comparison_plot["display_model"] = comparison_plot["model"].str.split(". ", n=1).str[-1]
        st.plotly_chart(apply_theme(metric_chart(comparison_plot, "MAE_hours")), width="stretch")
        st.caption("ML visual 1 — Which models minimize typical absolute error?")
    with ml2:
        st.plotly_chart(apply_theme(metric_chart(comparison_plot, "RMSE_hours")), width="stretch")
        st.caption("ML visual 2 — Which models are most affected by large misses?")

    benchmark = baselines.loc[baselines["split"].eq("test"), ["model", "MAE_hours", "RMSE_hours"]].copy()
    selected_benchmark = pd.DataFrame([{
        "model": selected_model, "MAE_hours": best_test["MAE_hours"], "RMSE_hours": best_test["RMSE_hours"],
    }])
    benchmark = pd.concat([benchmark, selected_benchmark], ignore_index=True)
    benchmark_fig = px.bar(
        benchmark, x="model", y="MAE_hours", color="model", text_auto=".2f",
        color_discrete_sequence=[PALETTE["gray"], PALETTE["teal"], PALETTE["gold"], PALETTE["orange"]],
        labels={"model": "Test benchmark", "MAE_hours": "Final-test MAE (hours)"},
        title="ML visual 3 — Model versus filing-time median baselines",
        hover_data={"RMSE_hours": ":.2f"},
    )
    benchmark_fig.update_layout(height=410, xaxis_tickangle=-25)
    st.plotly_chart(apply_theme(benchmark_fig), width="stretch")
    st.caption("The baselines use only filing-time information: global, heat-season, and filing-month medians.")
    best_baseline = baselines.loc[baselines["split"].eq("test")].sort_values("MAE_hours").iloc[0]
    if float(best_test["MAE_hours"]) > float(best_baseline["MAE_hours"]):
        st.warning(
            f"The best baseline ({best_baseline['model']}, {best_baseline['MAE_hours']:.2f} h MAE) "
            f"outperforms {selected_model} ({best_test['MAE_hours']:.2f} h MAE) on the final test period."
        )

    if not tuning.empty:
        st.markdown("**Latest hyperparameter-tuning runner output**")
        tuning_mae = tuning_chart(tuning, "MAE_hours")
        if tuning_mae is not None:
            st.plotly_chart(tuning_mae, width="stretch")
        tuning_table = tuning.loc[tuning["split"].eq("test"), ["model_family", "MAE_hours", "RMSE_hours", "R2_hours", "run_id"]].copy()
        st.dataframe(tuning_table.sort_values("MAE_hours").style.format({"MAE_hours": "{:.2f}", "RMSE_hours": "{:.2f}", "R2_hours": "{:.3f}"}), width="stretch", hide_index=True)
        st.caption("These rows come directly from the new notebook’s chronological RandomizedSearchCV runner.")

    sample = selected_predictions.sample(min(12000, len(selected_predictions)), random_state=42)
    actual_pred = px.scatter(
        sample, x="actual_hours", y="predicted_hours", color="heat_season",
        color_discrete_map={"HEATING SEASON": PALETTE["orange"], "OFF-SEASON": PALETTE["teal"]},
        labels={"actual_hours": "Actual closure time (hours)", "predicted_hours": "Predicted closure time (hours)", "heat_season": "Season"},
        title=f"ML visual 4 — Actual vs predicted closure time ({selected_model})", opacity=0.45,
    )
    max_axis = float(max(selected_predictions["actual_hours"].quantile(0.995), selected_predictions["predicted_hours"].quantile(0.995)))
    actual_pred.add_trace(go.Scatter(x=[0, max_axis], y=[0, max_axis], mode="lines", name="Perfect prediction", line=dict(color=PALETTE["gray"], dash="dash")))
    st.plotly_chart(apply_theme(actual_pred), width="stretch")
    st.caption("ML visual 4 — Are predictions close to the ideal actual = predicted line?")

    ml3, ml4 = st.columns(2)
    with ml3:
        residual_fig = px.histogram(
            selected_predictions, x="residual_hours", nbins=80,
            color_discrete_sequence=[PALETTE["purple"]],
            labels={"residual_hours": "Residual (actual − predicted, hours)", "count": "Requests"},
            title="ML visual 5 — Residual distribution",
        )
        residual_fig.add_vline(x=0, line_dash="dash", line_color=PALETTE["gray"])
        st.plotly_chart(apply_theme(residual_fig), width="stretch")
    with ml4:
        bucket = selected_predictions.groupby("duration_bucket", observed=False).agg(
            requests=("actual_hours", "size"), MAE_hours=("absolute_error_hours", "mean"),
            median_error_hours=("absolute_error_hours", "median"),
        ).reset_index()
        bucket_fig = px.bar(
            bucket, x="duration_bucket", y="MAE_hours", color="duration_bucket", text_auto=".1f",
            color_discrete_sequence=[PALETTE["blue"], PALETTE["teal"], PALETTE["gold"], PALETTE["orange"], PALETTE["purple"]],
            labels={"duration_bucket": "Actual-duration bucket", "MAE_hours": "Mean absolute error (hours)"},
            title="ML visual 6 — Error by actual-duration bucket",
            hover_data={"requests": ":,", "median_error_hours": ":.2f"},
        )
        st.plotly_chart(apply_theme(bucket_fig), width="stretch")

    subgroup_plot = subgroups.query("model == @selected_label and dimension == @subgroup_dimension").sort_values("MAE_hours", ascending=False)
    subgroup_fig = px.bar(
        subgroup_plot, x="group", y="MAE_hours", color="RMSE_hours", text_auto=".1f", color_continuous_scale="Blues",
        labels={"group": subgroup_dimension, "MAE_hours": "MAE (hours)", "RMSE_hours": "RMSE (hours)"},
        title=f"ML visual 7 — Error by {subgroup_dimension.lower()}",
        hover_data={"requests": ":,", "MedianAE_hours": ":.2f", "P90AE_hours": ":.2f"},
    )
    st.plotly_chart(apply_theme(subgroup_fig), width="stretch")
    st.caption("The left-side subgroup selector changes this chart among borough, season, year, workload band, and duration bucket.")

    calibration = (
        selected_predictions.assign(prediction_decile=pd.qcut(selected_predictions["predicted_hours"], 10, duplicates="drop"))
        .groupby("prediction_decile", observed=True)
        .agg(requests=("actual_hours", "size"), actual_median_hours=("actual_hours", "median"), predicted_median_hours=("predicted_hours", "median"))
        .reset_index()
    )
    calibration["prediction_decile"] = calibration["prediction_decile"].astype(str)
    calibration_fig = go.Figure()
    calibration_fig.add_trace(go.Scatter(
        x=calibration["predicted_median_hours"], y=calibration["actual_median_hours"], mode="lines+markers",
        name="Observed test deciles", line=dict(color=PALETTE["blue"], width=3),
        customdata=calibration[["prediction_decile", "requests"]],
        hovertemplate="Predicted median: %{x:.1f} h<br>Actual median: %{y:.1f} h<br>Requests: %{customdata[1]:,}<extra></extra>",
    ))
    calibration_max = float(calibration[["predicted_median_hours", "actual_median_hours"]].max().max())
    calibration_fig.add_trace(go.Scatter(x=[0, calibration_max], y=[0, calibration_max], mode="lines", name="Perfect calibration", line=dict(color=PALETTE["gray"], dash="dash")))
    calibration_fig.update_layout(title="ML visual 8 — Calibration by predicted-hours decile", xaxis_title="Median predicted closure time (hours)", yaxis_title="Median actual closure time (hours)", height=440)
    st.plotly_chart(apply_theme(calibration_fig), width="stretch")

with tab_explain:
    st.header("Model explainability")
    st.caption(
        "Visual 9 updates with the selected model. SHAP values are in log-hours: "
        "positive values push the prediction longer; negative values push it shorter. "
        "Visual 10 measures the increase in held-out MAE after shuffling each feature."
    )
    saved_shap_global = data["shap_global_by_model"]
    if saved_shap_global.empty:
        saved_shap_global = data["shap_global"]
    selected_shap_global = saved_shap_global.loc[saved_shap_global["model"].eq(selected_label)].copy()
    if selected_shap_global.empty:
        st.info("No saved global SHAP artifact is available for this model.")
    else:
        shap_plot = selected_shap_global.sort_values("mean_abs_shap", ascending=True).tail(top_k)
        shap_fig = px.bar(
            shap_plot, x="mean_abs_shap", y="feature", orientation="h", color="mean_abs_shap", color_continuous_scale="Blues",
            labels={"mean_abs_shap": "Mean absolute SHAP value (log-hours)", "feature": "Filing-time feature"},
            title=f"ML visual 9 — Global SHAP importance ({selected_model})",
        )
        shap_fig.update_layout(height=max(420, top_k * 30), coloraxis_showscale=False)
        st.plotly_chart(apply_theme(shap_fig), width="stretch")
        st.caption("Global SHAP ranks the filing-time information most used by the currently selected model.")

    permutation = data["permutation"].loc[data["permutation"]["model"].eq(selected_label)].copy()
    if permutation.empty:
        st.info("No held-out permutation-importance artifact is available for this model.")
    else:
        permutation_plot = permutation.sort_values("importance_mae_increase", ascending=True).tail(top_k)
        permutation_fig = px.bar(
            permutation_plot, x="importance_mae_increase", y="feature", orientation="h",
            error_x="importance_std", color="importance_mae_increase",
            color_continuous_scale="Oranges",
            labels={
                "importance_mae_increase": "Increase in MAE after shuffling (hours)",
                "importance_std": "Repeat-to-repeat standard deviation",
                "feature": "Filing-time feature",
            },
            title=f"ML visual 10 — Permutation feature importance ({selected_model})",
            hover_data={"baseline_mae_hours": ":.2f", "n_rows": ":,", "repeats": True},
        )
        permutation_fig.update_layout(height=max(420, top_k * 30), coloraxis_showscale=False)
        st.plotly_chart(apply_theme(permutation_fig), width="stretch")
        st.caption(
            "Permutation importance is computed on a held-out final-test sample. "
            "A larger MAE increase means the selected model relies more on that filing-time feature; "
            "it is predictive importance, not a causal effect."
        )

    fitted_importance = model_importance(bundle, selected_label)
    if not fitted_importance.empty:
        st.markdown(f"**Additional model-specific importance — {selected_model}**")
        st.caption(f"Method: {fitted_importance['method'].iloc[0]}. This is a predictive importance score, not a causal effect.")
        importance_view = fitted_importance.head(top_k).sort_values("importance")
        importance_fig = px.bar(
            importance_view, x="importance", y="feature", orientation="h",
            labels={"importance": "Importance", "feature": "Filing-time feature"},
            title=f"Fitted-model importance ({selected_model})",
        )
        st.plotly_chart(apply_theme(importance_fig), width="stretch")

with tab_predict:
    st.header("Predict closure time for a new complaint")
    st.info("Use only information available when the complaint is filed. Street Name and Additional Details are hidden because they are not required user inputs; the saved model receives its trained missing-value sentinel internally.")
    with st.form("prediction_form"):
        p1, p2, p3 = st.columns(3)
        with p1:
            complaint_date = st.date_input("Created date", value=date(2024, 1, 15))
            complaint_hour = st.slider("Created hour", min_value=0, max_value=23, value=12, step=1)
            borough_options = [x for x in sorted(eda_borough["borough"].dropna().unique()) if x != "__MISSING__"]
            borough = st.selectbox("Borough", borough_options or ["BRONX"])
            zip_code = st.text_input("Incident ZIP", value="10458")
        with p2:
            problem_detail = st.text_input("Problem detail", value="NO HEAT")
            channel = st.selectbox("Open-data channel", ["PHONE", "ONLINE", "MOBILE", "OTHER"])
            community_board = st.text_input("Community board", value="07 BRONX")
            council_district = st.text_input("Council district", value="15")
        with p3:
            police_precinct = st.text_input("Police precinct", value="48")
            latitude = st.number_input("Latitude", value=40.88, format="%.5f")
            longitude = st.number_input("Longitude", value=-73.88, format="%.5f")
            city_volume_before = st.number_input("Requests already filed today", min_value=0, max_value=100000, value=500, step=10)
        q1, q2 = st.columns(2)
        with q1:
            building_same_day_before = st.number_input("Same-building requests already filed today", min_value=0, max_value=1000, value=0, step=1)
        with q2:
            building_prior_365d = st.number_input("Same-building requests in prior 365 days", min_value=0, max_value=10000, value=2, step=1)
        submitted = st.form_submit_button("Predict closure time")

    if submitted:
        input_row = _build_input_row(
            bundle, complaint_date, complaint_hour, borough, zip_code, community_board,
            council_district, police_precinct, channel, problem_detail, latitude,
            longitude, city_volume_before, building_same_day_before, building_prior_365d,
        )
        predicted_hours = predict_hours(bundle, selected_label, input_row)
        residuals = selected_predictions["residual_hours"]
        low = max(0.0, predicted_hours + float(residuals.quantile(0.10)))
        high = max(low, predicted_hours + float(residuals.quantile(0.90)))
        result_cols = st.columns(4)
        result_cols[0].metric("Predicted closure time", f"{predicted_hours:.1f} hours")
        result_cols[1].metric("Approx. 80% empirical interval", f"{low:.1f}–{high:.1f} h")
        result_cols[2].metric("72-hour flag", "SLOW (>72 h)" if predicted_hours > 72 else "UNDER 72 h")
        result_cols[3].metric("Model", selected_model)
        try:
            local = local_shap_values(bundle, selected_label, input_row).head(top_k).sort_values("shap_value")
            local_fig = px.bar(
                local, x="shap_value", y="feature", orientation="h", color="shap_value",
                color_continuous_scale=[[0, PALETTE["teal"]], [0.5, "#D9D9D9"], [1, PALETTE["orange"]]],
                labels={"shap_value": "Local SHAP value (log-hours)", "feature": "Filing-time feature"},
                title="Local SHAP explanation for this complaint",
            )
            local_fig.update_layout(height=max(420, top_k * 30), coloraxis_showscale=False)
            st.plotly_chart(apply_theme(local_fig), width="stretch")
            st.caption("This explains which filing-time inputs pushed this complaint’s predicted closure time up or down.")
        except Exception as exc:
            st.warning(f"The prediction was generated, but local SHAP could not be computed for this model: {exc}")

    with st.expander("Method and data audit"):
        audit_table = pd.DataFrame([
            ["Target", metadata["target"]],
            ["Training target transform", metadata["target_transform"]],
            ["Training-only 99.5th percentile", f"{metadata['train_only_cap_hours']:.2f} hours"],
            ["Selected model families", ", ".join(metadata["selected_models"])],
            ["Excluded post-filing fields", ", ".join(metadata["excluded_leaky_columns"])],
            ["Aggregate workload features", "Cumulative-before-filing only"],
        ], columns=["Audit item", "Value"])
        st.table(audit_table)
        st.caption("Imputation, rare-level pooling, encoding, model selection, and workload thresholds are fitted without using the final test period.")
