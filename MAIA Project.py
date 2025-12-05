# Adam Yonas, yonas@usc.edu

import base64
import io
import os
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
from flask import (
    Flask,
    jsonify,
    redirect,
    render_template,
    request,
    session,
    url_for,
)
from flask_cors import CORS
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

MPL_CACHE_DIR = os.path.join(os.path.dirname(__file__), ".matplotlib_cache")
os.makedirs(MPL_CACHE_DIR, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", MPL_CACHE_DIR)

BASE_DIR = os.path.dirname(__file__)
DATASET_PATH = os.path.join(BASE_DIR, "Praedium_dataset_20251124.csv")
TARGET_COLUMN = "loan_ever_60_days_delinquent"
REFERENCE_YEAR = 2025

INPUT_NUMERIC_FEATURES = [
    "loan_acquisition_upb",
    "original_interest_rate",
    "amortization_term",
    "loan_acquisition_ltv",
    "underwritten_dscr",
    "property_acquisition_total_unit_count",
    "number_of_properties_at_acquisition",
    "physical_occupancy",
    "note_rate",
    "property_age",
]

CATEGORICAL_FEATURES = [
    "amortization_type",
    "interest_type",
    "lien_position",
    "specific_property_type",
    "property_state",
]

MODEL_NUMERIC_FEATURES = [
    "log_loan_acquisition_upb",
    "original_interest_rate",
    "amortization_term",
    "loan_acquisition_ltv",
    "underwritten_dscr",
    "property_acquisition_total_unit_count",
    "number_of_properties_at_acquisition",
    "physical_occupancy",
    "note_rate",
    "property_age",
]

MODEL_FEATURE_COLUMNS = MODEL_NUMERIC_FEATURES + CATEGORICAL_FEATURES

PRETTY_LABELS: Dict[str, str] = {
    "loan_acquisition_upb": "Loan UPB ($)",
    "log_loan_acquisition_upb": "Log Loan UPB",
    "original_interest_rate": "Original Interest Rate (%)",
    "amortization_term": "Amortization Term (months)",
    "loan_acquisition_ltv": "Acquisition LTV (%)",
    "underwritten_dscr": "Underwritten DSCR",
    "property_acquisition_total_unit_count": "Total Units",
    "number_of_properties_at_acquisition": "Number of Properties",
    "physical_occupancy": "Physical Occupancy (%)",
    "note_rate": "Note Rate (%)",
    "property_age": "Property Age (years)",
    "amortization_type": "Amortization Type",
    "interest_type": "Interest Type",
    "lien_position": "Lien Position",
    "specific_property_type": "Property Type",
    "property_state": "Property State",
}


def format_currency(value: float) -> str:
    return f"${value:,.0f}"


def format_percent(value: float, digits: int = 2) -> str:
    return f"{round(value, digits):.{digits}f}%"


def load_dataset() -> pd.DataFrame:
    raw_df = pd.read_csv(DATASET_PATH)
    return raw_df


def prepare_model_dataframe(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()
    df["year_built_clean"] = df["year_built"].astype(str).str.strip()
    df["year_built_num"] = pd.to_numeric(df["year_built_clean"], errors="coerce")
    df["property_age"] = REFERENCE_YEAR - df["year_built_num"]

    df["loan_acquisition_upb"] = (
        df["loan_acquisition_upb"]
        .astype(str)
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
    )
    df["loan_acquisition_upb"] = pd.to_numeric(
        df["loan_acquisition_upb"], errors="coerce"
    )
    df["log_loan_acquisition_upb"] = np.where(
        df["loan_acquisition_upb"] > 0,
        np.log(df["loan_acquisition_upb"]),
        np.nan,
    )

    df[TARGET_COLUMN] = (
        df[TARGET_COLUMN]
        .astype(str)
        .str.upper()
        .map({"N": 0, "Y": 1})
    )

    keep_columns = (
        INPUT_NUMERIC_FEATURES
        + CATEGORICAL_FEATURES
        + ["log_loan_acquisition_upb", TARGET_COLUMN]
    )
    df = df[[col for col in keep_columns if col in df.columns]]
    df = df.dropna(subset=[TARGET_COLUMN])
    return df


def build_input_metadata(df: pd.DataFrame) -> List[Dict]:
    metadata: List[Dict] = []
    for feature in INPUT_NUMERIC_FEATURES:
        if feature not in df.columns:
            continue
        series = df[feature].dropna()
        metadata.append(
            {
                "key": feature,
                "label": PRETTY_LABELS.get(feature, feature),
                "type": "numeric",
                "min": round(float(series.min()), 2),
                "max": round(float(series.max()), 2),
                "median": round(float(series.median()), 2),
            }
        )
    for feature in CATEGORICAL_FEATURES:
        if feature not in df.columns:
            continue
        options = sorted(df[feature].dropna().unique().tolist())
        metadata.append(
            {
                "key": feature,
                "label": PRETTY_LABELS.get(feature, feature),
                "type": "categorical",
                "options": options,
            }
        )
    return metadata


def build_story_stats(df: pd.DataFrame) -> Dict[str, str]:
    stats = {
        "medianLTV": "N/A",
        "medianDSCR": "N/A",
        "medianLoanSize": "N/A",
        "medianNoteRate": "N/A",
        "medianPropertyAge": "N/A",
    }
    if "loan_acquisition_ltv" in df:
        stats["medianLTV"] = format_percent(float(df["loan_acquisition_ltv"].median()))
    if "underwritten_dscr" in df:
        stats["medianDSCR"] = f"{float(df['underwritten_dscr'].median()):.2f}x"
    if "loan_acquisition_upb" in df:
        stats["medianLoanSize"] = format_currency(float(df["loan_acquisition_upb"].median()))
    if "note_rate" in df:
        stats["medianNoteRate"] = format_percent(float(df["note_rate"].median()))
    if "property_age" in df:
        stats["medianPropertyAge"] = f"{float(df['property_age'].median()):.0f} yrs"
    return stats


def train_model(df: pd.DataFrame) -> Pipeline:
    y = df[TARGET_COLUMN]
    x = df[MODEL_FEATURE_COLUMNS]

    numeric_features = MODEL_NUMERIC_FEATURES
    categorical_features = CATEGORICAL_FEATURES

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )

    clf = LogisticRegression(max_iter=500, class_weight="balanced")

    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("clf", clf),
        ]
    )
    model.fit(x, y)
    return model


def render_probability_gauge(pd_hat: float) -> str:
    pd_pct = pd_hat * 100
    if pd_pct >= 80:
        color = "#d62728"
    elif pd_pct >= 56:
        color = "#ff7f0e"
    else:
        color = "#2ca02c"

    fig, ax = plt.subplots(figsize=(6, 1.5))
    ax.barh([0], [pd_pct], color=color, height=0.4)
    ax.set_xlim(0, 100)
    ax.set_yticks([])
    ax.axvline(56, color="grey", linestyle="--", linewidth=1)
    ax.axvline(80, color="grey", linestyle="--", linewidth=1)
    ax.text(28, 0.45, "A", ha="center", va="bottom", fontsize=9)
    ax.text(68, 0.45, "B", ha="center", va="bottom", fontsize=9)
    ax.text(90, 0.45, "C", ha="center", va="bottom", fontsize=9)
    ax.set_xlabel("Predicted probability of 60+ day delinquency (%)")
    ax.set_title(f"PD = {pd_pct:.1f}%")
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)
    return base64.b64encode(buf.read()).decode("utf-8")


def render_correlation_chart(df: pd.DataFrame, features: List[str]) -> str:
    n_rows, n_cols = 4, 4
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(22, 18))
    axes = axes.flatten()

    tmp = df.copy()
    tmp["target_num"] = tmp[TARGET_COLUMN].astype(float)
    tmp["target_str"] = tmp[TARGET_COLUMN].map({0: "Non-delinquent", 1: "Delinquent"})
    numeric_features = [f for f in features if f in INPUT_NUMERIC_FEATURES]

    for ax, col in zip(axes, features):
        if col in numeric_features:
            sn.stripplot(
                data=tmp,
                x="target_str",
                y=col,
                hue="target_str",
                palette={"Non-delinquent": "#4c72b0", "Delinquent": "#dd8452"},
                dodge=False,
                alpha=0.45,
                ax=ax,
            )
            ax.set_xlabel("")
            ax.set_title(PRETTY_LABELS.get(col, col))
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(
                    handles[:2],
                    ["Non-delinquent", "Delinquent"],
                    fontsize=8,
                    loc="upper right",
            )
        else:
            stats = (
                tmp.groupby(col)["target_num"]
                .agg(default_rate="mean", count="count")
                .reset_index()
                .sort_values("default_rate", ascending=False)
            )
            sn.barplot(
                data=stats,
                x=col,
                y="default_rate",
                ax=ax,
                color="#4c72b0",
            )
            ax.set_xlabel("")
            ax.set_ylabel("Delinquency rate")
            ax.set_title(PRETTY_LABELS.get(col, col))
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
            for idx, row in stats.iterrows():
                ax.text(
                    idx,
                    row["default_rate"] + 0.005,
                    f"{int(row['count'])}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    for extra_ax in axes[len(features) :]:
        fig.delaxes(extra_ax)

    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)
    return base64.b64encode(buf.read()).decode("utf-8")


def grade_from_pd(pd_hat: float) -> str:
    pd_pct = pd_hat * 100
    if pd_pct >= 80:
        return "C"
    if pd_pct >= 56:
        return "B"
    return "A"


def prepare_sample_dataframe(payload: Dict[str, str | float]) -> pd.DataFrame:
    missing = [
        field for field in INPUT_NUMERIC_FEATURES + CATEGORICAL_FEATURES if field not in payload
    ]
    if missing:
        raise ValueError(f"Missing required features: {', '.join(missing)}")

    numeric_values: Dict[str, float] = {}
    for field in INPUT_NUMERIC_FEATURES:
        try:
            numeric_values[field] = float(str(payload[field]).replace(",", ""))
        except (TypeError, ValueError):
            raise ValueError(f"Invalid numeric value for {field}")

    sample = {**numeric_values}
    for field in CATEGORICAL_FEATURES:
        value = payload.get(field, "")
        if value is None or str(value).strip() == "":
            raise ValueError(f"Missing categorical value for {field}")
        sample[field] = str(value).strip()

    sample_df = pd.DataFrame([sample])
    sample_df["log_loan_acquisition_upb"] = np.where(
        sample_df["loan_acquisition_upb"] > 0,
        np.log(sample_df["loan_acquisition_upb"]),
        np.nan,
    )

    return sample_df[MODEL_FEATURE_COLUMNS]


def score_sample(sample_df: pd.DataFrame, model: Pipeline) -> Dict:
    probabilities = model.predict_proba(sample_df)[0]
    class_order = list(model.classes_)
    delinquent_index = class_order.index(1)
    pd_hat = float(probabilities[delinquent_index])

    risk_grade = grade_from_pd(pd_hat)
    gauge = render_probability_gauge(pd_hat)
    distribution = [
        {"category": "Non-delinquent", "probability": round(1 - pd_hat, 4)},
        {"category": "60+ day delinquent", "probability": round(pd_hat, 4)},
    ]
    return {
        "risk_grade": risk_grade,
        "pd_hat": pd_hat,
        "distribution": distribution,
        "chart": gauge,
    }


raw_dataset = load_dataset()
model_dataframe = prepare_model_dataframe(raw_dataset)
input_metadata = build_input_metadata(model_dataframe)
story_stats = build_story_stats(model_dataframe)
correlation_features = [
    col
    for col in INPUT_NUMERIC_FEATURES + CATEGORICAL_FEATURES
    if col in model_dataframe.columns
]
model_pipeline = train_model(model_dataframe)
correlation_chart_cache: str | None = None

app = Flask(__name__)
app.config["SECRET_KEY"] = "secret_key"
CORS(app)


@app.route("/")
def home():
    return render_template(
        "home_praedium.html",
        features=correlation_features,
        message="Welcome to Praedium!",
        LTV=story_stats["medianLTV"],
        DSCR=story_stats["medianDSCR"],
        Rate=story_stats["medianNoteRate"],
        Age=story_stats["medianPropertyAge"],
        Loan_Size=story_stats["medianLoanSize"],
    )


@app.route("/submit_feature_inputs", methods=["POST"])
def submit_feature_inputs():
    global correlation_chart_cache
    if correlation_chart_cache is None:
        correlation_chart_cache = render_correlation_chart(
            model_dataframe, correlation_features
        )
    numeric_features = [f for f in INPUT_NUMERIC_FEATURES if f in model_dataframe.columns]
    categorical_options = {
        feature: sorted(model_dataframe[feature].dropna().unique().tolist())
        for feature in CATEGORICAL_FEATURES
        if feature in model_dataframe.columns
    }
    session["img_data"] = correlation_chart_cache
    return render_template(
        "feature_praedium.html",
        numeric_features=numeric_features,
        categorical_features=categorical_options,
        chart_data=correlation_chart_cache,
    )


@app.route("/submit_projection", methods=["POST"])
def submit_projection():
    form_payload = {key: value for key, value in request.form.items()}
    try:
        sample_df = prepare_sample_dataframe(form_payload)
        scored = score_sample(sample_df, model_pipeline)
    except ValueError:
        return redirect(url_for("home"))

    return render_template(
        "projection_praedium.html",
        risk_grade=scored["risk_grade"],
        probability=round(scored["pd_hat"] * 100, 2),
        chart_data=scored["chart"],
    )


@app.route("/api/summary", methods=["GET"])
def api_summary():
    return jsonify(
        {
            "message": "Welcome to Praedium!",
            "stats": story_stats,
            "inputs": input_metadata,
            "features": correlation_features,
        }
    )


@app.route("/api/predict", methods=["POST"])
def api_predict():
    payload = request.get_json(silent=True)
    if not payload:
        return jsonify({"error": "Request body must be JSON."}), 400
    try:
        sample_df = prepare_sample_dataframe(payload)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    scored = score_sample(sample_df, model_pipeline)
    return jsonify(
        {
            "rating": scored["risk_grade"],
            "probability": scored["pd_hat"],
            "distribution": scored["distribution"],
            "chart": scored["chart"],
        }
    )


@app.route("/api/correlations", methods=["GET"])
def api_correlations():
    global correlation_chart_cache
    if correlation_chart_cache is None:
        correlation_chart_cache = render_correlation_chart(
            model_dataframe, correlation_features
        )
    return jsonify({"chart": correlation_chart_cache})


@app.route("/<path:path>")
def catch_all(path):
    return redirect(url_for("home"))


if __name__ == "__main__":
    app.secret_key = os.urandom(12)
    app.run(port=5006, debug=True)
