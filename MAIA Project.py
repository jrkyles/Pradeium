#Adam Yonas, yonas@usc.edu

import base64
import io
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from flask import Flask, jsonify, redirect, render_template, request, session, url_for, Response, send_file
from flask_cors import CORS
import os
import seaborn as sn
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

MPL_CACHE_DIR = os.path.join(os.path.dirname(__file__), ".matplotlib_cache")
os.makedirs(MPL_CACHE_DIR, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", MPL_CACHE_DIR)

API_STRING_COLUMNS = [
    "NOI",
    "IE (UNADJUSTED)",
    "Interest Expense",
    "Total Debt",
    "Total Equity",
    "Cash",
    "Total Assets",
]
API_ETF_COLUMNS = [
    "Sticker",
    "Full name",
    "Focus",
    "Numerical_Rating",
    "Residential",
    "Office",
    "Retail",
    "Industrial",
]
API_MODEL_FEATURES = [
    "NOI",
    "Interest Expense",
    "Total Debt",
    "Total Equity",
    "Cash",
    "Total Assets",
    "Interest_Expense_Coverage_Ratio",
    "Total_Leverage_Ratio",
    "Debt_to_Equity_Ratio",
    "EBITDA_Asset_Ratio",
    "EBITDA_Equity_Ratio",
]
API_CLASS_ORDER = ["A-", "BBB+", "BBB", "BBB-", "BB+", "BB", "B"]

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret_key'
CORS(app)
@app.route("/")
def home2():
    pd.set_option('display.max_columns', None)

    df = pd.read_csv('CRE_Credit_Rating.csv')
    #print(df.shape)
    #trying to find the columns with the highest correlation to the target
    #df_n = pd.get_dummies(data=df, columns=[' loan_status', ' self_employed', ' education'], drop_first=True)
    # print(df_n.columns)
    # print(abs(df_n.corr()[' loan_status_ Rejected']).sort_values(ascending=False))

    #deleting target
    del df['Rating']
    ETF_specific_columns = ['Sticker', 'Full name', 'Focus', 'Numerical_Rating']
    for i in ETF_specific_columns:
        del df[i]
    #converting string data to numerical data
    string_columns = [' NOI ', ' IE (UNADJUSTED) ', ' Interest Expense ', ' Total Debt ', ' Total Equity ', ' Cash ',
                      ' Total Assets ']
    for i in string_columns:
        df[i] = df[i].str.replace(',', '', regex=False).astype(float)

    #aggregating statistical properties of the dataset as variables to display to the user
    features_list = list(df.columns)
    Net_Operating_Income = ("$" + str(round(df[' NOI '].median(), 2)))
    Interest_Coverage_Ratio = (str(round(df['Interest_Expense_Coverage_Ratio'].median(), 2)))
    Leverage_Ratio = (str(round(df['Net_Debt_Leverage_Ratio'].median(), 2)))
    Defaults = (str(round(df['5Y_Default_Rate'].median(), 2)))
    #returning the home html template
    return render_template("home2.html",
        features= features_list,
        message="Welcome to Praedium!",
        NOI = Net_Operating_Income,
        Int_Coverage = Interest_Coverage_Ratio,
        Leverage = Leverage_Ratio,
        Defaults = Defaults)

@app.route("/submit_feature_inputs", methods=["POST"])
def submit_locale():
    pd.set_option('display.max_columns', None)

    df = pd.read_csv('CRE_Credit_Rating.csv')
    # converting string data to numerical data
    string_columns = [' NOI ', ' IE (UNADJUSTED) ', ' Interest Expense ', ' Total Debt ', ' Total Equity ', ' Cash ',
                      ' Total Assets ']
    for i in string_columns:
        df[i] = df[i].str.replace(',', '', regex=False).astype(float)



    df_n = df
    ETF_specific_columns = ['Sticker', 'Full name', 'Focus', 'Numerical_Rating', 'Rating', 'Residential', 'Office',
                            'Retail', 'Industrial']
    features = []
    for i in df.columns:
        if i not in ETF_specific_columns:
            features.append(i)
    print(features)
    print(df.info())

    count = 0
    fig, axes = plt.subplots(3, 5, figsize=(20, 12))  # Create grid
    axes = axes.flatten()
    for count, i in enumerate(features):
        ax = axes[count]
        feature_range = np.linspace(0, df[i].max(), 48)
        sn.stripplot(data=df, x=i, y='Rating', hue='Rating', alpha=1, ax=ax,
                     palette={'A-': 'blue',
                              'BBB+': 'green',
                              'BBB': 'orange',
                              'BBB-': 'indigo',
                              'BB+': 'black',
                              'BB': 'yellow',
                              'B': 'red'}, hue_order=['A-', 'BBB+', 'BBB', 'BBB-', 'BB+', 'BB', 'B'], legend=False,
                     order=['A-', 'BBB+', 'BBB', 'BBB-', 'BB+', 'BB', 'B'])
        ax.set_title(str(i))
    plt.title('Credit Rating Correlation')
    '''max_x = df[i].max()
    #print(max_x)
     plt.xticks(np.linspace(0, max_x, 6))  # 6 evenly spaced x-ticks

    # Set y-ticks
     max_y = feature_range.max()
     plt.yticks(np.linspace(0, max_y, 6))'''
    handles, labels = ax.get_legend_handles_labels()

    # Place the legend outside the grid
    fig.legend(handles, labels, loc='upper center', ncol=7, bbox_to_anchor=(0.5, 1.05))

    plt.tight_layout()
    plt.show()
#saving the image so i can display it to the html file
    buf = io.BytesIO()
    buf = io.BytesIO()
    fig.savefig(buf,
                format="png",
                dpi=100,  # or whatever resolution you like
                bbox_inches='tight'  # trim extra white space
                )
    buf.seek(0)
    session['img_data'] = base64.b64encode(buf.read()).decode('utf-8')

    return render_template("feature.html", chart_data=session['img_data'])

@app.route("/submit_projection", methods=["POST"])
def submit_feature():
    #target
    df = pd.read_csv('CRE_Credit_Rating.csv')
    # Cleaning up the data set
    # df_n = pd.get_dummies(data=df, columns=[' loan_status', ' self_employed', ' education'], drop_first=True)
    # print(df_n.columns)
    # print(abs(df_n.corr()[' loan_status_ Rejected']).sort_values(ascending=False))
    # dropping the bottom 4 features with the lowest correlation with the target

    ETF_specific_columns = ['Sticker', 'Full name', 'Focus', 'Numerical_Rating']
    for i in ETF_specific_columns:
        del df[i]
    string_columns = [' NOI ', ' IE (UNADJUSTED) ', ' Interest Expense ', ' Total Debt ', ' Total Equity ', ' Cash ',
                      ' Total Assets ']
    for i in string_columns:
        df[i] = df[i].str.replace(',', '', regex=False).astype(float)

    y = df['Rating']
    del df['Rating']
    filtered_features = [' NOI ', ' Interest Expense ', ' Total Debt ', ' Total Equity ', ' Cash ', ' Total Assets ',
                         'Interest_Expense_Coverage_Ratio', 'Total_Leverage_Ratio', 'Debt_to_Equity_Ratio',
                         'EBITDA_Asset_Ratio', 'EBITDA_Equity_Ratio']
    for i in df.columns:
        if i not in filtered_features:
            del df[i]
    x = df
    #creating the array of sample data
    sample_data_list = []
    for i in x.columns :
        # handling erroneous missing input values or non numerical
        # ******ERROR HANDLING********
        val_str = request.form.get(i)
        if not val_str:  # covers None or empty string
            print(f"Missing input for: {i}")
            return redirect(url_for("home2"))
        try:
            val = float(val_str.replace(",", ""))
        except ValueError:
            print(f"Invalid number for: {i} → '{val_str}'")
            return redirect(url_for("home2"))
        sample_data_list.append(request.form[i])
    sample_array =np.array([sample_data_list])

#transforming into a dataframe for ML
    sample_df = pd.DataFrame(data=sample_array, columns=x.columns,)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(x)
    # gives the method the mean and stdv of EACH column in the df

    x_normalized = scaler.transform(x)
    # uses the mean and stdv found in .fit() to shift the data how we saw above
    # returns as a numpy array

    x_normalized = pd.DataFrame(data=scaler.transform(x), columns=x.columns)
    # changes the numpy array into a dataframe
    print(x.columns)
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression()
    model.fit(x_normalized, y)
    sample_df_scaled = pd.DataFrame(scaler.transform(sample_df), columns=sample_df.columns)
    y_pred = model.predict_proba(sample_df_scaled)
    Y_predict = model.predict(sample_df_scaled)
#predicting the probability of loan approval based on sample data
    probabilities = y_pred[0]
#plotting the probabilities as a bar chart
    proba_df = pd.DataFrame({
        'Category': model.classes_,
        'Probability': y_pred[0]  # first and only row of probabilities
    })

    # --- Plot ---
    fig=plt.figure(figsize=(10, 6))
    sn.barplot(data=proba_df, x='Category', y='Probability', palette='muted',
               order=['A-', 'BBB+', 'BBB', 'BBB-', 'BB+', 'BB', 'B'])

    plt.title('Probability Distribution Across Credit Categories')
    plt.ylabel('Predicted Probability')
    plt.xlabel('Credit Rating Category')
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    img_data = base64.b64encode(buf.read()).decode('utf-8')
    #use argxmax
    idx = np.argmax(y_pred[0])

    # 2) extract the class name and its probability
    predicted_category = model.classes_[idx]
    predicted_probability = y_pred[0][idx]
    print(predicted_category)
    print(predicted_probability)
    return render_template("projection.html", rating=predicted_category, probability=predicted_probability, chart_data=img_data)


def load_api_dataset():
    df = pd.read_csv("CRE_Credit_Rating.csv")
    df = df.rename(columns=lambda col: col.strip())
    for column in API_STRING_COLUMNS:
        if column in df.columns:
            df[column] = (
                df[column]
                .astype(str)
                .str.replace(",", "", regex=False)
                .replace("nan", np.nan)
            )
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def build_api_story_stats(df: pd.DataFrame) -> Dict[str, str]:
    return {
        "medianNOI": f"${round(df['NOI'].median(), 2):,}" if "NOI" in df else "N/A",
        "medianInterestCoverage": str(
            round(df["Interest_Expense_Coverage_Ratio"].median(), 2)
        )
        if "Interest_Expense_Coverage_Ratio" in df
        else "N/A",
        "medianLeverage": str(
            round(df["Net_Debt_Leverage_Ratio"].median(), 2)
        )
        if "Net_Debt_Leverage_Ratio" in df
        else "N/A",
        "medianDefaultRate": str(
            round(df["5Y_Default_Rate"].median(), 2)
        )
        if "5Y_Default_Rate" in df
        else "N/A",
    }


def prepare_api_model_data(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    for column in API_ETF_COLUMNS:
        if column in work.columns:
            del work[column]
    work = work.dropna(subset=API_MODEL_FEATURES + ["Rating"])
    return work


def build_api_feature_metadata(df: pd.DataFrame) -> List[Dict[str, float]]:
    metadata = []
    for feature in API_MODEL_FEATURES:
        if feature not in df.columns:
            continue
        series = df[feature].dropna()
        metadata.append(
            {
                "key": feature,
                "label": feature.replace("_", " "),
                "min": round(float(series.min()), 2),
                "max": round(float(series.max()), 2),
                "median": round(float(series.median()), 2),
            }
        )
    return metadata


def build_api_visual_features(df: pd.DataFrame) -> List[str]:
    return [
        column
        for column in df.columns
        if column not in API_ETF_COLUMNS + ["Rating"]
    ]


def train_api_model(df: pd.DataFrame):
    y = df["Rating"]
    x = df[API_MODEL_FEATURES]
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    model = LogisticRegression(max_iter=2000, multi_class="auto")
    model.fit(x_scaled, y)
    return {"scaler": scaler, "model": model}


def render_api_probability_chart(distribution: pd.DataFrame) -> str:
    fig = plt.figure(figsize=(10, 6))
    sn.barplot(
        data=distribution,
        x="Category",
        y="Probability",
        palette="muted",
        order=[c for c in API_CLASS_ORDER if c in distribution["Category"].values],
    )
    plt.title("Probability Distribution Across Credit Categories")
    plt.ylabel("Predicted Probability")
    plt.xlabel("Credit Rating Category")
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=144, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def render_api_correlation_chart(df: pd.DataFrame, features: List[str]) -> str:
    fig, axes = plt.subplots(3, 5, figsize=(24, 14))
    axes = axes.flatten()
    for idx, feature in enumerate(features[: len(axes)]):
        ax = axes[idx]
        sn.stripplot(
            data=df,
            x=feature,
            y="Rating",
            hue="Rating",
            alpha=0.9,
            ax=ax,
            palette={
                "A-": "steelblue",
                "BBB+": "seagreen",
                "BBB": "orange",
                "BBB-": "indigo",
                "BB+": "black",
                "BB": "gold",
                "B": "crimson",
            },
            hue_order=API_CLASS_ORDER,
            legend=False,
            order=API_CLASS_ORDER,
        )
        ax.set_title(feature)
    for extra_ax in axes[len(features) :]:
        extra_ax.axis("off")
    fig.legend(API_CLASS_ORDER, loc="upper center", ncol=7, bbox_to_anchor=(0.5, 1.02))
    plt.suptitle("Credit Rating Correlation", fontsize=20)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=144, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


api_dataset = load_api_dataset()
api_story_stats = build_api_story_stats(api_dataset)
api_model_df = prepare_api_model_data(api_dataset)
api_feature_metadata = build_api_feature_metadata(api_model_df)
api_visual_features = build_api_visual_features(api_model_df)
api_artifacts = train_api_model(api_model_df)
api_correlation_chart_cache: str | None = None


@app.route("/api/summary", methods=["GET"])
def api_summary():
    return jsonify(
        {
            "message": "Welcome to Praedium!",
            "stats": api_story_stats,
            "inputs": api_feature_metadata,
            "features": api_visual_features,
        }
    )


@app.route("/api/predict", methods=["POST"])
def api_predict():
    payload = request.get_json(silent=True)
    if not payload:
        return jsonify({"error": "Request body must be JSON."}), 400
    missing = [feature for feature in API_MODEL_FEATURES if feature not in payload]
    if missing:
        return jsonify({"error": "Missing required features.", "fields": missing}), 400
    try:
        sample = [
            float(str(payload[feature]).replace(",", "")) for feature in API_MODEL_FEATURES
        ]
    except (ValueError, TypeError):
        return jsonify({"error": "All feature values must be numeric."}), 400

    sample_df = pd.DataFrame([sample], columns=API_MODEL_FEATURES)
    scaler = api_artifacts["scaler"]
    model = api_artifacts["model"]
    sample_scaled = scaler.transform(sample_df)
    probabilities = model.predict_proba(sample_scaled)[0]
    prediction_index = int(np.argmax(probabilities))
    predicted_category = model.classes_[prediction_index]
    distribution_df = pd.DataFrame({
        "Category": model.classes_,
        "Probability": [float(prob) for prob in probabilities],
    })
    chart_base64 = render_api_probability_chart(distribution_df)
    return jsonify(
        {
            "rating": predicted_category,
            "probability": float(probabilities[prediction_index]),
            "distribution": [
                {"category": cat, "probability": float(prob)}
                for cat, prob in zip(model.classes_, probabilities)
            ],
            "chart": chart_base64,
        }
    )


@app.route("/api/correlations", methods=["GET"])
def api_correlations():
    global api_correlation_chart_cache
    if api_correlation_chart_cache is None:
        api_correlation_chart_cache = render_api_correlation_chart(
            api_model_df,
            [feature for feature in api_model_df.columns if feature != "Rating"],
        )
    return jsonify({"chart": api_correlation_chart_cache})


@app.route('/<path:path>')
def catch_all(path):
    return redirect(url_for("home2"))

if __name__ == "__main__":
    # print(db_get_locales())
    app.secret_key = os.urandom(12)
    app.run(port=5006, debug=True)