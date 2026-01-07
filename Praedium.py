#Adam Yonas, yonas@usc.edu

import base64
import io



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from flask import Flask, redirect, render_template, request, session, url_for, Response, send_file
import os
import seaborn as sn

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret_key'
pd.set_option('display.max_columns', None)

data = pd.read_csv('Praedium_dataset_20251124.csv')
data.head()

    #trans
data["year_built_clean"] = (
        data["year_built"]
        .astype(str)
        .str.strip()
    )

    # Convert to numeric; "multiple assets" → NaN
data["year_built_num"] = pd.to_numeric(data["year_built_clean"], errors="coerce")

REFERENCE_YEAR = 2025
data["property_age"] = REFERENCE_YEAR - data["year_built_num"]
data["loan_acquisition_upb"] = (
        data["loan_acquisition_upb"]
        .astype(str)
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
    )

data["loan_acquisition_upb"] = pd.to_numeric(data["loan_acquisition_upb"], errors="coerce")
data['log_loan_acquisition_upb'] = np.log(data['loan_acquisition_upb'])

    # filtering for only the relevant columns
filtered_features = ['log_loan_acquisition_upb', 'amortization_type', 'interest_type', 'original_interest_rate',
                         'amortization_term', 'lien_position',
                         'loan_acquisition_ltv', 'underwritten_dscr', 'property_acquisition_total_unit_count',
                         'number_of_properties_at_acquisition', 'specific_property_type',
                         'property_state', 'physical_occupancy', 'note_rate', 'loan_ever_60_days_delinquent',
                         'property_age']
    # transforming year_built into property age


    #aggregating statistical properties of the dataset as variables to display to the user
numeric_cols = [
    'log_loan_acquisition_upb',
    'original_interest_rate',
    'amortization_term',
    'loan_acquisition_ltv',
    'underwritten_dscr',
    'property_acquisition_total_unit_count',
    'number_of_properties_at_acquisition',
    'physical_occupancy',
    'note_rate',
    'property_age']

filtered_data = data[filtered_features]
tmp = filtered_data.copy()
for col in numeric_cols:
    median_value = filtered_data[col].median()
    filtered_data[col] = filtered_data[col].fillna(median_value)

filtered_data["loan_ever_60_days_delinquent"] = (
    filtered_data["loan_ever_60_days_delinquent"]
        .astype(str)
        .str.upper()
        .map({"N": 0, "Y": 1})
)

#seperating the target
y = filtered_data['loan_ever_60_days_delinquent']
del filtered_data['loan_ever_60_days_delinquent']



@app.route("/")
def home3():
    drop_down_features = ['log_loan_acquisition_upb', 'amortization_type', 'interest_type', 'original_interest_rate',
                         'amortization_term', 'lien_position',
                         'loan_acquisition_ltv', 'underwritten_dscr', 'property_acquisition_total_unit_count',
                         'number_of_properties_at_acquisition', 'specific_property_type',
                         'property_state', 'physical_occupancy', 'note_rate',
                         'property_age']
    LTV = ("$" + str(round(data['loan_acquisition_ltv'].median(), 2)))
    DSCR = (str(round(data['underwritten_dscr'].median(), 2)))
    Age = (str(round(data['property_age'].median(), 2)))
    Loan_Size = (str(round(data['loan_acquisition_upb'].median(), 2)))
    Rate = (str(round(data['note_rate'].median(), 2)))
    #returning the home html template
    return render_template("home_praedium.html",
        features= drop_down_features,
        message="Welcome to Praedium!",
        LTV = LTV,
        DSCR = DSCR,
        Rate = Rate,
        Age = Age,
        Loan_Size = Loan_Size
        )

@app.route("/submit_feature_inputs", methods=["POST"])
def submit_locale():
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    from pandas.api.types import is_numeric_dtype

    # --------------------
    # CONFIG
    # --------------------
    target_col = "loan_ever_60_days_delinquent"

    # Optional prettier names (add more if you want)
    pretty_names = {
        "log_loan_acquisition_upb": "Log Loan UPB",
        "amortization_type": "Amortization Type",
        "interest_type": "Interest Type",
        "original_interest_rate": "Interest Rate",
        "amortization_term": "Amortization Term (Months)",
        "lien_position": "Lien Position",
        "loan_acquisition_ltv": "Acquisition LTV",
        "underwritten_dscr": "Underwritten DSCR",
        "property_acquisition_total_unit_count": "Units at Acquisition",
        "number_of_properties_at_acquisition": "Number of Properties",
        "specific_property_type": "Property Type",
        "physical_occupancy": "Occupancy (%)",
        "note_rate": "Note Rate",
        "property_age": "Property Age (Years)",
    }

    sns.set_theme(style="whitegrid", context="talk")

    # Exclude target and state
    feature_cols = [
        c for c in filtered_data.columns
        if c not in [target_col, "property_state"]
    ]

    # --------------------
    # 4×4 GRID FIGURE
    # --------------------
    n_rows, n_cols = 4, 4
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(22, 18))
    axes = axes.flatten()
    target_col = "loan_ever_60_days_delinquent"

    tmp["target_num"] = (tmp[target_col] == "Y").astype(int)

    for ax, col in zip(axes, feature_cols):

        # Use pretty name if it exists
        col_title = pretty_names.get(col, col)

        # ---------- NUMERIC → STRIPPLOT ----------
        if is_numeric_dtype(filtered_data[col]):
            tmp["target_str"] = tmp[target_col].map({"N": "Non-default", "Y": "Default"})

            sns.stripplot(
                data=tmp,
                x="target_str",
                y=col,
                hue=target_col,
                palette={"N": "#4c72b0", "Y": "#dd8452"},
                dodge=False,
                alpha=0.5,
                ax=ax
            )
            ax.set_title(col_title)
            ax.set_xlabel("")
            ax.set_ylabel(col_title)

            # cleanup legend
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(
                    handles[:2],
                    ["Non-default", "Default"],
                    fontsize=8,
                    loc="upper right"
                )
            else:
                ax.legend().set_visible(False)

        # ---------- CATEGORICAL → DEFAULT RATE BARPLOT ----------
        else:
            stats = (
                tmp
                .groupby(col)["target_num"]
                .agg(default_rate="mean", count="count")
                .reset_index()
            )

            stats = stats.sort_values("default_rate", ascending=False)

            sns.barplot(
                data=stats,
                x=col,
                y="default_rate",
                ax=ax,
                color="#4c72b0"
            )

            ax.set_title(col_title)
            ax.set_xlabel("")
            ax.set_ylabel("Default rate")  # cleaner English
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

            # Add **just the count number**, no prefix
            for i, row in stats.iterrows():
                ax.text(
                    i,
                    row["default_rate"] + 0.005,
                    f"{int(row['count'])}",  # <-- no "n="
                    ha="center",
                    va="bottom",
                    fontsize=8
                )

    # Remove any empty axes (in case < 16 features)
    for i in range(len(feature_cols), len(axes)):
        fig.delaxes(axes[i])

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
    numeric_features = numeric_cols

    categorical_features = {
        "amortization_type": sorted(filtered_data["amortization_type"].dropna().unique()),
        "interest_type": sorted(filtered_data["interest_type"].dropna().unique()),
        "lien_position": sorted(filtered_data["lien_position"].dropna().unique()),
        "specific_property_type": sorted(filtered_data["specific_property_type"].dropna().unique()),
        "property_state": sorted(filtered_data["property_state"].dropna().unique()),
    }
    return render_template("feature_praedium.html", categorical_features = categorical_features,
                           numeric_features=numeric_features, chart_data=session['img_data'])

@app.route("/submit_projection", methods=["POST"])
def submit_feature():
    import pandas as pd
    # getting dummies for categorical columns
    categorical_columns = ['amortization_type', 'interest_type', 'lien_position', 'specific_property_type',
                           'property_state']
    filtered_dummy_data = pd.get_dummies(data=filtered_data, columns=categorical_columns, drop_first=True)


    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(filtered_dummy_data, y, test_size=.2, stratify=y)
    training_columns = list(x_train.columns)
    #instantiating and training the XGBoost model
    from xgboost import XGBClassifier
    from sklearn.metrics import (
        roc_auc_score,
        average_precision_score,
        brier_score_loss,
        classification_report,
        confusion_matrix
    )

    # 2. Class imbalance handling: compute scale_pos_weight
    #    scale_pos_weight ≈ (# negatives) / (# positives)
    pos = y_train.sum()
    neg = len(y_train) - pos
    scale_pos_weight = neg / pos
    print("scale_pos_weight:", scale_pos_weight)

    # 3. Define the XGBoost model
    xgb_model = XGBClassifier(
        n_estimators=500,  # number of trees
        max_depth=4,  # tree depth
        learning_rate=0.05,  # shrinkage
        subsample=0.8,  # row sampling
        colsample_bytree=0.8,  # column sampling
        objective="binary:logistic",  # output = probability
        eval_metric="logloss",  # training metric
        scale_pos_weight=scale_pos_weight,  # handle imbalance
        n_jobs=-1,
        random_state=42
    )

    # 4. Fit the model
    xgb_model.fit(x_train, y_train)

    # 5. Predict probabilities (PD = P(y=1))
    y_proba_xgb = xgb_model.predict_proba(x_test)[:, 1]

    # 6. Evaluate performance
    roc = roc_auc_score(y_test, y_proba_xgb)
    pr_auc = average_precision_score(y_test, y_proba_xgb)
    brier = brier_score_loss(y_test, y_proba_xgb)

    print(f"XGB ROC-AUC:  {roc:.4f}")
    print(f"XGB PR-AUC:   {pr_auc:.4f}")
    print(f"XGB Brier:    {brier:.4f}")

    # 7. Optional: choose a threshold and see classification metrics
    threshold = 0.5  # you can tune this later
    y_pred_xgb = (y_proba_xgb >= threshold).astype(int)

    print("\nClassification report @ threshold =", threshold)
    print(classification_report(y_test, y_pred_xgb))

    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred_xgb))

    # 8. Optional: attach PDs back to a dataframe
    xgb_results = pd.DataFrame(x_test.copy())
    xgb_results["PD_XGB"] = y_proba_xgb
    xgb_results["actual"] = y_test.values

    xgb_results.head()

    import numpy as np
    import pandas as pd
    from pandas.api.types import is_numeric_dtype
    from flask import request, redirect, url_for

    # filtered_data: your original training features (target already removed)
    # training_columns: list of columns used to train the model after get_dummies()

    # 1. Collect raw inputs from form
    raw_inputs = {}

    for col in filtered_data.columns:
        # Get the raw string from the form
        val_str = request.form.get(col)

        # Handle missing input
        if val_str is None or val_str.strip() == "":
            print(f"Missing input for {col}")
            return redirect(url_for("home_praedium"))

        # If this column was numeric in training, parse as float
        if is_numeric_dtype(filtered_data[col].dtype):
            try:
                # allow commas in numbers
                val = float(val_str.replace(",", ""))
            except ValueError:
                print(f"Invalid numeric value for {col}: '{val_str}'")
                return redirect(url_for("home_praedium"))
        else:
            # Categorical: keep as cleaned string
            val = val_str.strip()

        raw_inputs[col] = val

    # 2. Convert to a single-row DataFrame
    input_df = pd.DataFrame([raw_inputs])

    # 3. One-hot encode to match training preprocessing
    input_dummies = pd.get_dummies(input_df, drop_first=False)

    # 4. Align columns to training data (VERY important)
    input_dummies = input_dummies.reindex(columns=training_columns, fill_value=0)

    # 5. Convert to numpy array for the model
    sample_array = input_dummies.to_numpy()



    import numpy as np
    # sample_array: shape (1, n_features), built from your form inputs
    # xgb_model: trained XGBClassifier you fit earlier
    # 1) Predict probability of default (class 1)
    pd_hat = xgb_model.predict_proba(sample_array)[0, 1]

    pd_pct = pd_hat * 100  # convert to percent

    if pd_pct >= 80:
        risk_grade = "C"
    elif pd_pct >= 56:
        risk_grade = "B"
    else:
        risk_grade = "A"

    import matplotlib.pyplot as plt
    import io
    import base64
#making the chart
    def make_pd_chart(pd_hat):
        pd_pct = pd_hat * 100

        # choose color by rating
        if pd_pct >= 80:
            color = "#d62728"  # red for C
        elif pd_pct >= 56:
            color = "#ff7f0e"  # orange for B
        else:
            color = "#2ca02c"  # green for A

        fig, ax = plt.subplots(figsize=(6, 1.5))

        # horizontal bar representing PD
        ax.barh([0], [pd_pct], color=color, height=0.4)
        ax.set_xlim(0, 100)
        ax.set_yticks([])

        # thresholds for A/B/C bands
        ax.axvline(56, color="grey", linestyle="--", linewidth=1)
        ax.axvline(80, color="grey", linestyle="--", linewidth=1)

        # labels for bands
        ax.text(28, 0.45, "A", ha="center", va="bottom", fontsize=9)
        ax.text(68, 0.45, "B", ha="center", va="bottom", fontsize=9)
        ax.text(90, 0.45, "C", ha="center", va="bottom", fontsize=9)

        # main PD label
        ax.set_xlabel("Predicted probability of delinquency (%)")
        ax.set_title(f"PD = {pd_pct:.1f}%")

        plt.tight_layout()

        # convert to base64 string for HTML <img>
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode("utf-8")
        plt.close(fig)

        return img_base64

    chart_data = make_pd_chart(pd_hat)

    return render_template("projection_praedium.html", risk_grade=risk_grade, probability=round(pd_pct,2), chart_data=chart_data)


@app.route('/<path:path>')
def catch_all(path):
    return redirect(url_for("home_praedium"))

if __name__ == "__main__":
    # print(db_get_locales())
    app.secret_key = os.urandom(12)
    app.run(port=5007, debug=True)



