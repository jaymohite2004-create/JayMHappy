
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, label_binarize
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)

st.set_page_config(page_title="ReBrew Subscription Interest Dashboard", layout="wide")

DATA_PATH = "ReBrew_Market_Survey_Synthetic_Data_600_responses.xlsx"
TARGET_COL = "Q33_Subscription_Interest"
DROP_COLS = ["Response_ID", "Timestamp"]


@st.cache_data
def load_data(path: str):
    df = pd.read_excel(path)
    return df


def build_models():
    return {
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=None, random_state=42
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=300, learning_rate=0.05, random_state=42
        ),
    }


def train_and_evaluate(df, selected_models):
    # Basic cleaning
    df = df.copy()
    df = df.dropna(subset=[TARGET_COL])
    X = df.drop(columns=DROP_COLS + [TARGET_COL])
    y = df[TARGET_COL]

    categorical_features = X.columns.tolist()

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocess = ColumnTransformer(
        transformers=[("cat", categorical_transformer, categorical_features)]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = build_models()

    metrics_records = []
    roc_info = {}
    pipelines = {}
    classes = np.unique(y_train)
    y_test_bin = label_binarize(y_test, classes=classes)

    for name in selected_models:
        clf = models[name]
        pipe = Pipeline(steps=[("preprocess", preprocess), ("model", clf)])
        pipe.fit(X_train, y_train)

        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
        rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

        auc = roc_auc_score(
            label_binarize(y_test, classes=classes),
            y_proba,
            multi_class="ovr",
            average="macro",
        )

        # Macro-average ROC
        fpr_dict = {}
        tpr_dict = {}
        for i, cls in enumerate(classes):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
            fpr_dict[cls] = fpr
            tpr_dict[cls] = tpr

        all_fpr = np.unique(
            np.concatenate([fpr_dict[cls] for cls in classes])
        )
        mean_tpr = np.zeros_like(all_fpr)
        for cls in classes:
            mean_tpr += np.interp(all_fpr, fpr_dict[cls], tpr_dict[cls])
        mean_tpr /= len(classes)

        roc_info[name] = {
            "fpr": all_fpr,
            "tpr": mean_tpr,
            "auc": auc,
        }

        metrics_records.append(
            {
                "Algorithm": name,
                "Accuracy": acc,
                "Precision (macro)": prec,
                "Recall (macro)": rec,
                "F1-score (macro)": f1,
                "ROC AUC (macro)": auc,
            }
        )

        pipelines[name] = pipe

    metrics_df = pd.DataFrame(metrics_records).set_index("Algorithm")
    return metrics_df, roc_info, pipelines, X, y, classes


def show_marketing_insights(df):
    st.subheader("1. Top Marketing Insights")

    c1, c2 = st.columns(2)

    with c1:
        fig1 = px.histogram(
            df,
            x=TARGET_COL,
            title="Subscription Interest Distribution",
            color=TARGET_COL,
        )
        st.plotly_chart(fig1, use_container_width=True)

    with c2:
        fig2 = px.histogram(
            df,
            x="Q1_Age_Group",
            color=TARGET_COL,
            barmode="group",
            title="Subscription Interest by Age Group",
        )
        st.plotly_chart(fig2, use_container_width=True)

    c3, c4 = st.columns(2)

    with c3:
        fig3 = px.histogram(
            df,
            x="Q8_Coffee_Frequency",
            color=TARGET_COL,
            barmode="group",
            title="Subscription Interest vs Coffee Consumption Frequency",
        )
        st.plotly_chart(fig3, use_container_width=True)

    with c4:
        fig4 = px.histogram(
            df,
            x="Q20_Specific_Products",
            color=TARGET_COL,
            title="Interest by Specific Product Preferences",
        )
        st.plotly_chart(fig4, use_container_width=True)

    fig5 = px.histogram(
        df,
        x="Q18_Shopping_Channels",
        color=TARGET_COL,
        title="Preferred Shopping Channels vs Subscription Interest",
    )
    st.plotly_chart(fig5, use_container_width=True)


def build_prediction_form(df, pipelines, default_model_name):
    st.subheader("3. Predict New Customer's Subscription Interest")

    if not pipelines:
        st.info("Train at least one model in section 2 to enable predictions.")
        return

    model_name = st.selectbox(
        "Select model for prediction", list(pipelines.keys()), index=0
    )
    model = pipelines[model_name]

    X_all = df.drop(columns=DROP_COLS + [TARGET_COL])
    cols = X_all.columns.tolist()

    st.markdown("#### Enter new customer information")

    with st.form(key="prediction_form"):
        input_data = {}
        for col in cols:
            unique_vals = sorted(
                [str(v) for v in df[col].dropna().unique().tolist()]
            )
            if len(unique_vals) > 0 and len(unique_vals) <= 25:
                default_val = unique_vals[0]
                input_data[col] = st.selectbox(
                    col, options=unique_vals, index=0
                )
            else:
                input_data[col] = st.text_input(col, value=str(df[col].dropna().iloc[0]) if df[col].dropna().shape[0] > 0 else "")

        submitted = st.form_submit_button("Predict")

    if submitted:
        new_row = pd.DataFrame([input_data])
        pred_label = model.predict(new_row)[0]
        st.success(f"Predicted Subscription Interest: **{pred_label}**")

        # Apply model to full dataset for download
        full_X = df.drop(columns=DROP_COLS + [TARGET_COL])
        full_pred = model.predict(full_X)
        df_with_pred = df.copy()
        df_with_pred[f"Predicted_{TARGET_COL}_{model_name.replace(' ', '_')}"] = full_pred

        csv_bytes = df_with_pred.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download dataset with predicted labels as CSV",
            data=csv_bytes,
            file_name="rebew_subscription_predictions.csv",
            mime="text/csv",
        )


def main():
    st.title("ReBrew Subscription Interest – Marketing & AI Dashboard")
    st.write(
        "This dashboard helps ReBrew understand which customers are likely to subscribe to coffee ground deliveries and why."
    )

    df = load_data(DATA_PATH)

    show_marketing_insights(df)

    st.subheader("2. Model Training & Evaluation")
    st.write(
        "Select one or more algorithms and compare their performance on predicting subscription interest."
    )

    all_models = list(build_models().keys())
    selected_models = st.multiselect(
        "Choose algorithms to train",
        all_models,
        default=all_models,
    )

    if selected_models:
        metrics_df, roc_info, pipelines, X, y, classes = train_and_evaluate(
            df, selected_models
        )

        st.markdown("#### Model Performance Metrics (on test set)")
        st.dataframe(metrics_df.style.format("{:.3f}"))

        # ROC curve
        st.markdown("#### ROC Curve Comparison (Macro-average)")
        fig, ax = plt.subplots(figsize=(6, 4))
        colors = {
            "Decision Tree": "tab:blue",
            "Random Forest": "tab:green",
            "Gradient Boosting": "tab:red",
        }

        for name, info in roc_info.items():
            ax.plot(
                info["fpr"],
                info["tpr"],
                label=f"{name} (AUC={info['auc']:.3f})",
                color=colors.get(name, None),
            )

        ax.plot([0, 1], [0, 1], "k--", label="No Skill")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve – Subscription Interest Models")
        ax.legend()
        st.pyplot(fig)

        build_prediction_form(df, pipelines, selected_models[0])
    else:
        st.info("Select at least one algorithm to train and evaluate models.")

if __name__ == "__main__":
    main()
