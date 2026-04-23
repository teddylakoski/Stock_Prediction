"""
Streamlit web app for my Loan Default Prediction project.

Same structure as the other Streamlit apps we've built: pulls model + explainer
from S3, calls the SageMaker endpoint, and shows a SHAP waterfall for the
prediction.

Secrets needed in .streamlit/secrets.toml:
    [aws_credentials]
    AWS_ACCESS_KEY_ID     = "..."
    AWS_SECRET_ACCESS_KEY = "..."
    AWS_SESSION_TOKEN     = "..."
    AWS_BUCKET            = "teddy-lakoski-s3-bucket"
    AWS_ENDPOINT          = "loan-default-endpoint-v1"

Run with:
    streamlit run streamlit_app.py
"""

import os, sys, warnings
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import posixpath
import joblib
import tarfile
import tempfile
import boto3
import sagemaker
from sagemaker.predictor import Predictor
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import JSONDeserializer
from sklearn.pipeline import Pipeline
import shap

# -------------------------------------------------------------------
# Setup
# -------------------------------------------------------------------
warnings.simplefilter("ignore")

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# Access secrets
aws_id       = st.secrets["aws_credentials"]["AWS_ACCESS_KEY_ID"]
aws_secret   = st.secrets["aws_credentials"]["AWS_SECRET_ACCESS_KEY"]
aws_token    = st.secrets["aws_credentials"]["AWS_SESSION_TOKEN"]
aws_bucket   = st.secrets["aws_credentials"]["AWS_BUCKET"]
aws_endpoint = st.secrets["aws_credentials"]["AWS_ENDPOINT"]


# -------------------------------------------------------------------
# AWS Session Management
# -------------------------------------------------------------------
@st.cache_resource
def get_session(aws_id, aws_secret, aws_token):
    return boto3.Session(
        aws_access_key_id=aws_id,
        aws_secret_access_key=aws_secret,
        aws_session_token=aws_token,
        region_name='us-east-1'
    )

session    = get_session(aws_id, aws_secret, aws_token)
sm_session = sagemaker.Session(boto_session=session)


# -------------------------------------------------------------------
# Model & feature configuration
# -------------------------------------------------------------------
MODEL_INFO = {
    "endpoint":  aws_endpoint,
    "explainer": "explainer_loan_default.shap",
    "pipeline":  "finalized_loan_default_pipeline.tar.gz",
    "s3_prefix": "sklearn-pipeline-deployment",
    # Raw feature names the pipeline expects (the pipeline does its own
    # feature engineering + one-hot encoding internally)
    "keys": [
        "loan_amnt", "term", "int_rate", "installment", "annual_inc",
        "dti", "emp_length", "fico_range_low", "fico_range_high",
        "grade", "home_ownership", "purpose",
    ],
    # UI config for each input
    "inputs": [
        {"name": "loan_amnt",       "label": "Loan amount ($)",       "type": "number",   "min": 500,   "max": 40000,  "default": 10000, "step": 500},
        {"name": "term",            "label": "Term (months)",         "type": "select",   "options": [36, 60],          "default": 36},
        {"name": "int_rate",        "label": "Interest rate (%)",     "type": "number",   "min": 5.0,   "max": 30.0,   "default": 12.5,  "step": 0.25},
        {"name": "installment",     "label": "Monthly installment",   "type": "number",   "min": 20.0,  "max": 2000.0, "default": 300.0, "step": 10.0},
        {"name": "annual_inc",      "label": "Annual income ($)",     "type": "number",   "min": 10000, "max": 500000, "default": 60000, "step": 1000},
        {"name": "dti",             "label": "Debt-to-income (%)",    "type": "number",   "min": 0.0,   "max": 50.0,   "default": 18.0,  "step": 0.5},
        {"name": "emp_length",      "label": "Employment (years)",    "type": "number",   "min": 0,     "max": 10,     "default": 5,     "step": 1},
        {"name": "fico_range_low",  "label": "FICO (low)",            "type": "number",   "min": 600,   "max": 850,    "default": 700,   "step": 1},
        {"name": "fico_range_high", "label": "FICO (high)",           "type": "number",   "min": 600,   "max": 850,    "default": 705,   "step": 1},
        {"name": "grade",           "label": "Grade",                 "type": "select",   "options": list("ABCDEFG"),   "default": "B"},
        {"name": "home_ownership",  "label": "Home ownership",        "type": "select",   "options": ["RENT","MORTGAGE","OWN","OTHER"], "default": "MORTGAGE"},
        {"name": "purpose",         "label": "Purpose",               "type": "select",
         "options": ["debt_consolidation","credit_card","home_improvement","major_purchase",
                     "medical","small_business","car","vacation","moving","other"],
         "default": "debt_consolidation"},
    ],
}


# -------------------------------------------------------------------
# Download the pipeline from S3 (only used for local fallback / debugging)
# -------------------------------------------------------------------
def load_pipeline(_session, bucket, key):
    s3_client = _session.client('s3')
    filename = MODEL_INFO["pipeline"]
    s3_client.download_file(
        Filename=filename,
        Bucket=bucket,
        Key=f"{key}/{os.path.basename(filename)}",
    )
    # Extract the .joblib file from the .tar.gz
    with tarfile.open(filename, "r:gz") as tar:
        tar.extractall(path=".")
        joblib_file = [f for f in tar.getnames() if f.endswith('.joblib')][0]
    return joblib.load(joblib_file)


# -------------------------------------------------------------------
# Download the SHAP explainer from S3
# -------------------------------------------------------------------
def load_shap_explainer(_session, bucket, key, local_path):
    s3_client = _session.client('s3')
    if not os.path.exists(local_path):
        s3_client.download_file(Filename=local_path, Bucket=bucket, Key=key)
    with open(local_path, "rb") as f:
        return shap.Explainer.load(f)


# -------------------------------------------------------------------
# Call the live SageMaker endpoint
# -------------------------------------------------------------------
def call_model_api(input_df):
    predictor = Predictor(
        endpoint_name=MODEL_INFO["endpoint"],
        sagemaker_session=sm_session,
        serializer=JSONSerializer(),
        deserializer=JSONDeserializer(),
    )
    try:
        payload = input_df.iloc[0].to_dict()
        result  = predictor.predict(payload)
        proba   = float(result.get("default_probability", 0.0))
        pred    = int(result.get("prediction", 0))
        return {"proba": round(proba, 4), "pred": pred}, 200
    except Exception as e:
        return f"Error: {str(e)}", 500


# -------------------------------------------------------------------
# Local SHAP explanation (uses local pipeline preprocessing + downloaded explainer)
# -------------------------------------------------------------------
def display_explanation(input_df, _session, bucket):
    explainer_name = MODEL_INFO["explainer"]
    explainer = load_shap_explainer(
        _session, bucket,
        posixpath.join("explainer", explainer_name),
        os.path.join(tempfile.gettempdir(), explainer_name),
    )

    # Apply the pipeline's preprocessing steps locally so we get the feature
    # matrix the explainer expects (skip the SMOTE resampler and the classifier)
    try:
        pipeline = load_pipeline(_session, bucket, MODEL_INFO["s3_prefix"])
        pre_steps = [(n, s) for n, s in pipeline.steps
                     if n not in ("classifier", "resampler")]
        X_pre = Pipeline(pre_steps).transform(input_df)
    except Exception as e:
        st.warning(f"Could not run local preprocessing for SHAP: {e}")
        return

    shap_values = explainer(X_pre)

    st.subheader("Decision Transparency (SHAP)")
    fig, ax = plt.subplots(figsize=(10, 4))
    shap.plots.waterfall(shap_values[0], max_display=10, show=False)
    st.pyplot(fig)

    top_feature = shap_values[0].feature_names[0]
    st.info(f"**Business Insight:** The most influential factor in this decision was **{top_feature}**.")


# -------------------------------------------------------------------
# Streamlit UI
# -------------------------------------------------------------------
st.set_page_config(page_title="Loan Default Predictor", layout="wide")
st.title("Loan Default Predictor")
st.caption("Teddy Lakoski — Final ML Project")

with st.form("pred_form"):
    st.subheader("Applicant inputs")
    cols = st.columns(2)
    user_inputs = {}

    for i, inp in enumerate(MODEL_INFO["inputs"]):
        with cols[i % 2]:
            if inp["type"] == "select":
                user_inputs[inp["name"]] = st.selectbox(
                    inp["label"], inp["options"],
                    index=inp["options"].index(inp["default"]),
                )
            else:
                user_inputs[inp["name"]] = st.number_input(
                    inp["label"],
                    min_value=inp["min"], max_value=inp["max"],
                    value=inp["default"], step=inp["step"],
                )

    submitted = st.form_submit_button("Run Prediction")

if submitted:
    input_df = pd.DataFrame([user_inputs], columns=MODEL_INFO["keys"])

    res, status = call_model_api(input_df)
    if status == 200:
        c1, c2 = st.columns(2)
        c1.metric("Default probability", f"{res['proba']:.1%}")
        c2.metric("Prediction", "Charge-off" if res["pred"] else "Fully paid")

        if res["pred"] == 1:
            st.error("The model thinks this applicant is high-risk.")
        else:
            st.success("The model expects this applicant to pay back the loan.")

        display_explanation(input_df, session, aws_bucket)
    else:
        st.error(res)
