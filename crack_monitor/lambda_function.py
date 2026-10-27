"""
AWS Lambda entry point (behind a public Lambda Function URL)

Settings come from the function's environment variables (set by deploy/deploy.py):
    TABLE_NAME, BUCKET_NAME, BUCKET_PREFIX   where cases, readings and photos are kept
    TOPIC_ARN                                Amazon SNS topic that e-mails the engineer
    ACCESS_CODE                              needed to upload photos and see all cracks
    BEDROCK_MODEL_ID, AGENT_MODE             the AI model; AGENT_MODE=rules switches it off
    MODELS_PREFIX                            S3 folder with the ALIKED + LightGlue ONNX models (optional)
"""

import os

import boto3

# The OpenCV 5 ALIKED + LightGlue models (52 MB) don't fit in the Lambda layer next to OpenCV, so they
# are copied from S3 to /tmp once per container, before align.py looks for them
MODEL_DIR = os.environ.setdefault("MODEL_DIR", "/tmp/models")


def fetch_models():
    from align import ALIKED_MODEL, LIGHTGLUE_MODEL

    prefix = os.environ.get("MODELS_PREFIX")
    if not prefix:
        return
    os.makedirs(MODEL_DIR, exist_ok=True)
    s3 = boto3.client("s3")
    try:
        for name in (ALIKED_MODEL, LIGHTGLUE_MODEL):
            if not os.path.exists(os.path.join(MODEL_DIR, name)):
                s3.download_file(os.environ["BUCKET_NAME"], prefix + name, os.path.join(MODEL_DIR, name) + ".part")
                os.replace(os.path.join(MODEL_DIR, name) + ".part", os.path.join(MODEL_DIR, name))
    except Exception as e:      # alignment falls back to ORB only
        print(f"ALIKED + LightGlue models not loaded: {e}")


fetch_models()

from agent import make_llm  # noqa: E402
from app import App  # noqa: E402
from store import AwsStore  # noqa: E402


class SnsNotifier:
    def __init__(self, topic_arn):
        self.topic_arn = topic_arn
        self.sns = boto3.client("sns")

    def notify_engineer(self, subject, body):
        self.sns.publish(TopicArn=self.topic_arn, Subject=subject[:100], Message=body)


# Created once per Lambda container and reused for the next requests (faster)
app = App(AwsStore(os.environ["TABLE_NAME"], os.environ["BUCKET_NAME"], os.environ.get("BUCKET_PREFIX", "crack-monitor/")),
          SnsNotifier(os.environ["TOPIC_ARN"]), make_llm, os.environ.get("ACCESS_CODE", ""),
          os.environ.get("BEDROCK_MODEL_ID"))


def lambda_handler(event, context):
    return app.handle(event)
