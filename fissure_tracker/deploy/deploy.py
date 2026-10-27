"""
Deploy the crack monitor to AWS with one command (safe to run again after code changes)

    python deploy/deploy.py --check   read-only: checks access, Bedrock, settings and what exists - creates nothing
    python deploy/deploy.py           build and deploy everything, prints the web address

What it creates in us-east-1 (all named "crack-monitor..."):
    Lambda layer     OpenCV 5 + numpy for Linux (~76 MB, uploaded only when it changes)
    Models           ALIKED + LightGlue ONNX files (52 MB) in the bucket; the function copies them to /tmp
    Lambda function  the web app + agent (Python 3.13, 2 GB, 90 s) with a public Function URL
    DynamoDB table   cases, agent traces and readings (pay per request)
    SNS topic        e-mails the engineer review links (the engineer confirms the subscription once)
    IAM role         only its bucket folder, its table, its topic and the Bedrock model
    Log group        14-day retention
    Budget           e-mails ALERT_EMAIL when this month's usage passes $20 (counted before credits)
Photos go into an existing private S3 bucket under the folder crack-monitor/.

Settings: environment variables or fissure_tracker/.env (never commit it):
    ENGINEER_EMAIL   who gets review requests          ALERT_EMAIL   who gets the budget alert
    BUCKET_NAME      existing private bucket (default krishidrishti-images-asron)
    BEDROCK_MODEL_ID default us.amazon.nova-2-lite-v1:0
The AWS account ID is used for ARNs but never printed.
"""

import hashlib
import io
import json
import os
import secrets
import shutil
import subprocess
import sys
import time
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import boto3  # noqa: E402
from botocore.exceptions import ClientError  # noqa: E402

NAME = "crack-monitor"
REGION = "us-east-1"
TABLE, ROLE, TOPIC, LAYER, BUDGET = NAME, f"{NAME}-lambda-role", f"{NAME}-engineer", f"{NAME}-opencv", f"{NAME}-monthly"
BUDGET_USD = "20"
PREFIX = f"{NAME}/"
MODELS_PREFIX = f"{PREFIX}models/"
RUNTIME, PYTHON_VERSION = "python3.13", "3.13"     # numpy 2.4's Linux wheels need Amazon Linux 2023
MEMORY_MB, TIMEOUT_S, LOG_DAYS = 2048, 90, 14
APP_FILES = ["agent.py", "align.py", "app.py", "crack.py", "lambda_function.py", "marker.py", "quality.py", "store.py"]
BUILD = os.path.join(ROOT, "build")
PACKAGE_DIR = os.path.join(BUILD, "package")              # OpenCV + numpy for Linux
LAYER_ZIP = os.path.join(BUILD, "opencv-layer.zip")
GEO_PREFIXES = {"us", "eu", "apac", "global", "us-gov", "au", "jp", "ca"}


def load_env(path=os.path.join(ROOT, ".env")):
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            if "=" in line and not line.strip().startswith("#"):
                key, value = line.strip().split("=", 1)
                os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def setting(name, default=""):
    return os.environ.get(name) or default


def step(text):
    print(f"\n== {text}")


def exists(call, missing_codes, **kwargs):
    try:
        return call(**kwargs)
    except ClientError as e:
        if e.response["Error"]["Code"] in missing_codes:
            return None
        raise


# --- Checks (read-only) ---------------------------------------------------------------------------

def check(session):
    step("Checking access and settings (read-only)")
    ident = session.client("sts").get_caller_identity()
    print(f"signed in as {ident['Arn'].split('/')[-1]} (account ID hidden), region {REGION}")
    for name in ("ENGINEER_EMAIL", "ALERT_EMAIL"):
        print(f"{name}: {setting(name) or '(not set)'}")
    bucket = setting("BUCKET_NAME", "krishidrishti-images-asron")
    ok_bucket = exists(session.client("s3").head_bucket, ("404", "NoSuchBucket", "403"), Bucket=bucket) is not None
    print(f"bucket {bucket}: {'found' if ok_bucket else 'NOT FOUND - set BUCKET_NAME'}")

    model_id = setting("BEDROCK_MODEL_ID", "us.amazon.nova-2-lite-v1:0")
    base_id = model_id.split(".", 1)[1] if model_id.split(".", 1)[0] in GEO_PREFIXES else model_id
    try:
        status = session.client("bedrock").get_foundation_model_availability(modelId=base_id)
        print(f"Bedrock {base_id}: {status.get('authorizationStatus')}")
    except ClientError as e:
        print(f"Bedrock availability check failed: {e.response['Error']['Code']}")
    try:   # one tiny call (a fraction of a cent)
        session.client("bedrock-runtime").converse(
            modelId=model_id, messages=[{"role": "user", "content": [{"text": "Reply with OK"}]}],
            inferenceConfig={"maxTokens": 10})
        print(f"Bedrock {model_id}: answers - the AI agent will be used")
    except ClientError as e:
        print(f"Bedrock {model_id}: {e.response['Error']['Code']} - the rules backup will decide until this works")

    lam = session.client("lambda")
    for label, found in [
        ("Lambda function", exists(lam.get_function, ("ResourceNotFoundException",), FunctionName=NAME)),
        ("DynamoDB table", exists(session.client("dynamodb").describe_table, ("ResourceNotFoundException",), TableName=TABLE)),
        ("IAM role", exists(session.client("iam").get_role, ("NoSuchEntity",), RoleName=ROLE)),
    ]:
        print(f"{label}: {'exists (will be updated)' if found else 'will be created'}")
    print(f"OpenCV package for Linux: {'ready' if os.path.isdir(os.path.join(PACKAGE_DIR, 'cv2')) else 'will be downloaded (~76 MB)'}")
    return ident["Account"], bucket, model_id, ok_bucket


# --- Build ----------------------------------------------------------------------------------------

def build_layer():
    step("OpenCV layer package")
    if not os.path.isdir(os.path.join(PACKAGE_DIR, "cv2")):
        print("downloading Linux versions of OpenCV + numpy (~76 MB, first time only)...")
        shutil.rmtree(PACKAGE_DIR, ignore_errors=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "--disable-pip-version-check",
                        "--platform", "manylinux_2_34_x86_64", "--platform", "manylinux_2_28_x86_64",
                        "--platform", "manylinux2014_x86_64", "--implementation", "cp", "--python-version",
                        PYTHON_VERSION, "--only-binary=:all:", "--target", PACKAGE_DIR,
                        "-r", os.path.join(ROOT, "requirements.txt")], check=True)
        shutil.rmtree(os.path.join(PACKAGE_DIR, "bin"), ignore_errors=True)
    if not os.path.exists(LAYER_ZIP):
        with zipfile.ZipFile(LAYER_ZIP, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
            for folder, dirs, files in os.walk(PACKAGE_DIR):
                dirs[:] = [d for d in dirs if d not in ("tests", "__pycache__")]
                for name in files:
                    path = os.path.join(folder, name)
                    z.write(path, os.path.join("python", os.path.relpath(path, PACKAGE_DIR)))   # layers need python/
    digest = hashlib.sha256(open(LAYER_ZIP, "rb").read()).hexdigest()[:16]
    print(f"{os.path.getsize(LAYER_ZIP) / 1e6:.0f} MB, fingerprint {digest}")
    return digest


def app_zip():
    data = io.BytesIO()
    with zipfile.ZipFile(data, "w", zipfile.ZIP_DEFLATED) as z:
        for name in APP_FILES:
            z.write(os.path.join(ROOT, name), name)
    return data.getvalue()


# --- AWS resources --------------------------------------------------------------------------------

def ensure_layer(session, bucket, digest):
    step(f"Lambda layer {LAYER}")
    lam = session.client("lambda")
    versions = lam.list_layer_versions(LayerName=LAYER).get("LayerVersions", [])
    for v in versions:
        if v.get("Description", "").endswith(digest):
            print(f"unchanged (version {v['Version']})")
            return v["LayerVersionArn"]
    key = f"{PREFIX}deploy/opencv-layer.zip"
    print("uploading ~76 MB (one time)...")
    session.client("s3").upload_file(LAYER_ZIP, bucket, key)
    v = lam.publish_layer_version(LayerName=LAYER, Description=f"OpenCV 5.0.0 + numpy 2.4.6, {digest}",
                                  Content={"S3Bucket": bucket, "S3Key": key},
                                  CompatibleRuntimes=[RUNTIME], CompatibleArchitectures=["x86_64"])
    print(f"published version {v['Version']}")
    return v["LayerVersionArn"]


def ensure_models(session, bucket):
    from align import ALIKED_MODEL, LIGHTGLUE_MODEL, MODEL_DIR

    step("ALIKED + LightGlue models (OpenCV 5)")
    paths = [os.path.join(MODEL_DIR, name) for name in (ALIKED_MODEL, LIGHTGLUE_MODEL)]
    if not all(os.path.exists(p) for p in paths):
        print("not downloaded - alignment will use ORB only (run tools/get_models.py to add them)")
        return ""
    s3 = session.client("s3")
    for path in paths:
        key = MODELS_PREFIX + os.path.basename(path)
        head = exists(s3.head_object, ("404", "NoSuchKey"), Bucket=bucket, Key=key)
        if head is None or head["ContentLength"] != os.path.getsize(path):
            print(f"uploading {os.path.basename(path)} ({os.path.getsize(path) / 1e6:.0f} MB)...")
            s3.upload_file(path, bucket, key)
    print("in the bucket")
    return MODELS_PREFIX


def ensure_table(dynamodb):
    step(f"DynamoDB table {TABLE}")
    if exists(dynamodb.describe_table, ("ResourceNotFoundException",), TableName=TABLE) is None:
        dynamodb.create_table(
            TableName=TABLE, BillingMode="PAY_PER_REQUEST",
            AttributeDefinitions=[{"AttributeName": "pk", "AttributeType": "S"},
                                  {"AttributeName": "sk", "AttributeType": "S"}],
            KeySchema=[{"AttributeName": "pk", "KeyType": "HASH"}, {"AttributeName": "sk", "KeyType": "RANGE"}])
        dynamodb.get_waiter("table_exists").wait(TableName=TABLE)
        print("created (pay per request)")
    else:
        print("exists")
    return dynamodb.describe_table(TableName=TABLE)["Table"]["TableArn"]


def ensure_topic(sns, email):
    step(f"SNS topic {TOPIC} (engineer e-mails)")
    arn = sns.create_topic(Name=TOPIC)["TopicArn"]          # returns the existing one if it is already there
    subs = sns.list_subscriptions_by_topic(TopicArn=arn)["Subscriptions"]
    if email and not any(s["Endpoint"].lower() == email.lower() for s in subs):
        sns.subscribe(TopicArn=arn, Protocol="email", Endpoint=email)
        print(f"{email} subscribed - they must click 'Confirm subscription' in the e-mail from AWS")
    elif email:
        pending = any(s["Endpoint"].lower() == email.lower() and s["SubscriptionArn"] == "PendingConfirmation" for s in subs)
        print(f"{email}: {'still waiting for the confirmation click' if pending else 'subscribed'}")
    else:
        print("no ENGINEER_EMAIL set - review requests will not be e-mailed")
    return arn


def model_resources(model_id, account):
    prefix, _, rest = model_id.partition(".")
    if prefix in GEO_PREFIXES:   # an inference profile can route to the model in several regions
        return [f"arn:aws:bedrock:{REGION}:{account}:inference-profile/{model_id}", f"arn:aws:bedrock:*::foundation-model/{rest}"]
    return [f"arn:aws:bedrock:{REGION}::foundation-model/{model_id}"]


def ensure_role(iam, bucket, table_arn, topic_arn, model_id, account):
    step(f"IAM role {ROLE}")
    role = exists(iam.get_role, ("NoSuchEntity",), RoleName=ROLE)
    if role is None:
        trust = {"Version": "2012-10-17", "Statement": [{
            "Effect": "Allow", "Principal": {"Service": "lambda.amazonaws.com"}, "Action": "sts:AssumeRole"}]}
        role = iam.create_role(RoleName=ROLE, AssumeRolePolicyDocument=json.dumps(trust),
                               Description="Crack monitor web app + agent (Lambda)")
        iam.attach_role_policy(RoleName=ROLE, PolicyArn="arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole")
        print("created")
    statements = [
        {"Effect": "Allow", "Action": ["s3:GetObject", "s3:PutObject"], "Resource": f"arn:aws:s3:::{bucket}/{PREFIX}photos/*"},
        {"Effect": "Allow", "Action": "s3:GetObject", "Resource": f"arn:aws:s3:::{bucket}/{MODELS_PREFIX}*"},
        {"Effect": "Allow", "Action": ["dynamodb:GetItem", "dynamodb:PutItem", "dynamodb:Query"], "Resource": table_arn},
        {"Effect": "Allow", "Action": "sns:Publish", "Resource": topic_arn},
        {"Effect": "Allow", "Action": "bedrock:InvokeModel", "Resource": model_resources(model_id, account)},
    ]
    iam.put_role_policy(RoleName=ROLE, PolicyName=f"{NAME}-access",
                        PolicyDocument=json.dumps({"Version": "2012-10-17", "Statement": statements}))
    print("permissions: its photo folder, its models (read), its table, its topic, the Bedrock model - nothing else")
    return role["Role"]["Arn"]


def ensure_log_group(logs):
    name = f"/aws/lambda/{NAME}"
    try:
        logs.create_log_group(logGroupName=name)
    except ClientError as e:
        if e.response["Error"]["Code"] != "ResourceAlreadyExistsException":
            raise
    logs.put_retention_policy(logGroupName=name, retentionInDays=LOG_DAYS)


def ensure_function(lam, role_arn, layer_arn, settings):
    step(f"Lambda function {NAME}")
    config = {"Runtime": RUNTIME, "Role": role_arn, "Handler": "lambda_function.lambda_handler", "Layers": [layer_arn],
              "Timeout": TIMEOUT_S, "MemorySize": MEMORY_MB, "Environment": {"Variables": settings},
              "Description": "Crack monitor: OpenCV 5 measurements + Bedrock agent"}
    code = app_zip()
    if exists(lam.get_function, ("ResourceNotFoundException",), FunctionName=NAME) is None:
        for attempt in range(12):     # a brand-new IAM role takes a few seconds to become usable
            try:
                lam.create_function(FunctionName=NAME, Code={"ZipFile": code}, Architectures=["x86_64"], **config)
                break
            except ClientError as e:
                if e.response["Error"]["Code"] != "InvalidParameterValueException" or attempt == 11:
                    raise
                time.sleep(5)
        lam.get_waiter("function_active_v2").wait(FunctionName=NAME)
        print("created")
    else:
        lam.update_function_code(FunctionName=NAME, ZipFile=code)
        lam.get_waiter("function_updated_v2").wait(FunctionName=NAME)
        lam.update_function_configuration(FunctionName=NAME, **config)
        lam.get_waiter("function_updated_v2").wait(FunctionName=NAME)
        print("updated")


def ensure_url(lam):
    step("Public web address (Function URL)")
    url = exists(lam.get_function_url_config, ("ResourceNotFoundException",), FunctionName=NAME)
    if url is None:
        url = lam.create_function_url_config(FunctionName=NAME, AuthType="NONE")
    # A public Function URL needs BOTH permissions; the app itself checks the access code
    for sid, extra in (("FunctionURLAllowPublicAccess", {"Action": "lambda:InvokeFunctionUrl", "FunctionUrlAuthType": "NONE"}),
                       ("FunctionURLInvokeAllowPublicAccess", {"Action": "lambda:InvokeFunction", "InvokedViaFunctionUrl": True})):
        try:
            lam.add_permission(FunctionName=NAME, StatementId=sid, Principal="*", **extra)
        except ClientError as e:
            if e.response["Error"]["Code"] != "ResourceConflictException":
                raise
    return url["FunctionUrl"]


def ensure_budget(budgets, account, email):
    step(f"Spending alert (${BUDGET_USD}/month)")
    if not email:
        print("skipped - set ALERT_EMAIL")
        return
    try:
        budgets.create_budget(
            AccountId=account,
            Budget={"BudgetName": BUDGET, "BudgetType": "COST", "TimeUnit": "MONTHLY",
                    "BudgetLimit": {"Amount": BUDGET_USD, "Unit": "USD"},
                    "CostTypes": {"IncludeCredit": False, "IncludeRefund": False}},   # count usage before credits
            NotificationsWithSubscribers=[
                {"Notification": {"NotificationType": kind, "ComparisonOperator": "GREATER_THAN",
                                  "Threshold": threshold, "ThresholdType": "PERCENTAGE"},
                 "Subscribers": [{"SubscriptionType": "EMAIL", "Address": email}]}
                for kind, threshold in (("ACTUAL", 50.0), ("ACTUAL", 100.0), ("FORECASTED", 100.0))])
        print(f"created - alerts go to {email}")
    except ClientError as e:
        code = e.response["Error"]["Code"]
        print("already exists" if code == "DuplicateRecordException" else f"could not create ({code}) - use Billing > Budgets")


def main():
    load_env()
    session = boto3.Session(region_name=REGION)
    account, bucket, model_id, ok_bucket = check(session)
    if "--check" in sys.argv:
        print("\nCheck finished - nothing was created or changed.")
        return
    if not ok_bucket:
        raise SystemExit("Bucket not found - set BUCKET_NAME to an existing private bucket.")

    lam = session.client("lambda")
    old = (exists(lam.get_function_configuration, ("ResourceNotFoundException",), FunctionName=NAME) or {}) \
        .get("Environment", {}).get("Variables", {})
    digest = build_layer()
    layer_arn = ensure_layer(session, bucket, digest)
    models_prefix = ensure_models(session, bucket)
    table_arn = ensure_table(session.client("dynamodb"))
    topic_arn = ensure_topic(session.client("sns"), setting("ENGINEER_EMAIL"))
    role_arn = ensure_role(session.client("iam"), bucket, table_arn, topic_arn, model_id, account)
    ensure_log_group(session.client("logs"))
    settings = {"TABLE_NAME": TABLE, "BUCKET_NAME": bucket, "BUCKET_PREFIX": PREFIX, "TOPIC_ARN": topic_arn,
                "BEDROCK_MODEL_ID": model_id, "AGENT_MODE": setting("AGENT_MODE", "auto"), "MODELS_PREFIX": models_prefix,
                "ACCESS_CODE": old.get("ACCESS_CODE") or secrets.token_urlsafe(9)}   # keep the code across deploys
    ensure_function(lam, role_arn, layer_arn, settings)
    url = ensure_url(lam)
    ensure_budget(session.client("budgets", region_name="us-east-1"), account, setting("ALERT_EMAIL"))
    print(f"\nDone. Open on a phone:\n  {url}?code={settings['ACCESS_CODE']}\n"
          "Share this link only with people who should upload photos (e.g. the judges).")


if __name__ == "__main__":
    main()
