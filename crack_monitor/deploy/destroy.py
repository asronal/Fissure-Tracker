"""
Delete everything deploy.py created (run after judging). Asks before deleting.

    python deploy/destroy.py

Deletes: the Lambda function + its URL, all layer versions, the DynamoDB table (all cases and readings),
the SNS topic, the IAM role, the log group, the budget, and the photos under crack-monitor/ in the bucket.
The bucket itself is kept (it existed before).
"""

import os
import sys

import boto3
from botocore.exceptions import ClientError

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from deploy import BUDGET, LAYER, NAME, PREFIX, REGION, ROLE, TABLE, TOPIC, load_env, setting  # noqa: E402


def attempt(label, fn, **kwargs):
    try:
        fn(**kwargs)
        print(f"deleted  {label}")
    except ClientError as e:
        print(f"skipped  {label} ({e.response['Error']['Code']})")


def main():
    load_env()
    answer = input(f"This deletes the crack monitor and ALL its data in AWS ({REGION}). Type 'delete' to go on: ")
    if answer.strip() != "delete":
        print("Nothing deleted.")
        return
    s = boto3.Session(region_name=REGION)
    account = s.client("sts").get_caller_identity()["Account"]
    lam, iam, sns = s.client("lambda"), s.client("iam"), s.client("sns")

    attempt("function URL", lam.delete_function_url_config, FunctionName=NAME)
    attempt("Lambda function", lam.delete_function, FunctionName=NAME)
    for v in lam.list_layer_versions(LayerName=LAYER).get("LayerVersions", []):
        attempt(f"layer version {v['Version']}", lam.delete_layer_version, LayerName=LAYER, VersionNumber=v["Version"])
    attempt("DynamoDB table", s.client("dynamodb").delete_table, TableName=TABLE)
    for t in sns.list_topics()["Topics"]:
        if t["TopicArn"].endswith(f":{TOPIC}"):
            attempt("SNS topic", sns.delete_topic, TopicArn=t["TopicArn"])
    attempt("role policy", iam.delete_role_policy, RoleName=ROLE, PolicyName=f"{NAME}-access")
    attempt("role managed policy", iam.detach_role_policy, RoleName=ROLE,
            PolicyArn="arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole")
    attempt("IAM role", iam.delete_role, RoleName=ROLE)
    attempt("log group", s.client("logs").delete_log_group, logGroupName=f"/aws/lambda/{NAME}")
    attempt("budget", s.client("budgets", region_name="us-east-1").delete_budget, AccountId=account, BudgetName=BUDGET)

    bucket = setting("BUCKET_NAME", "krishidrishti-images-asron")
    s3 = s.resource("s3").Bucket(bucket)
    count = sum(len(batch["Deleted"]) for batch in s3.objects.filter(Prefix=PREFIX).delete() if "Deleted" in batch)
    print(f"deleted  {count} files under {bucket}/{PREFIX}")


if __name__ == "__main__":
    main()
