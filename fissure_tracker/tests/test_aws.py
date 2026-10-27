"""AWS pieces checked offline: AwsStore against fake DynamoDB/S3 objects, and the Lambda entry point."""

import importlib
import json
import os

import pytest

from agent import LocalNotifier, handle_photo
from store import AwsStore
from test_agent import shot


class FakeTable:
    def __init__(self):
        self.items = {}

    def put_item(self, Item):
        self.items[(Item["pk"], Item["sk"])] = Item

    def get_item(self, Key):
        item = self.items.get((Key["pk"], Key["sk"]))
        return {"Item": item} if item else {}

    def query(self, KeyConditionExpression, ScanIndexForward=True, Limit=None, **_):
        pk = KeyConditionExpression.get_expression()["values"][1]
        rows = sorted((v for (p, _), v in self.items.items() if p == pk), key=lambda i: i["sk"],
                      reverse=not ScanIndexForward)
        return {"Items": rows[:Limit] if Limit else rows}


class FakeS3:
    class exceptions:
        class NoSuchKey(Exception):
            pass

    def __init__(self):
        self.objects = {}

    def put_object(self, Bucket, Key, Body, ContentType):
        self.objects[Key] = Body

    def get_object(self, Bucket, Key):
        if Key not in self.objects:
            raise self.exceptions.NoSuchKey()

        class Body:
            def read(inner):
                return self.objects[Key]
        return {"Body": Body()}

    def generate_presigned_url(self, op, ExpiresIn, Params):
        return f"https://s3.example/{Params['Key']}?expires={ExpiresIn}"


@pytest.fixture
def aws_store():
    store = AwsStore("table", "bucket", region="us-east-1")       # creating boto3 objects makes no AWS calls
    store.table, store.s3 = FakeTable(), FakeS3()
    return store


def test_agent_works_on_the_aws_store(aws_store):
    first = handle_photo(aws_store, shot(), notifier=LocalNotifier())
    second = handle_photo(aws_store, shot(tilt_deg=15), notifier=LocalNotifier())
    assert second["reading"]["status"] == "stable"
    assert [r["case_id"] for r in aws_store.readings(1)] == [first["case_id"], second["case_id"]]
    assert [c["case_id"] for c in aws_store.list_cases()] == [second["case_id"], first["case_id"]]
    assert aws_store.get_case(first["case_id"])["trace"] == first["trace"]
    photo_id = first["photos"][0]["photo_id"]
    assert aws_store.load_photo(photo_id, "marked")[:2] == b"\xff\xd8"
    assert aws_store.load_photo("missing") is None
    assert aws_store.photo_url(photo_id, "marked").startswith("https://s3.example/crack-monitor/photos/")


def test_lambda_entry_point(monkeypatch, tmp_path):
    for key, value in {"TABLE_NAME": "t", "BUCKET_NAME": "b", "TOPIC_ARN": "arn:aws:sns:us-east-1:0:x",
                       "ACCESS_CODE": "code", "AWS_DEFAULT_REGION": "us-east-1", "AGENT_MODE": "rules",
                       "MODEL_DIR": str(tmp_path)}.items():
        monkeypatch.setenv(key, value)
    module = importlib.import_module("lambda_function")
    event = {"rawPath": "/", "queryStringParameters": {"code": "code"}, "headers": {"host": "x.lambda-url.aws"},
             "requestContext": {"http": {"method": "GET"}}}
    response = module.lambda_handler(event, None)
    assert response["statusCode"] == 200 and 'id="file"' in response["body"]
    assert json.loads(module.lambda_handler({**event, "rawPath": "/api/photo", "queryStringParameters": None,
                                             "requestContext": {"http": {"method": "POST"}}}, None)["body"])["error"]


def test_lambda_copies_the_models_from_s3(monkeypatch, tmp_path):
    for key, value in {"TABLE_NAME": "t", "BUCKET_NAME": "b", "TOPIC_ARN": "arn:aws:sns:us-east-1:0:x",
                       "AWS_DEFAULT_REGION": "us-east-1", "MODEL_DIR": str(tmp_path)}.items():
        monkeypatch.setenv(key, value)
    module = importlib.import_module("lambda_function")
    monkeypatch.setattr(module, "MODEL_DIR", str(tmp_path))
    fetched = []

    class FakeS3:
        def download_file(self, bucket, key, path):
            fetched.append((bucket, key))
            open(path, "wb").write(b"onnx")

    monkeypatch.setattr(module.boto3, "client", lambda name: FakeS3())
    monkeypatch.setenv("MODELS_PREFIX", "crack-monitor/models/")
    module.fetch_models()
    module.fetch_models()                     # a warm container doesn't download again
    assert sorted(fetched) == [("b", "crack-monitor/models/aliked-n16rot-top1k-640.onnx"),
                               ("b", "crack-monitor/models/aliked_lightglue.onnx")]
    assert sorted(os.listdir(tmp_path)) == ["aliked-n16rot-top1k-640.onnx", "aliked_lightglue.onnx"]
