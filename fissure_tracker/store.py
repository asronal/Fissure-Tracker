"""
Where cases, readings and photos are saved.

A case    = one conversation about one crack: the photos the owner sent, what the agent did (trace), status.
A reading = one accepted measurement of a crack (marker), used as history for the next photo.

LocalStore - files in outputs/store/ (laptop + tests, no AWS needed)
AwsStore   - DynamoDB table + S3 bucket (on AWS Lambda)
"""

import json
import os
import secrets
from datetime import datetime, timezone


def now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def new_id(prefix):
    """Sortable, unguessable id, e.g. c20261023T163000123-3f9a1c (time to the millisecond + random)."""
    return f"{prefix}{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')[:-3]}-{secrets.token_hex(3)}"


class LocalStore:
    def __init__(self, folder):
        self.folder = folder
        for sub in ("photos", "cases"):
            os.makedirs(os.path.join(folder, sub), exist_ok=True)
        self.readings_file = os.path.join(folder, "readings.jsonl")

    # --- photos (kind: "photo" = what the owner sent, "marked" = with the crack drawn in, "matches") ---
    def _photo_path(self, photo_id, kind):
        return os.path.join(self.folder, "photos", f"{photo_id}_{kind}.jpg")

    def save_photo(self, photo_id, jpeg, kind="photo"):
        with open(self._photo_path(photo_id, kind), "wb") as f:
            f.write(jpeg)

    def load_photo(self, photo_id, kind="photo"):
        path = self._photo_path(photo_id, kind)
        if not os.path.exists(path):
            return None
        with open(path, "rb") as f:
            return f.read()

    def photo_url(self, photo_id, kind="photo"):
        return f"/photo/{photo_id}/{kind}"

    # --- cases ---
    def put_case(self, case):
        with open(os.path.join(self.folder, "cases", f"{case['case_id']}.json"), "w", encoding="utf-8") as f:
            json.dump(case, f, indent=1)

    def get_case(self, case_id):
        path = os.path.join(self.folder, "cases", f"{os.path.basename(case_id)}.json")
        if not os.path.exists(path):
            return None
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def list_cases(self, limit=100):
        names = sorted(os.listdir(os.path.join(self.folder, "cases")), reverse=True)[:limit]
        return [self.get_case(n[:-5]) for n in names if n.endswith(".json")]

    # --- readings ---
    def add_reading(self, reading):
        with open(self.readings_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(reading) + "\n")

    def readings(self, marker_id=None):
        """Oldest first."""
        if not os.path.exists(self.readings_file):
            return []
        with open(self.readings_file, encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
        rows = [r for r in rows if marker_id is None or r["marker_id"] == marker_id]
        return sorted(rows, key=lambda r: r["time"])


class AwsStore:
    """
    One DynamoDB table with string keys pk + sk (data kept as JSON text, so floats need no conversion):
        cases:    pk = "case",    sk = case_id               (case ids sort by time)
        readings: pk = "reading", sk = "<time>#<photo_id>"
    Photos in S3 under <prefix>photos/<photo_id>_<kind>.jpg; pages show them through short-lived links.
    """

    def __init__(self, table_name, bucket, prefix="crack-monitor/", region=None):
        import boto3  # only needed on AWS

        self.table = boto3.resource("dynamodb", region_name=region).Table(table_name)
        self.s3 = boto3.client("s3", region_name=region)
        self.bucket, self.prefix = bucket, prefix

    def _key(self, photo_id, kind):
        return f"{self.prefix}photos/{photo_id}_{kind}.jpg"

    def save_photo(self, photo_id, jpeg, kind="photo"):
        self.s3.put_object(Bucket=self.bucket, Key=self._key(photo_id, kind), Body=jpeg, ContentType="image/jpeg")

    def load_photo(self, photo_id, kind="photo"):
        try:
            return self.s3.get_object(Bucket=self.bucket, Key=self._key(photo_id, kind))["Body"].read()
        except self.s3.exceptions.NoSuchKey:
            return None

    def photo_url(self, photo_id, kind="photo", expires=3600):
        return self.s3.generate_presigned_url("get_object", ExpiresIn=expires,
                                              Params={"Bucket": self.bucket, "Key": self._key(photo_id, kind)})

    def put_case(self, case):
        self.table.put_item(Item={"pk": "case", "sk": case["case_id"], "data": json.dumps(case)})

    def get_case(self, case_id):
        item = self.table.get_item(Key={"pk": "case", "sk": case_id}).get("Item")
        return json.loads(item["data"]) if item else None

    def _query(self, pk, newest_first=False, limit=None):
        from boto3.dynamodb.conditions import Key

        kwargs = {"KeyConditionExpression": Key("pk").eq(pk), "ScanIndexForward": not newest_first}
        items = []
        while True:
            if limit:
                kwargs["Limit"] = limit - len(items)
            resp = self.table.query(**kwargs)
            items.extend(resp["Items"])
            if "LastEvaluatedKey" not in resp or (limit and len(items) >= limit):
                return items
            kwargs["ExclusiveStartKey"] = resp["LastEvaluatedKey"]

    def list_cases(self, limit=100):
        return [json.loads(i["data"]) for i in self._query("case", newest_first=True, limit=limit)]

    def add_reading(self, reading):
        self.table.put_item(Item={"pk": "reading", "sk": f"{reading['time']}#{reading['photo_id']}",
                                  "data": json.dumps(reading)})

    def readings(self, marker_id=None):
        """Oldest first. A hackathon-sized table: reading them all is fine."""
        rows = [json.loads(i["data"]) for i in self._query("reading")]
        return [r for r in rows if marker_id is None or r["marker_id"] == marker_id]
