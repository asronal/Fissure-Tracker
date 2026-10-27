"""
Download the OpenCV 5 ALIKED + LightGlue models used by align.py (52 MB, once)

The same two ONNX files, from the same place and with the same checksums, that OpenCV's own tests use
(opencv_extra/testdata/dnn/download_models.py). Without them align.py simply uses ORB only.

Usage:
    python tools/get_models.py
Saves them to models/ (git-ignored). deploy/deploy.py uploads them to S3 for AWS Lambda.
"""

import hashlib
import os
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from align import ALIKED_MODEL, LIGHTGLUE_MODEL, MODEL_DIR  # noqa: E402

SOURCE = "https://raw.githubusercontent.com/YangGuanyuhan/lightglue_opencv_project/main/model/"
SHA1 = {ALIKED_MODEL: "41faa7bf5d7eb68a2851471ba03aa20c9db30e4c",       # 6 MB
        LIGHTGLUE_MODEL: "02723aa521990e57fe33d90b67977590c460e351"}    # 46 MB


def sha1(path):
    return hashlib.sha1(open(path, "rb").read()).hexdigest()


if __name__ == "__main__":
    os.makedirs(MODEL_DIR, exist_ok=True)
    for name, expected in SHA1.items():
        path = os.path.join(MODEL_DIR, name)
        if os.path.exists(path) and sha1(path) == expected:
            print(f"{name}: already there")
            continue
        print(f"{name}: downloading...")
        urllib.request.urlretrieve(SOURCE + name, path + ".part")
        if sha1(path + ".part") != expected:
            os.remove(path + ".part")
            raise SystemExit(f"{name}: checksum does not match - not saved")
        os.replace(path + ".part", path)
        print(f"{name}: ok ({os.path.getsize(path) / 1e6:.0f} MB)")
