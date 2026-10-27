import base64
import json
import re

import pytest

from app import App
from store import LocalStore
from test_agent import shot

CODE = "secret-code"


class Mailbox:
    def __init__(self):
        self.sent = []

    def notify_engineer(self, subject, body):
        self.sent.append(body)


@pytest.fixture
def app(tmp_path):
    return App(LocalStore(str(tmp_path)), Mailbox(), access_code=CODE)


def call(app, method, path, query=None, body=b"", headers=None, cookies=None):
    """A Lambda Function URL event, like AWS sends it."""
    event = {"rawPath": path, "queryStringParameters": query, "headers": {"host": "example.test", **(headers or {})},
             "requestContext": {"http": {"method": method}}, "cookies": cookies or [],
             "body": base64.b64encode(body).decode(), "isBase64Encoded": True}
    return app.handle(event)


def upload(app, photo, case_id=None):
    response = call(app, "POST", "/api/photo", {"case": case_id} if case_id else None, photo, {"x-access-code": CODE})
    assert response["statusCode"] == 200, response["body"]
    return json.loads(response["body"])["case_id"]


def test_home_asks_for_the_code_and_remembers_it(app):
    assert 'name="code"' in call(app, "GET", "/")["body"]
    response = call(app, "GET", "/", {"code": CODE})
    assert 'id="file"' in response["body"] and 'capture="environment"' in response["body"]
    assert response["headers"]["Set-Cookie"].startswith(f"code={CODE};") and "Secure" in response["headers"]["Set-Cookie"]
    assert 'id="file"' in call(app, "GET", "/", cookies=[f"code={CODE}"])["body"]


def test_upload_needs_the_code(app):
    response = call(app, "POST", "/api/photo", None, shot())
    assert response["statusCode"] == 403


def test_upload_runs_the_agent_and_the_case_page_shows_every_step(app):
    case_id = upload(app, shot())
    body = call(app, "GET", f"/case/{case_id}")["body"]
    for text in ("Crack #1", "Checked the photo", "Measured the crack", "Compared with history", "Saved the reading",
                 "Rules backup", "Max width", "First reading saved"):
        assert text in body
    photo_id = app.store.get_case(case_id)["photos"][0]["photo_id"]
    image = call(app, "GET", f"/photo/{photo_id}/marked")
    assert image["statusCode"] == 200 and image["isBase64Encoded"]
    assert base64.b64decode(image["body"])[:2] == b"\xff\xd8"        # a JPEG


def test_requested_photo_continues_the_same_case(app):
    case_id = upload(app, shot(marker=False))
    page = call(app, "GET", f"/case/{case_id}")["body"]
    assert "Photo needed" in page and "Send the requested photo" in page
    assert upload(app, shot(), case_id) == case_id
    assert app.store.get_case(case_id)["status"] == "closed"


def test_engineer_review_needs_the_token_and_a_button_press(app):
    case_id = upload(app, shot(width_mm=2.5, px_per_mm=6))
    link = app.notifier.sent[0]
    token = re.search(r"token=([\w-]+)", link).group(1)
    assert link.startswith("Crack #1") and f"https://example.test/review/{case_id}?token=" in link
    assert call(app, "GET", f"/review/{case_id}", {"token": "wrong"})["statusCode"] == 403
    page = call(app, "GET", f"/review/{case_id}", {"token": token})["body"]
    assert 'value="inspect"' in page and app.store.get_case(case_id)["status"] == "waiting_for_engineer"
    form = f"token={token}&decision=not_needed&note=%3Cscript%3Ealert(1)%3C%2Fscript%3E".encode()
    assert call(app, "POST", f"/review/{case_id}", body=form)["statusCode"] == 200
    case_page = call(app, "GET", f"/case/{case_id}")["body"]
    assert "no action is needed" in case_page and "<script>alert" not in case_page    # note is escaped
    assert call(app, "POST", f"/review/{case_id}", body=form)["statusCode"] == 403      # works only once


def test_cracks_page_shows_the_history_chart(app):
    assert call(app, "GET", "/cracks")["statusCode"] == 403
    upload(app, shot())
    upload(app, shot(tilt_deg=15))
    body = call(app, "GET", "/cracks", cookies=[f"code={CODE}"])["body"]
    assert "Crack #1" in body and "<svg" in body and "Table view" in body and "alert 2 mm" in body


def test_bad_paths(app):
    assert call(app, "GET", "/nope")["statusCode"] == 404
    assert call(app, "GET", "/case/../../etc")["statusCode"] == 404
    assert call(app, "GET", "/photo/p1/photo")["statusCode"] == 404


def test_health_says_which_alignment_methods_are_loaded(app):
    response = call(app, "GET", "/health")
    body = json.loads(response["body"])
    assert response["statusCode"] == 200 and body["ok"] and body["opencv"].startswith("5.")
    assert body["alignment"][0] == "orb"
