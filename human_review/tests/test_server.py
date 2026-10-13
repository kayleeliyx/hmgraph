import json
import tempfile
import threading
import unittest
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from server import ReviewServer, byte_range
from store import LabelStore
from test_store import FixtureCatalog, payload


class RangeTests(unittest.TestCase):
    def test_open_closed_suffix_and_clamped_ranges(self):
        self.assertEqual(byte_range(None, 100), (0, 99))
        self.assertEqual(byte_range("bytes=0-9", 100), (0, 9))
        self.assertEqual(byte_range("bytes=90-", 100), (90, 99))
        self.assertEqual(byte_range("bytes=-12", 100), (88, 99))
        self.assertEqual(byte_range("bytes=0-200", 100), (0, 99))

    def test_invalid_and_unsatisfiable_ranges(self):
        for header in ["bytes=100-", "bytes=9-2", "bytes=-0", "bytes=0-1,4-5", "bytes=-", "other=0-4"]:
            with self.subTest(header=header), self.assertRaises(ValueError):
                byte_range(header, 100)


class HTTPTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.catalog = FixtureCatalog()
        media = Path(self.temp.name) / "sample.mp4"
        media.write_bytes(bytes(range(100)))
        self.catalog.media = {"video-token": media}
        self.server = ReviewServer(("127.0.0.1", 0), self.catalog,
                                   LabelStore(Path(self.temp.name) / "labels.sqlite3"))
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        self.temp.cleanup()

    def request(self, path, data=None, headers=None, method=None):
        headers = headers or {}
        if data is not None:
            data = json.dumps(data).encode()
            headers.setdefault("Content-Type", "application/json")
        return urlopen(Request(self.url + path, data=data, headers=headers, method=method), timeout=5)

    def test_bootstrap_examples_and_save_reload_export(self):
        boot = json.load(self.request("/api/bootstrap"))
        headers = {"X-Review-Token": boot["csrf_token"]}
        self.assertEqual(len(json.load(self.request("/api/examples"))["examples"]), 1)
        result = json.load(self.request("/api/labels", payload(), headers))
        self.assertEqual(result["saved"]["revision"], 1)
        detail = json.load(self.request("/api/example?id=example&reviewer=Alice"))
        self.assertEqual(detail["labels"][0]["label"], "correct")
        exported = self.request("/api/export?reviewer=Alice&format=judgments")
        self.assertIn("attachment", exported.headers["Content-Disposition"])
        self.assertEqual(json.loads(exported.read())["judge"], "human:Alice")

    def test_missing_token_and_cross_origin_writes_blocked(self):
        with self.assertRaises(HTTPError) as caught:
            self.request("/api/labels", payload())
        self.assertEqual(caught.exception.code, 403)
        with self.assertRaises(HTTPError) as caught:
            self.request("/api/labels", payload(), {"X-Review-Token": self.server.review_token, "Origin": "https://other.example"})
        self.assertEqual(caught.exception.code, 403)

    def test_media_seek_and_head(self):
        response = self.request("/media/video-token", headers={"Range": "bytes=20-29"})
        self.assertEqual(response.status, 206)
        self.assertEqual(response.read(), bytes(range(20, 30)))
        self.assertEqual(response.headers["Content-Range"], "bytes 20-29/100")
        response = self.request("/media/video-token", method="HEAD")
        self.assertEqual(response.headers["Content-Length"], "100")
        self.assertEqual(response.read(), b"")

    def test_unknown_media_and_traversal_not_served(self):
        for path in ["/media/missing", "/media/..%2Fconfig.json", "/config.json", "/../config.json"]:
            with self.subTest(path=path), self.assertRaises(HTTPError) as caught:
                self.request(path)
            self.assertEqual(caught.exception.code, 404)

    def test_invalid_byte_range_returns_416(self):
        with self.assertRaises(HTTPError) as caught:
            self.request("/media/video-token", headers={"Range": "bytes=1000-"})
        self.assertEqual(caught.exception.code, 416)
        self.assertEqual(caught.exception.headers["Content-Range"], "bytes */100")

    def test_stale_prediction_409_and_invalid_label_400(self):
        headers = {"X-Review-Token": self.server.review_token}
        for updates, status in [({"answer_sha256": "stale"}, 409), ({"label": "invalid"}, 400)]:
            with self.subTest(updates=updates), self.assertRaises(HTTPError) as caught:
                self.request("/api/labels", payload(**updates), headers)
            self.assertEqual(caught.exception.code, status)


if __name__ == "__main__":
    unittest.main()
