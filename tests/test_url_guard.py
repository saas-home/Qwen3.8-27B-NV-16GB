"""The image_url SSRF guard (audit finding F-1): http(s) only, and no
loopback / private / link-local / reserved targets, so a client on the LAN
cannot make the server fetch internal hosts or cloud metadata."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import serve_openai  # noqa: E402


class SafeHttpUrl(unittest.TestCase):
    def _refuses(self, url, needle):
        with self.assertRaises(ValueError) as cm:
            serve_openai._safe_http_url(url)
        self.assertIn(needle, str(cm.exception))

    def test_non_http_schemes_refused(self):
        for u in ("file:///etc/passwd", "gopher://evil/", "ftp://h/x.png", ""):
            self._refuses(u, "http(s)")

    def test_loopback_refused(self):
        self._refuses("http://localhost:8888/v1/models", "private or reserved")
        self._refuses("http://127.0.0.1/x.png", "private or reserved")
        self._refuses("http://[::1]/x.png", "private or reserved")

    def test_cloud_metadata_refused(self):
        self._refuses("http://169.254.169.254/latest/meta-data/", "private or reserved")

    def test_private_ranges_refused(self):
        for u in ("http://10.0.0.9/x.png", "http://203.0.113.50/x.png",
                  "http://203.0.113.43:8888/x.png"):
            self._refuses(u, "private or reserved")

    def test_reserved_and_multicast_refused(self):
        for u in ("http://0.0.0.0/x", "http://224.0.0.1/x", "http://240.0.0.1/x"):
            self._refuses(u, "private or reserved")

    def test_unresolvable_host_refused(self):
        self._refuses("http://no-such-host.invalid/x.png", "does not resolve")

    def test_public_ip_literal_allowed(self):
        # 8.8.8.8 is global; the guard must not break legitimate fetches.
        self.assertEqual(serve_openai._safe_http_url("https://8.8.8.8/x.png"),
                         "https://8.8.8.8/x.png")

    def test_data_urls_bypass_the_guard(self):
        # decode_image handles data: before ever reaching the guard
        import base64
        png = base64.b64encode(b"\x89PNG\r\n\x1a\n").decode()
        # not a real image, so decode_image must fail AFTER the guard is skipped,
        # i.e. with a PIL error, not a ValueError from _safe_http_url
        try:
            serve_openai.decode_image("data:image/png;base64," + png)
        except ValueError as e:
            self.fail(f"data: URL wrongly hit the SSRF guard: {e}")
        except Exception:
            pass  # PIL cannot open the stub - expected


if __name__ == "__main__":
    unittest.main()
